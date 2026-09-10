#!/usr/bin/env python3
"""
market_breadth.py  --  Overall Market Breadth Dashboard  (interactive)
======================================================================

Pulls live daily prices and computes classic market-breadth indicators for the
S&P 500 (large cap), NASDAQ Composite (tech/growth), and S&P SmallCap 600 (small
cap), blends them into a single 0-100 "Invested-o-meter" per index and overall,
adds a risk-context strip (VIX, high-yield credit spreads, equal- vs cap-weight)
and a GICS sector-breadth heatmap, and renders an INTERACTIVE, self-contained
dashboard.html (hover tooltips + a hero chart you can toggle between indices)
you open in any browser. Logs each run to a CSV and can email itself daily.

Breadth answers the question price alone cannot: "Is the whole market
participating, or is the index being carried by a handful of names?"
Broad participation = healthier trend = a better time to be invested.

INDICATORS (per index)
  * % of stocks above their 20 / 50 / 200-day moving average
  * Advance-Decline line (cumulative advancers minus decliners)
  * Net new 52-week highs minus new lows (% of universe)
  * McClellan Oscillator (EMA19 - EMA39 of ratio-adjusted net advances)
  * McClellan Summation Index (running total of the oscillator)
  * Zweig Breadth-Thrust watch (rare "get invested" launch signal)

COMPOSITE (0 risk-off .. 100 risk-on):
  >=70 RISK-ON | 58-70 CONSTRUCTIVE | 45-58 NEUTRAL | 32-45 CAUTION | <32 RISK-OFF

USAGE
  pip install -r requirements.txt
  python market_breadth.py                      # full run (Yahoo, no key)
  python market_breadth.py --sample 150         # faster sample
  python market_breadth.py --source tiingo      # official keyed feed (TIINGO_API_KEY)
  python market_breadth.py --demo               # synthetic data, no network
  python market_breadth.py --email you@x.com    # email it (BREADTH_SMTP_* env)

Data: Yahoo Finance (yfinance) or Tiingo, + FRED. No key required for Yahoo.
Not investment advice — a decision-support tool.
"""

from __future__ import annotations

import argparse, io, json, math, os, time, pickle, datetime as dt
import webbrowser, urllib.request, urllib.error
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, ".cache")
OUT_HTML = os.path.join(HERE, "dashboard.html")


# ============================================================================
# 1.  UNIVERSE
# ============================================================================

def _clean(sym): return str(sym).strip().upper().replace(".", "-")

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
      "Accept": "text/html,application/xhtml+xml,text/csv,*/*;q=0.8",
      "Accept-Language": "en-US,en;q=0.9"}
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_SP600 = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
# Backup S&P 500 list (GitHub-hosted, always reachable from servers)
SP500_FALLBACK = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def _fetch_text(url, timeout=60):
    """GET a URL with a browser-like User-Agent (Wikipedia 403s bare requests)."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _wiki_symbols(url, cols=("Symbol", "Ticker", "Ticker symbol")):
    for tbl in pd.read_html(io.StringIO(_fetch_text(url))):
        for c in cols:
            if c in tbl.columns:
                return sorted({_clean(s) for s in tbl[c].tolist() if isinstance(s, str)})
    raise RuntimeError(f"No symbol column at {url}")


def _sp500_fallback_df():
    return pd.read_csv(io.StringIO(_fetch_text(SP500_FALLBACK)))


def fetch_sp500_tickers():
    try:
        return _wiki_symbols(WIKI_SP500)
    except Exception as e:
        print(f"    warn: Wikipedia S&P 500 list failed ({e}); using backup list")
        df = _sp500_fallback_df()
        return sorted({_clean(s) for s in df["Symbol"].tolist() if isinstance(s, str)})


def fetch_sp600_tickers():
    return _wiki_symbols(WIKI_SP600)


def fetch_sp500_sectors():
    for source in ("wiki", "fallback"):
        try:
            if source == "wiki":
                df = pd.read_html(io.StringIO(_fetch_text(WIKI_SP500)))[0]
            else:
                df = _sp500_fallback_df()
            sym = "Symbol" if "Symbol" in df.columns else df.columns[0]
            sec = next((c for c in df.columns if "Sector" in str(c)), None)
            if sec is None:
                continue
            return {_clean(s): str(x) for s, x in zip(df[sym], df[sec]) if isinstance(s, str)}
        except Exception as e:
            print(f"    warn: sector map via {source} failed ({e})")
    return {}


def fetch_nasdaq_tickers():
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw), sep="|")
    df = df[df["Symbol"].notna()]
    if "ETF" in df.columns:
        df = df[df["ETF"] != "Y"]
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] != "Y"]
    out = []
    for s in df["Symbol"].tolist():
        s = str(s).strip().upper()
        if not s or any(c in s for c in "$.+ ") or len(s) > 5:
            continue
        out.append(s)
    return sorted(set(out))


INDEX_SPECS = [
    {"key": "sp500",  "name": "S&P 500",             "sub": "Large cap",
     "color": "var(--series-1)", "fetch": fetch_sp500_tickers},
    {"key": "nasdaq", "name": "NASDAQ Composite",    "sub": "Tech / growth",
     "color": "var(--series-2)", "fetch": fetch_nasdaq_tickers},
    {"key": "sp600",  "name": "Small Caps (S&P 600)", "sub": "Small cap",
     "color": "var(--series-3)", "fetch": fetch_sp600_tickers},
]


# ============================================================================
# 2.  DATA
# ============================================================================

def _cache_path(tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{tag}_{dt.date.today().isoformat()}.pkl")


def _yf_closes(data, batch):
    if data is None or len(data) == 0:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = set(data.columns.get_level_values(0))
        return data["Close"] if "Close" in lvl0 else data.xs("Close", axis=1, level=-1)
    if "Close" not in data.columns:
        return None
    out = data[["Close"]].copy()
    out.columns = batch[:1]
    return out


def _tvdatafeed_closes(tickers, tag, tv_user, tv_pass, exchanges, period_days=730, pause=0.0):
    """
    Pull daily closes from TradingView via the UNOFFICIAL `tvdatafeed` library.
    Caveats: unofficial (reverse-engineered), needs a TradingView login, is
    against TradingView's ToS, and is slow (one websocket call per symbol) — use
    --sample. tvdatafeed must be installed:  pip install tvdatafeed
    (or the maintained fork: pip install git+https://github.com/rongardF/tvdatafeed).
    """
    try:
        from tvDatafeed import TvDatafeed, Interval
    except Exception:
        raise RuntimeError("tvdatafeed not installed (it is NOT on PyPI). Run: "
                           'pip install "git+https://github.com/rongardF/tvdatafeed.git"')
    tv = TvDatafeed(tv_user, tv_pass) if tv_user else TvDatafeed()
    if not tv_user:
        print("    note: no TV_USERNAME/TV_PASSWORD set — using anonymous (heavily limited).")
    n_bars = min(5000, int(period_days / 7 * 5) + 60)
    series, ok, fail, n = {}, 0, 0, len(tickers)
    for i, tk in enumerate(tickers):
        got = None
        for exch in exchanges:
            try:
                df = tv.get_hist(symbol=tk, exchange=exch, interval=Interval.in_daily, n_bars=n_bars)
                if df is not None and len(df):
                    got = df["close"].copy()
                    idx = pd.to_datetime(got.index)
                    if getattr(idx, "tz", None) is not None:
                        idx = idx.tz_localize(None)
                    got.index = idx.normalize()
                    break
            except Exception:
                pass
        if got is not None:
            series[tk] = got; ok += 1
        else:
            fail += 1
        if (i + 1) % 50 == 0:
            print(f"  tradingview {tag}: {i+1}/{n} ({ok} ok, {fail} skipped) ...", flush=True)
        if pause:
            time.sleep(pause)
    if not series:
        raise RuntimeError(f"TradingView returned no data for {tag}. "
                           "Check TV_USERNAME/TV_PASSWORD, symbol/exchange, and that tvdatafeed works.")
    df = pd.concat(series, axis=1).sort_index()
    return df.dropna(axis=1, thresh=int(len(df) * 0.6))


def _tiingo_closes(tickers, tag, key, period_days=730, pause=0.15):
    if not key:
        raise RuntimeError("Tiingo selected but no key. Set TIINGO_API_KEY or --tiingo-key.")
    start = (dt.date.today() - dt.timedelta(days=period_days)).isoformat()
    series, ok, fail, n = {}, 0, 0, len(tickers)
    for i, tk in enumerate(tickers):
        url = (f"https://api.tiingo.com/tiingo/daily/{tk.lower()}/prices"
               f"?startDate={start}&token={key}&format=json")
        try:
            req = urllib.request.Request(url, headers={
                "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            if data:
                series[tk] = pd.Series(
                    {pd.Timestamp(d["date"][:10]): d.get("adjClose", d.get("close")) for d in data})
                ok += 1
        except urllib.error.HTTPError as e:
            fail += 1
            if e.code == 429:
                print(f"    Tiingo rate limit (429) after {ok} — using partial. Try --sample or paid tier.")
                break
        except Exception:
            fail += 1
        if (i + 1) % 100 == 0:
            print(f"  tiingo {tag}: {i+1}/{n} ({ok} ok) ...", flush=True)
        time.sleep(pause)
    if not series:
        raise RuntimeError(f"Tiingo returned no data for {tag}.")
    df = pd.concat(series, axis=1).sort_index()
    return df.dropna(axis=1, thresh=int(len(df) * 0.6))


TV_EXCHANGES = {"sp500": ["NYSE", "NASDAQ", "AMEX"],
                "nasdaq": ["NASDAQ", "NYSE", "AMEX"],
                "sp600": ["NYSE", "NASDAQ", "AMEX"]}


def download_closes(tickers, tag, refresh=False, period="2y", source="yahoo",
                    tiingo_key=None, tv_user=None, tv_pass=None):
    cp = _cache_path(f"{tag}_{source}")
    if os.path.exists(cp) and not refresh:
        try:
            with open(cp, "rb") as f:
                df = pickle.load(f)
            print(f"  [cache] {tag} ({source}): {df.shape[1]} names, {df.shape[0]} days")
            return df
        except Exception:
            pass
    period_days = int(period[:-1]) * 365 if period.endswith("y") and period[:-1].isdigit() else 730
    if source == "tiingo":
        df = _tiingo_closes(tickers, tag, tiingo_key, period_days=period_days)
    elif source == "tradingview":
        df = _tvdatafeed_closes(tickers, tag, tv_user, tv_pass,
                                TV_EXCHANGES.get(tag, ["NYSE", "NASDAQ", "AMEX"]), period_days=period_days)
    else:
        import yfinance as yf
        frames, chunk, n = [], 120, len(tickers)
        for i in range(0, n, chunk):
            batch = tickers[i:i + chunk]
            print(f"  downloading {tag} {i+1}-{min(i+chunk, n)}/{n} ...", flush=True)
            data = None
            for attempt in range(3):
                try:
                    data = yf.download(batch, period=period, interval="1d",
                                       auto_adjust=True, progress=False, threads=True)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"    warn: batch failed ({e})")
                    else:
                        time.sleep(2 * (attempt + 1))
            c = _yf_closes(data, batch)
            if c is not None:
                frames.append(c)
        if not frames:
            raise RuntimeError(f"No data for {tag}. Check your network.")
        df = pd.concat(frames, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(axis=1, thresh=int(len(df) * 0.6))
    with open(cp, "wb") as f:
        pickle.dump(df, f)
    print(f"  [ok] {tag} ({source}): {df.shape[1]} names, {df.shape[0]} days")
    return df


def download_series(ticker, tag, refresh=False, period="3y"):
    cp = _cache_path(f"s_{tag}")
    if os.path.exists(cp) and not refresh:
        try:
            with open(cp, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        import yfinance as yf
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        s = _yf_closes(data, [ticker])
        if s is None:
            return None
        s = s.iloc[:, 0] if hasattr(s, "columns") else s
        s.name = tag
        with open(cp, "wb") as f:
            pickle.dump(s, f)
        return s
    except Exception as e:
        print(f"    warn: {tag} fetch failed ({e})")
        return None


def fetch_fred_series(series_id, tag, refresh=False):
    cp = _cache_path(f"fred_{tag}")
    if os.path.exists(cp) and not refresh:
        try:
            with open(cp, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=45) as r:
            df = pd.read_csv(io.StringIO(r.read().decode()))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"]), name=tag)
        with open(cp, "wb") as f:
            pickle.dump(s, f)
        return s
    except Exception as e:
        print(f"    warn: FRED {tag} failed ({e})")
        return None


# ============================================================================
# 3.  BREADTH MATH
# ============================================================================

def clamp(x, lo=0.0, hi=100.0): return np.clip(x, lo, hi)


@dataclass
class Breadth:
    name: str; sub: str; n: int
    pct_above_20: float; pct_above_50: float; pct_above_200: float
    advancers: int; decliners: int
    net_new_highs: float; mcclellan_osc: float; mcclellan_sum: float
    thrust_ratio: float; thrust_signal: bool
    score: float = 0.0; regime: str = ""
    key: str = ""; color: str = "var(--series-1)"
    series_pct200: list = field(default_factory=list)
    series_pct50: list = field(default_factory=list)
    series_adline: list = field(default_factory=list)
    series_mcsum: list = field(default_factory=list)
    series_mcosc: list = field(default_factory=list)
    series_nhnl: list = field(default_factory=list)
    score_ts: pd.Series = None
    components: dict = field(default_factory=dict)


def _ser(idx, values):
    out = []
    for t, v in zip(idx, values):
        vv = None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 3)
        out.append({"t": pd.Timestamp(t).strftime("%Y-%m-%d"), "v": vv})
    return out


def composite_series(pct20, pct50, pct200, mcosc, nhnl, adline, vc):
    s_trend, s_inter, s_short = clamp(pct200), clamp(pct50), clamp(pct20)
    s_mom = clamp(50 + mcosc * 0.5)
    s_lead = clamp(50 + nhnl * 5.0)
    s_ad = clamp(50 + (adline - adline.shift(20)) / vc.replace(0, np.nan) * 100 * 0.5)
    return clamp(0.30 * s_trend + 0.20 * s_inter + 0.20 * s_mom +
                 0.15 * s_lead + 0.15 * s_ad + (s_short - 50) * 0.06)


def compute_breadth(closes, name, sub):
    closes = closes.sort_index()
    valid = closes.notna(); vc = valid.sum(axis=1)
    pct20 = ((closes > closes.rolling(20).mean()) & valid).sum(axis=1) / vc * 100
    pct50 = ((closes > closes.rolling(50).mean()) & valid).sum(axis=1) / vc * 100
    pct200 = ((closes > closes.rolling(200).mean()) & valid).sum(axis=1) / vc * 100
    chg = closes.diff()
    adv, dec = (chg > 0).sum(axis=1), (chg < 0).sum(axis=1)
    net_adv = adv - dec; ad_line = net_adv.cumsum()
    hi = closes.rolling(252, min_periods=60).max(); lo = closes.rolling(252, min_periods=60).min()
    nh = ((closes >= hi) & valid).sum(axis=1); nl = ((closes <= lo) & valid).sum(axis=1)
    net_nhnl = (nh - nl) / vc.replace(0, np.nan) * 100
    rana = (net_adv / (adv + dec).replace(0, np.nan)) * 1000
    mcosc = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
    mcsum = mcosc.cumsum()
    thrust = (adv / (adv + dec).replace(0, np.nan)).ewm(span=10, adjust=False).mean()
    tv = thrust.dropna().values; thr = False
    if len(tv) > 12:
        for j in range(len(tv) - 1, max(len(tv) - 11, 9), -1):
            if tv[max(0, j - 10):j + 1].min() < 0.40 and tv[j] > 0.615:
                thr = True; break
    score_ts = composite_series(pct20, pct50, pct200, mcosc, net_nhnl, ad_line, vc)
    L = -1
    b = Breadth(name=name, sub=sub, n=int(vc.iloc[L]),
                pct_above_20=float(pct20.iloc[L]), pct_above_50=float(pct50.iloc[L]),
                pct_above_200=float(pct200.iloc[L]),
                advancers=int(adv.iloc[L]), decliners=int(dec.iloc[L]),
                net_new_highs=float(net_nhnl.iloc[L]),
                mcclellan_osc=float(mcosc.iloc[L]), mcclellan_sum=float(mcsum.iloc[L]),
                thrust_ratio=float(thrust.iloc[L]), thrust_signal=bool(thr))
    t = 252; idx = closes.index[-t:]
    b.series_pct200 = _ser(idx, pct200.iloc[-t:].values)
    b.series_pct50 = _ser(idx, pct50.iloc[-t:].values)
    b.series_adline = _ser(idx, ad_line.iloc[-t:].values)
    b.series_mcsum = _ser(idx, mcsum.iloc[-t:].values)
    b.series_mcosc = _ser(idx, mcosc.iloc[-t:].values)
    b.series_nhnl = _ser(idx, net_nhnl.iloc[-t:].values)
    b.score_ts = score_ts.dropna()
    b.score = round(float(score_ts.iloc[L]), 1)
    if b.thrust_signal:
        b.score = round(min(100, b.score + 8), 1)
    b.regime = regime_for(b.score)
    b.components = {
        "Trend (>200d MA)": round(float(clamp(b.pct_above_200)), 0),
        "Intermediate (>50d MA)": round(float(clamp(b.pct_above_50)), 0),
        "Momentum (McClellan)": round(float(clamp(50 + b.mcclellan_osc * 0.5)), 0),
        "Leadership (NH-NL)": round(float(clamp(50 + b.net_new_highs * 5)), 0),
        "Short-term (>20d MA)": round(float(clamp(b.pct_above_20)), 0)}
    return b


def compute_sector_breadth(closes, sectmap):
    from collections import defaultdict
    closes = closes.sort_index(); valid = closes.notna()
    a50 = closes > closes.rolling(50).mean(); a200 = closes > closes.rolling(200).mean()
    groups = defaultdict(list)
    for t in closes.columns:
        s = sectmap.get(t)
        if s and s.lower() != "nan":
            groups[s].append(t)
    rows = []
    for sec, tks in groups.items():
        v = valid[tks].iloc[-1]; cnt = int(v.sum())
        if cnt == 0:
            continue
        rows.append({"name": sec, "n": cnt,
                     "p50": float((a50[tks].iloc[-1] & v).sum() / cnt * 100),
                     "p200": float((a200[tks].iloc[-1] & v).sum() / cnt * 100)})
    rows.sort(key=lambda r: -r["p200"])
    return rows


REGIME_ORDER = ["RISK-ON", "CONSTRUCTIVE", "NEUTRAL", "CAUTION", "RISK-OFF"]
HORIZONS = [(21, "1-month"), (63, "3-month"), (126, "6-month")]


def compute_backtest(history):
    """How did the S&P 500 actually do after each composite regime?
    Buckets every historical day by regime, then measures forward S&P returns."""
    comp = history["series"]["Overall"]; price = history["price"]; n = len(comp)
    occ = {r: 0 for r in REGIME_ORDER}
    fwd = {h: {r: [] for r in REGIME_ORDER} for h, _ in HORIZONS}
    base = {h: [] for h, _ in HORIZONS}
    total = 0
    for i in range(n):
        c = comp[i]
        if c is None:
            continue
        r = regime_for(c); occ[r] += 1; total += 1
        for h, _ in HORIZONS:
            if i + h < n and price[i] and price[i + h]:
                ret = price[i + h] / price[i] - 1
                fwd[h][r].append(ret); base[h].append(ret)
    out = {"order": REGIME_ORDER, "labels": [lbl for _, lbl in HORIZONS],
           "occupancy": {}, "cells": {}, "baseline": {}, "total": total}
    for h, lbl in HORIZONS:
        arr = base[h]
        out["baseline"][lbl] = ({"n": len(arr), "mean": sum(arr) / len(arr) * 100,
                                 "hit": sum(x > 0 for x in arr) / len(arr) * 100} if arr else None)
    for r in REGIME_ORDER:
        out["occupancy"][r] = (occ[r] / total * 100) if total else 0
        out["cells"][r] = {}
        for h, lbl in HORIZONS:
            arr = fwd[h][r]
            out["cells"][r][lbl] = ({"n": len(arr), "mean": sum(arr) / len(arr) * 100,
                                     "hit": sum(x > 0 for x in arr) / len(arr) * 100} if arr else None)
    return out


def regime_for(s):
    return ("RISK-ON" if s >= 70 else "CONSTRUCTIVE" if s >= 58 else
            "NEUTRAL" if s >= 45 else "CAUTION" if s >= 32 else "RISK-OFF")


REGIME_META = {
    "RISK-ON":      ("good",     "Broad participation. Internals healthy — a favorable backdrop to be more invested."),
    "CONSTRUCTIVE": ("good",     "Solid participation. Trend supported by breadth — stay invested, add on dips."),
    "NEUTRAL":      ("warning",  "Mixed internals. Neither confirming nor breaking down — keep normal exposure, wait for direction."),
    "CAUTION":      ("serious",  "Breadth weakening. Fewer stocks holding up — trim risk, tighten stops, raise some cash."),
    "RISK-OFF":     ("critical", "Broad deterioration. Most stocks below trend — defensive posture, protect capital.")}


# ============================================================================
# 4.  RISK CONTEXT, LOGGING, EMAIL
# ============================================================================

@dataclass
class RiskCtx:
    vix: float = None; vix_role: str = ""; vix_series: list = field(default_factory=list)
    hy: float = None; hy_role: str = ""; hy_series: list = field(default_factory=list)
    hy_proxy: bool = False; hy_label: str = "BAML HY OAS · rising = risk-off"
    ewcw: float = None; ewcw_role: str = ""; ewcw_series: list = field(default_factory=list)


def build_risk(refresh=False, demo=False):
    r = RiskCtx()
    if demo:
        rng = np.random.default_rng(7); d = pd.bdate_range(end="2026-08-28", periods=252)
        vix = 14 + np.cumsum(rng.normal(0, 0.6, 252)).clip(-6, 20)
        r.vix_series = _ser(d, vix); r.vix = float(vix[-1])
        hy = 3.2 + np.cumsum(rng.normal(0, 0.02, 252)).clip(-1, 3)
        r.hy_series = _ser(d, hy); r.hy = float(hy[-1])
        rr = pd.Series(np.cumprod(1 + rng.normal(0, 0.004, 252)), index=d)
        r.ewcw_series = _ser(d, (rr / rr.iloc[0]).values)
        r.ewcw = float((rr.iloc[-1] / rr.iloc[-51] - 1) * 100)
    else:
        print("Fetching risk context (VIX, credit spreads, equal-weight) ...")
        vix = download_series("^VIX", "vix", refresh)
        if vix is not None:
            v = vix.dropna().iloc[-252:]; r.vix = float(v.iloc[-1]); r.vix_series = _ser(v.index, v.values)
        hy = fetch_fred_series("BAMLH0A0HYM2", "hyoas", refresh)
        if hy is not None:
            h = hy.dropna().iloc[-252:]; r.hy = float(h.iloc[-1]); r.hy_series = _ser(h.index, h.values)
        else:
            # Fallback: HYG/IEF ratio (high-yield vs treasuries) as a credit-stress proxy.
            # Rising ratio = spreads tightening (risk-on); falling = widening (risk-off).
            print("    FRED unavailable — using HYG/IEF credit proxy")
            hyg, ief = download_series("HYG", "hyg", refresh), download_series("IEF", "ief", refresh)
            if hyg is not None and ief is not None:
                ratio = (hyg / ief).dropna().iloc[-252:]
                if len(ratio) > 51:
                    r.hy = float((ratio.iloc[-1] / ratio.iloc[-51] - 1) * 100)
                    r.hy_series = _ser(ratio.index, (ratio / ratio.iloc[0]).values)
                    r.hy_proxy = True
                    r.hy_label = "HYG/IEF credit proxy, 50-day · falling = risk-off"
        rsp, spy = download_series("RSP", "rsp", refresh), download_series("SPY", "spy", refresh)
        if rsp is not None and spy is not None:
            ratio = (rsp / spy).dropna().iloc[-252:]
            if len(ratio) > 51:
                r.ewcw = float((ratio.iloc[-1] / ratio.iloc[-51] - 1) * 100)
                r.ewcw_series = _ser(ratio.index, (ratio / ratio.iloc[0]).values)
    if r.vix is not None:
        r.vix_role = "good" if r.vix < 15 else "warning" if r.vix < 20 else "serious" if r.vix < 30 else "critical"
    if r.hy is not None:
        if r.hy_proxy:   # % change of HYG/IEF: up = good
            r.hy_role = "good" if r.hy > 1 else "warning" if r.hy > -1 else "serious" if r.hy > -3 else "critical"
        else:            # OAS spread level in %: low = good
            r.hy_role = "good" if r.hy < 3.5 else "warning" if r.hy < 5 else "serious" if r.hy < 7 else "critical"
    if r.ewcw is not None:
        r.ewcw_role = "good" if r.ewcw > 1 else "warning" if r.ewcw > -1 else "serious" if r.ewcw > -4 else "critical"
    return r


def log_history(indexes, overall, risk, csv_path):
    today = dt.date.today().isoformat()
    row = {"date": today, "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "overall": overall}
    for b in indexes:
        k = b.name.split("(")[0].strip().replace("&", "").replace(" ", "_").lower()
        row[f"{k}_score"] = b.score; row[f"{k}_pct200"] = round(b.pct_above_200, 1)
    row["vix"] = round(risk.vix, 2) if risk.vix is not None else ""
    row["hy_spread"] = round(risk.hy, 2) if risk.hy is not None else ""
    row["ew_cw_50d"] = round(risk.ewcw, 2) if risk.ewcw is not None else ""
    try:
        ex = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        if not ex.empty:
            ex = ex[ex["date"] != today]
        pd.concat([ex, pd.DataFrame([row])], ignore_index=True).to_csv(csv_path, index=False)
        print(f"  logged -> {csv_path}")
    except Exception as e:
        print(f"  warn: history log ({e})")


def send_email(to, html_path, subject, summary):
    import smtplib, ssl
    from email.message import EmailMessage
    host = os.environ.get("BREADTH_SMTP_HOST")
    if not host:
        print("  email: set BREADTH_SMTP_HOST/_PORT/_USER/_PASS to enable."); return
    port = int(os.environ.get("BREADTH_SMTP_PORT", "465"))
    user, pw = os.environ.get("BREADTH_SMTP_USER", ""), os.environ.get("BREADTH_SMTP_PASS", "")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, os.environ.get("BREADTH_EMAIL_FROM", user), to
    msg.set_content(summary)
    with open(html_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="text", subtype="html", filename="dashboard.html")
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as s:
                if user: s.login(user, pw)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port) as s:
                s.starttls(context=ssl.create_default_context())
                if user: s.login(user, pw)
                s.send_message(msg)
        print(f"  emailed -> {to}")
    except Exception as e:
        print(f"  warn: email failed ({e})")


# ============================================================================
# 5.  DEMO DATA
# ============================================================================

def demo_closes(n, seed, drift):
    rng = np.random.default_rng(seed); days = 620
    d = pd.bdate_range(end="2026-08-28", periods=days)
    base = rng.uniform(20, 200, n); mkt = np.cumsum(rng.normal(drift, 0.011, days))
    mkt[300:360] -= np.linspace(0, 0.28, 60); mkt[360:] -= 0.28
    cols = {}
    for k in range(n):
        beta = rng.uniform(0.4, 1.6); idio = np.cumsum(rng.normal(0, 0.018, days))
        cols[f"N{k:04d}"] = base[k] * np.exp(beta * mkt + idio)
    return pd.DataFrame(cols, index=d)


def demo_price(seed=11):
    rng = np.random.default_rng(seed); days = 620
    d = pd.bdate_range(end="2026-08-28", periods=days)
    r = rng.normal(0.0007, 0.01, days); r[300:360] -= 0.004
    return pd.Series(4200 * np.cumprod(1 + r), index=d)


# ============================================================================
# 6.  RENDER
# ============================================================================

def _sv(role):
    return {"good": "var(--st-good)", "warning": "var(--st-warning)",
            "serious": "var(--st-serious)", "critical": "var(--st-critical)"}[role]


def _score_color(v): return _sv(REGIME_META[regime_for(v)][0])
def _heat(v):
    return ("var(--st-good)" if v >= 70 else "var(--st-good-dim)" if v >= 55 else
            "var(--st-warning)" if v >= 45 else "var(--st-serious)" if v >= 30 else "var(--st-critical)")
def _role_pct(v): return "good" if v >= 60 else "warning" if v >= 45 else "serious" if v >= 30 else "critical"
def _role_nhnl(v): return "good" if v >= 1 else "warning" if v >= -1 else "serious" if v >= -4 else "critical"
def _role_mcosc(v): return "good" if v >= 25 else "warning" if v >= 0 else "serious" if v >= -50 else "critical"


def _spark(series, w=240, h=44, color="var(--series-1)", zero=False, band=None):
    pts = [p for p in series if p.get("v") is not None]
    if len(pts) < 2:
        return '<svg viewBox="0 0 240 44" class="spark"></svg>'
    ys = [p["v"] for p in pts]; ymin, ymax = min(ys), max(ys)
    if zero: ymin, ymax = min(ymin, 0), max(ymax, 0)
    if band: ymin, ymax = band
    if ymax == ymin: ymax = ymin + 1
    m = len(pts); pad = 4
    X = lambda i: pad + i / (m - 1) * (w - 2 * pad)
    Y = lambda v: h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad)
    p = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
    out = [f'<polygon points="{X(0):.1f},{h-pad:.1f} {p} {X(m-1):.1f},{h-pad:.1f}" fill="var(--fill-1)"/>']
    if zero or (band and band[0] < 0 < band[1]):
        out.append(f'<line x1="{pad}" y1="{Y(0):.1f}" x2="{w-pad}" y2="{Y(0):.1f}" stroke="var(--gridline)" stroke-width="1"/>')
    out.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
    out.append(f'<circle cx="{X(m-1):.1f}" cy="{Y(ys[-1]):.1f}" r="2.6" fill="{color}"/>')
    js = json.dumps(pts, separators=(",", ":"))
    return (f'<div class="sparkwrap" data-series=\'{js}\'>'
            f'<svg viewBox="0 0 {w} {h}" class="spark" preserveAspectRatio="none">{"".join(out)}</svg></div>')


def _gauge(score, size=240):
    cx, cy, r = size / 2, size * 0.58, size * 0.42
    pt = lambda f: (cx + r * math.cos(math.pi * (1 - f)), cy - r * math.sin(math.pi * (1 - f)))
    bands = [(0, .32, "var(--st-critical)"), (.32, .45, "var(--st-serious)"), (.45, .58, "var(--st-warning)"),
             (.58, .70, "var(--st-good-dim)"), (.70, 1, "var(--st-good)")]
    seg = ""
    for a, b, c in bands:
        x1, y1 = pt(a); x2, y2 = pt(b)
        seg += f'<path d="M {x1:.1f} {y1:.1f} A {r:.1f} {r:.1f} 0 0 1 {x2:.1f} {y2:.1f}" stroke="{c}" stroke-width="{size*0.075:.1f}" fill="none"/>'
    nx, ny = pt(score / 100)
    needle = (f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="var(--text-primary)" stroke-width="3" stroke-linecap="round"/>'
              f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size*0.03:.1f}" fill="var(--text-primary)"/>')
    ticks = "".join(f'<text x="{pt(t/100)[0]:.1f}" y="{pt(t/100)[1]-size*0.055:.1f}" class="gtick">{t}</text>' for t in (0, 25, 50, 75, 100))
    return f'<svg viewBox="0 0 {size} {size*0.62:.0f}" class="gauge">{seg}{ticks}{needle}</svg>'


def render_body(indexes, risk, history, generated, note, sectors, backtest=None):
    overall = round(sum(b.score for b in indexes) / len(indexes), 1)
    o_reg = regime_for(overall); o_role, o_msg = REGIME_META[o_reg]

    def tile(label, value, sub="", role=None, series=None, band=None, zero=False, color="var(--series-1)"):
        chip = f'<span class="dot" style="background:{_sv(role)}"></span>' if role else ""
        sp = f'<div class="tilespark">{_spark(series, color=color, band=band, zero=zero)}</div>' if series is not None else ""
        return f'<div class="tile"><div class="tlabel">{label}</div><div class="tval">{chip}{value}</div><div class="tsub">{sub}</div>{sp}</div>'

    def card(b, color):
        role, msg = REGIME_META[b.regime]
        bars = "".join(f'<div class="subrow"><span class="sublabel">{k}</span>'
                       f'<span class="subbar"><span class="subfill" style="width:{v:.0f}%;background:{_score_color(v)}"></span></span>'
                       f'<span class="subval">{v:.0f}</span></div>' for k, v in b.components.items())
        thrust = '<span class="badge good">⚡ Breadth-thrust</span>' if b.thrust_signal else ""
        return f"""<section class="card"><div class="cardhead">
          <div><h2>{b.name}</h2><div class="muted">{b.sub} · {b.n} stocks</div></div>
          <div class="scorebox"><div class="bignum" style="color:{_sv(role)}">{b.score:.0f}</div>
          <div class="regime" style="color:{_sv(role)}">{b.regime}</div></div></div>
          <div class="regmsg"><span class="dot" style="background:{_sv(role)}"></span>{msg} {thrust}</div>
          <div class="tilegrid">
            {tile("% above 200-day MA", f"{b.pct_above_200:.0f}%", "long-term trend", _role_pct(b.pct_above_200), b.series_pct200, (0,100), color=color)}
            {tile("% above 50-day MA", f"{b.pct_above_50:.0f}%", "intermediate", _role_pct(b.pct_above_50), b.series_pct50, (0,100), color=color)}
            {tile("Net new highs − lows", f"{b.net_new_highs:+.1f}%", "52-week leadership", _role_nhnl(b.net_new_highs), b.series_nhnl, zero=True, color=color)}
            {tile("McClellan Oscillator", f"{b.mcclellan_osc:+.0f}", "breadth momentum", _role_mcosc(b.mcclellan_osc), b.series_mcosc, zero=True, color=color)}
            {tile("Advance-Decline line", f"{b.advancers}▲ / {b.decliners}▼", "today · cumulative", series=b.series_adline, color=color)}
            {tile("McClellan Summation", f"{b.mcclellan_sum:+.0f}", "long-term breadth tide", series=b.series_mcsum, zero=True, color=color)}
          </div><div class="subscores"><div class="subtitle">Composite components (0–100)</div>{bars}</div></section>"""

    cards = "".join(card(b, b.color) for b in indexes)
    mini = "".join(f'<div><span class="mk" style="background:{b.color}"></span>{b.name} <b>{b.score:.0f}</b> · {b.regime}</div>'
                   for b in indexes)

    # sector heatmap
    sec_html = ""
    if sectors:
        head = ('<div class="hmrow hmhead"><span class="hmname">S&P 500 sector</span><span class="hmn">names</span>'
                '<span class="hmcell">% &gt; 50d</span><span class="hmcell">% &gt; 200d</span></div>')
        rr = "".join(f'<div class="hmrow"><span class="hmname">{r["name"]}</span><span class="hmn">{r["n"]}</span>'
                     f'<span class="hmcell" style="background:{_heat(r["p50"])}">{r["p50"]:.0f}%</span>'
                     f'<span class="hmcell" style="background:{_heat(r["p200"])}">{r["p200"]:.0f}%</span></div>' for r in sectors)
        sec_html = f"""<section class="card"><div class="cardhead"><div><h2>Sector breadth — S&amp;P 500</h2>
          <div class="muted">Which parts of the market are participating. Green = broad, red = washed out. Sorted by long-term trend.</div></div></div>
          <div class="heatmap">{head}{rr}</div>
          <div class="muted" style="margin-top:8px">Broad green across cyclicals (financials, industrials, discretionary) confirms a healthy tape; leadership narrowing into just defensives or tech is a late-cycle tell.</div></section>"""

    def rtile(label, value, sub, role, series, color):
        if value is None:
            return f'<div class="tile"><div class="tlabel">{label}</div><div class="tval muted">n/a</div><div class="tsub">{sub}</div></div>'
        chip = f'<span class="dot" style="background:{_sv(role)}"></span>' if role else ""
        sp = f'<div class="tilespark">{_spark(series, color=color)}</div>' if series else ""
        return f'<div class="tile"><div class="tlabel">{label}</div><div class="tval">{chip}{value}</div><div class="tsub">{sub}</div>{sp}</div>'

    risk_html = f"""<section class="card"><div class="cardhead"><div><h2>Risk Context</h2>
      <div class="muted">Market-wide conditions around the breadth signal</div></div></div>
      <div class="tilegrid rg3">
        {rtile("VIX (volatility)", f"{risk.vix:.1f}" if risk.vix is not None else None, "&lt;15 calm · 20+ elevated · 30+ stress", risk.vix_role, risk.vix_series, "var(--series-4)")}
        {rtile("High-yield credit" + (" (proxy)" if risk.hy_proxy else " spread"), (None if risk.hy is None else (f"{risk.hy:+.1f}%" if risk.hy_proxy else f"{risk.hy:.2f}%")), risk.hy_label, risk.hy_role, risk.hy_series, "var(--series-4)")}
        {rtile("Equal-wt vs cap-wt (50d)", f"{risk.ewcw:+.1f}%" if risk.ewcw is not None else None, "RSP/SPY · rising = broadening", risk.ewcw_role, risk.ewcw_series, "var(--series-3)")}
      </div></section>"""

    toggles = "".join(f'<button class="tgl{" active" if k=="Overall" else ""}" data-k="{k}">{k}</button>'
                      for k in history["series"].keys())
    hist_html = f"""<section class="card"><div class="cardhead"><div><h2>Track record — breadth composite vs. S&amp;P 500</h2>
      <div class="muted">~2 years. Colored bands are the regime zones. Toggle the composite; hover for values.</div></div>
      <div class="toggles">{toggles}</div></div>
      <div id="histWrap" class="histwrap"><svg id="histSvg" preserveAspectRatio="none"></svg><div id="histTip" class="tip"></div></div>
      <div class="muted" style="margin-top:8px">Weight-of-evidence: composite sinking into the orange/red zones has tended to precede or accompany drawdowns; recoveries above ~58 mark broadening participation.</div></section>"""

    # backtest table
    bt_html = ""
    if backtest and backtest.get("total"):
        def rcol(cell):
            if not cell:
                return '<td class="btna">—</td>'
            col = "var(--st-good)" if cell["mean"] >= 0 else "var(--st-critical)"
            return (f'<td><span class="btret" style="color:{col}">{cell["mean"]:+.1f}%</span>'
                    f'<span class="bthit">{cell["hit"]:.0f}% up · n={cell["n"]}</span></td>')
        rows = ""
        for r in backtest["order"]:
            role = REGIME_META[r][0]
            cells = "".join(rcol(backtest["cells"][r][lbl]) for lbl in backtest["labels"])
            rows += (f'<tr><td class="btreg"><span class="dot" style="background:{_sv(role)}"></span>{r}</td>'
                     f'<td class="btocc">{backtest["occupancy"][r]:.0f}%</td>{cells}</tr>')
        bcells = "".join(rcol(backtest["baseline"][lbl]) for lbl in backtest["labels"])
        base_row = (f'<tr class="btbase"><td class="btreg">All days (buy &amp; hold)</td>'
                    f'<td class="btocc">100%</td>{bcells}</tr>')
        heads = "".join(f"<th>{lbl}</th>" for lbl in backtest["labels"])
        bt_html = f"""<section class="card"><div class="cardhead"><div>
          <h2>Does the dial work? — regime backtest</h2>
          <div class="muted">Every day in history sorted by its composite regime, then the S&amp;P 500's <b>actual forward return</b> from those days. Higher regimes should show better returns and higher win rates.</div></div></div>
          <div class="btwrap"><table class="bt"><thead><tr><th>Regime</th><th>% of time</th>{heads}</tr></thead>
          <tbody>{rows}{base_row}</tbody></table></div>
          <div class="muted" style="margin-top:8px">Forward returns are overlapping and drawn from ~{backtest["total"]} trading days of history — a real but <b>limited</b> sample (widen it with <code>--years 5</code>). Read the <i>ordering</i> (do better regimes beat worse ones?) more than any single number. Past performance ≠ future results.</div></section>"""

    payload = json.dumps({"history": history}, separators=(",", ":"))
    return f"""<div class="wrap">
  <header class="top"><div><h1>Market Breadth Dashboard</h1>
    <div class="muted">When to be more invested — and when not to. {note}</div></div>
    <div class="stamp">{generated}</div></header>
  <section class="hero"><div class="herogauge">{_gauge(overall)}
    <div class="herolabel"><div class="heroscore" style="color:{_sv(o_role)}">{overall:.0f}<span>/100</span></div>
    <div class="heroregime" style="color:{_sv(o_role)}">{o_reg}</div></div></div>
    <div class="heromsg"><div class="herotitle">Overall Invested-o-meter</div><p>{o_msg}</p>
    <div class="heromini">{mini}</div></div></section>
  <div class="cards cards3">{cards}</div>
  {sec_html}{risk_html}{hist_html}{bt_html}
  <section class="legend"><div class="subtitle">How to read this</div>
    <div class="scale">
      <span class="seg" style="background:var(--st-critical)">RISK-OFF<br><small>&lt;32</small></span>
      <span class="seg" style="background:var(--st-serious)">CAUTION<br><small>32–45</small></span>
      <span class="seg" style="background:var(--st-warning)">NEUTRAL<br><small>45–58</small></span>
      <span class="seg" style="background:var(--st-good-dim)">CONSTRUCTIVE<br><small>58–70</small></span>
      <span class="seg" style="background:var(--st-good)">RISK-ON<br><small>70+</small></span></div>
    <ul class="notes">
      <li><b>Breadth</b> measures how many stocks participate, not just where the index is. A rising index on <i>narrowing</i> breadth is fragile.</li>
      <li><b>% above 200-day MA</b> is the backbone: above ~60% healthy, below ~40% a broad downtrend, under ~15% often a capitulation low.</li>
      <li><b>Small caps</b> (S&amp;P 600) are the most breadth-sensitive — they lead broadenings and warn first when liquidity tightens.</li>
      <li><b>Risk context:</b> low VIX + tight credit spreads + a rising equal-weight ratio reinforce a risk-on read; the reverse argues caution.</li>
      <li>The composite is a <i>weight-of-evidence</i> risk dial, not a precise timing trigger.</li></ul></section>
  <footer class="foot"><p><b>Methodology:</b> per-index breadth from daily closes of each index's constituents.
    Composite — Trend (&gt;200d) 30%, Intermediate (&gt;50d) 20%, Momentum (McClellan) 20%, Leadership (NH−NL) 15%,
    AD-line slope 15%, short-term tilt + Zweig breadth-thrust bonus. Overall = average of the three indices. {note}
    Not investment advice — a decision-support tool.</p></footer>
</div>
<script>{JS.replace("__PAYLOAD__", payload)}</script>"""


def render_full(indexes, risk, history, generated, note, sectors, backtest=None):
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Market Breadth Dashboard</title><style>{CSS}</style></head><body>"
            f"{render_body(indexes, risk, history, generated, note, sectors, backtest)}</body></html>")


def render_artifact(indexes, risk, history, generated, note, sectors, backtest=None):
    return (f"<title>Market Breadth</title><style>{CSS}</style>"
            f"{render_body(indexes, risk, history, generated, note, sectors, backtest)}")


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--gridline:#e1e0d9;--border:rgba(11,11,11,0.10);
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#4a3aa7;--fill-1:rgba(42,120,214,0.10);
--st-good:#0ca30c;--st-good-dim:#5cae3a;--st-warning:#fab219;--st-serious:#ec835a;--st-critical:#d03b3b;
--band-good:rgba(12,163,12,.10);--band-gooddim:rgba(92,174,58,.10);--band-warning:rgba(250,178,25,.10);
--band-serious:rgba(236,131,90,.13);--band-critical:rgba(208,59,59,.13);}
@media(prefers-color-scheme:dark){:root:where(:not([data-theme=light])){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;--gridline:#2c2c2a;
--border:rgba(255,255,255,0.10);--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#9085e9;--fill-1:rgba(57,135,229,0.12);
--band-good:rgba(12,163,12,.16);--band-gooddim:rgba(92,174,58,.14);--band-warning:rgba(250,178,25,.13);
--band-serious:rgba(236,131,90,.16);--band-critical:rgba(208,59,59,.18);}}
:root[data-theme=dark]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
--gridline:#2c2c2a;--border:rgba(255,255,255,0.10);--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#9085e9;--fill-1:rgba(57,135,229,0.12);
--band-good:rgba(12,163,12,.16);--band-gooddim:rgba(92,174,58,.14);--band-warning:rgba(250,178,25,.13);--band-serious:rgba(236,131,90,.16);--band-critical:rgba(208,59,59,.18);}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased;line-height:1.45}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:0}
.muted{color:var(--muted);font-size:13px}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:22px;flex-wrap:wrap}
.stamp{text-align:right;font-size:12px;color:var(--muted)}.stamp b{color:var(--text-secondary)}
.hero{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:22px 26px;display:flex;gap:30px;align-items:center;flex-wrap:wrap;margin-bottom:20px}
.herogauge{width:240px;flex:0 0 auto;text-align:center}.gauge{width:240px;height:auto;overflow:visible;display:block}
.gtick{fill:var(--muted);font-size:11px;text-anchor:middle}.herolabel{margin-top:6px}
.heroscore{font-size:46px;font-weight:800;line-height:1}.heroscore span{font-size:18px;color:var(--muted);font-weight:600}
.heroregime{font-size:15px;font-weight:800;letter-spacing:.06em;margin-top:2px}
.heromsg{flex:1;min-width:260px}.herotitle{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700}
.heromsg p{font-size:16px;margin:6px 0 14px;max-width:56ch}
.heromini{display:flex;gap:22px;flex-wrap:wrap;font-size:14px;color:var(--text-secondary)}
.mk{display:inline-block;width:10px;height:10px;border-radius:3px;vertical-align:middle;margin-right:6px}
.cards{display:grid;gap:20px}.cards3{grid-template-columns:repeat(3,1fr)}
@media(max-width:900px){.cards3{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:20px 22px}
.wrap>.card,.wrap>section.card{margin-top:20px}
.cardhead{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;gap:12px;flex-wrap:wrap}
.scorebox{text-align:right}.bignum{font-size:38px;font-weight:800;line-height:1}
.regime{font-size:12px;font-weight:800;letter-spacing:.06em}
.regmsg{font-size:13px;color:var(--text-secondary);margin-bottom:14px;min-height:34px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}
.tilegrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.rg3{grid-template-columns:repeat(3,1fr)}
@media(max-width:900px){.rg3{grid-template-columns:1fr}}@media(max-width:520px){.tilegrid{grid-template-columns:1fr}}
.tile{background:var(--page);border:1px solid var(--border);border-radius:11px;padding:11px 13px}
.tlabel{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.tval{font-size:21px;font-weight:750;margin:3px 0 1px;font-variant-numeric:tabular-nums}.tsub{font-size:11.5px;color:var(--muted)}
.tilespark{margin-top:8px}.sparkwrap{position:relative}.spark{width:100%;height:44px;display:block;cursor:crosshair}
.subscores{margin-top:16px}
.subtitle{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:700;margin-bottom:9px}
.subrow{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:12.5px}
.sublabel{flex:0 0 148px;color:var(--text-secondary)}
.subbar{flex:1;height:7px;background:var(--gridline);border-radius:4px;overflow:hidden}
.subfill{display:block;height:100%;border-radius:4px}
.subval{flex:0 0 24px;text-align:right;color:var(--text-secondary);font-variant-numeric:tabular-nums}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;margin-left:6px}
.badge.good{background:rgba(12,163,12,.15);color:var(--st-good)}
.heatmap{display:flex;flex-direction:column;gap:4px}
.hmrow{display:grid;grid-template-columns:1fr 60px 92px 92px;gap:8px;align-items:center}
.hmhead{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:700;margin-bottom:2px}
.hmname{font-size:13.5px;color:var(--text-secondary)}.hmn{font-size:12px;color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
.hmhead .hmcell{background:none;color:var(--muted);padding:0;text-align:center}
.hmcell{text-align:center;color:#fff;font-weight:750;font-size:13.5px;padding:7px 4px;border-radius:7px;font-variant-numeric:tabular-nums;text-shadow:0 1px 2px rgba(0,0,0,.35)}
@media(max-width:560px){.hmrow{grid-template-columns:1fr 44px 70px 70px}}
.toggles{display:flex;gap:6px;flex-wrap:wrap}
.tgl{font:inherit;font-size:12.5px;font-weight:600;padding:5px 11px;border-radius:20px;cursor:pointer;
background:var(--page);color:var(--text-secondary);border:1px solid var(--border)}
.tgl.active{background:var(--text-primary);color:var(--page);border-color:var(--text-primary)}
.histwrap{position:relative;margin-top:10px}
#histSvg{width:100%;height:300px;display:block}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:7px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.18);opacity:0;transition:opacity .08s;white-space:nowrap;z-index:5}
.tip b{font-variant-numeric:tabular-nums}
.axlab{fill:var(--muted);font-size:10px}
.btwrap{overflow-x:auto}
.bt{width:100%;border-collapse:collapse;font-size:13.5px;min-width:520px}
.bt th{text-align:right;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:700;padding:6px 10px;border-bottom:1px solid var(--border)}
.bt th:first-child{text-align:left}
.bt td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--border);vertical-align:middle}
.btreg{text-align:left!important;font-weight:650;white-space:nowrap}
.btocc{color:var(--text-secondary);font-variant-numeric:tabular-nums}
.btret{display:block;font-weight:750;font-variant-numeric:tabular-nums}
.bthit{display:block;font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
.btna{color:var(--muted)}
.btbase{border-top:2px solid var(--border)}.btbase td{color:var(--text-secondary);font-style:italic}
.btbase .btret{font-style:normal}
.legend{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:20px 22px;margin-top:20px}
.scale{display:flex;gap:4px;margin:6px 0 14px}
.seg{flex:1;text-align:center;color:#fff;font-size:11px;font-weight:800;letter-spacing:.04em;padding:9px 4px;border-radius:7px;text-shadow:0 1px 2px rgba(0,0,0,.35)}
.seg small{font-weight:600;opacity:.9}
.notes{margin:0;padding-left:18px;color:var(--text-secondary);font-size:13.5px}.notes li{margin:5px 0}
.foot{color:var(--muted);font-size:12px;margin-top:22px;line-height:1.6}
"""


JS = r"""
(function(){
  var P = __PAYLOAD__;
  var H = P.history;
  var cs = getComputedStyle(document.documentElement);
  function cvar(n){return cs.getPropertyValue(n).trim() || n;}
  var SVGNS = "http://www.w3.org/2000/svg";
  var svg = document.getElementById('histSvg');
  var tip = document.getElementById('histTip');
  var wrap = document.getElementById('histWrap');
  var W=1000, HT=300, L=34, R=14, PAD=10;
  var topH=HT*0.62, gap=14, botH=HT*0.30;
  svg.setAttribute('viewBox','0 0 '+W+' '+HT);
  var dates=H.dates, price=H.price, cur='Overall';
  var bands=[[0,32,'--band-critical'],[32,45,'--band-serious'],[45,58,'--band-warning'],[58,70,'--band-gooddim'],[70,100,'--band-good']];
  var m=dates.length, plotW=W-L-R;
  function X(i){return L + (m<2?0:i/(m-1)*plotW);}
  function Yt(v){return PAD + (100-v)/100*(topH-2*PAD);}
  var pmin=Infinity,pmax=-Infinity;
  for(var i=0;i<price.length;i++){var v=price[i]; if(v==null)continue; if(v<pmin)pmin=v; if(v>pmax)pmax=v;}
  if(pmax<=pmin)pmax=pmin+1;
  var y0=topH+gap;
  function Yp(v){return y0+PAD+(pmax-v)/(pmax-pmin)*(botH-2*PAD);}
  function el(n,a){var e=document.createElementNS(SVGNS,n);for(var k in a)e.setAttribute(k,a[k]);return e;}
  function draw(){
    while(svg.firstChild)svg.removeChild(svg.firstChild);
    // regime bands
    bands.forEach(function(b){svg.appendChild(el('rect',{x:L,y:Yt(b[1]),width:plotW,height:Yt(b[0])-Yt(b[1]),fill:cvar(b[2])}));});
    [0,50,100].forEach(function(v){
      svg.appendChild(el('line',{x1:L,y1:Yt(v),x2:W-R,y2:Yt(v),stroke:cvar('--gridline'),'stroke-width':1}));
      var t=el('text',{x:L-5,y:Yt(v)+3,'class':'axlab','text-anchor':'end'});t.textContent=v;svg.appendChild(t);
    });
    // price area+line
    var pd='', pl='';
    for(var i=0;i<m;i++){var v=price[i]; if(v==null)continue; var x=X(i),y=Yp(v); pl+=(pl?' ':'')+x.toFixed(1)+','+y.toFixed(1);}
    if(pl){
      svg.appendChild(el('polygon',{points:X(0).toFixed(1)+','+(y0+botH).toFixed(1)+' '+pl+' '+X(m-1).toFixed(1)+','+(y0+botH).toFixed(1),fill:cvar('--fill-1')}));
      svg.appendChild(el('polyline',{points:pl,fill:'none',stroke:cvar('--series-1'),'stroke-width':2}));
      var pt=el('text',{x:L,y:y0+11,'class':'axlab'});pt.textContent='S&P 500 price';svg.appendChild(pt);
    }
    // composite line
    var s=H.series[cur], cp='';
    for(var i=0;i<m;i++){var v=s[i]; if(v==null)continue; cp+=(cp?' ':'')+X(i).toFixed(1)+','+Yt(v).toFixed(1);}
    svg.appendChild(el('polyline',{points:cp,fill:'none',stroke:cvar('--text-primary'),'stroke-width':2.5,'stroke-linejoin':'round'}));
    // x date ticks
    [0,.25,.5,.75,1].forEach(function(f,k){
      var i=Math.round(f*(m-1)); var t=el('text',{x:X(i).toFixed(1),y:HT-1,'class':'axlab','text-anchor':k===0?'start':k===4?'end':'middle'});
      t.textContent=(dates[i]||'').slice(0,7); svg.appendChild(t);
    });
    // crosshair holders
    cross=el('line',{x1:0,y1:PAD,x2:0,y2:y0+botH,stroke:cvar('--muted'),'stroke-width':1,'stroke-dasharray':'3 3',opacity:0});svg.appendChild(cross);
    dotC=el('circle',{r:4,fill:cvar('--text-primary'),opacity:0});svg.appendChild(dotC);
    dotP=el('circle',{r:3.5,fill:cvar('--series-1'),opacity:0});svg.appendChild(dotP);
  }
  var cross,dotC,dotP;
  draw();
  function move(ev){
    var r=svg.getBoundingClientRect();
    var px=(ev.touches?ev.touches[0].clientX:ev.clientX)-r.left;
    var xv=px/r.width*W;
    var i=Math.round((xv-L)/plotW*(m-1)); if(i<0)i=0; if(i>m-1)i=m-1;
    var sx=X(i); var sc=H.series[cur][i], pr=price[i];
    cross.setAttribute('x1',sx);cross.setAttribute('x2',sx);cross.setAttribute('opacity',1);
    if(sc!=null){dotC.setAttribute('cx',sx);dotC.setAttribute('cy',Yt(sc));dotC.setAttribute('opacity',1);}else dotC.setAttribute('opacity',0);
    if(pr!=null){dotP.setAttribute('cx',sx);dotP.setAttribute('cy',Yp(pr));dotP.setAttribute('opacity',1);}else dotP.setAttribute('opacity',0);
    var reg = sc==null?'':(sc>=70?'RISK-ON':sc>=58?'CONSTRUCTIVE':sc>=45?'NEUTRAL':sc>=32?'CAUTION':'RISK-OFF');
    tip.innerHTML='<div>'+dates[i]+'</div><div>'+cur+': <b>'+(sc==null?'—':sc.toFixed(1))+'</b> '+reg+'</div>'+(pr!=null?'<div>S&P 500: <b>'+pr.toLocaleString()+'</b></div>':'');
    tip.style.opacity=1;
    var tx=px+14; if(tx>r.width-150)tx=px-tip.offsetWidth-14;
    tip.style.left=tx+'px'; tip.style.top='6px';
  }
  function leave(){tip.style.opacity=0;if(cross){cross.setAttribute('opacity',0);dotC.setAttribute('opacity',0);dotP.setAttribute('opacity',0);}}
  svg.addEventListener('mousemove',move);svg.addEventListener('mouseleave',leave);
  svg.addEventListener('touchstart',move);svg.addEventListener('touchmove',function(e){e.preventDefault();move(e);},{passive:false});
  document.querySelectorAll('.tgl').forEach(function(btn){
    btn.addEventListener('click',function(){
      document.querySelectorAll('.tgl').forEach(function(b){b.classList.remove('active');});
      btn.classList.add('active'); cur=btn.getAttribute('data-k'); draw();
    });
  });
  // sparkline tooltips
  var stip=document.createElement('div');stip.className='tip';document.body.appendChild(stip);
  document.querySelectorAll('.sparkwrap').forEach(function(w){
    var data; try{data=JSON.parse(w.getAttribute('data-series'));}catch(e){return;}
    var sp=w.querySelector('.spark');
    sp.addEventListener('mousemove',function(ev){
      var r=sp.getBoundingClientRect(); var f=(ev.clientX-r.left)/r.width; var i=Math.round(f*(data.length-1));
      if(i<0)i=0; if(i>data.length-1)i=data.length-1; var d=data[i];
      stip.innerHTML='<div>'+d.t+'</div><div><b>'+(typeof d.v==='number'?d.v.toLocaleString():d.v)+'</b></div>';
      stip.style.opacity=1; stip.style.left=(ev.pageX+12)+'px'; stip.style.top=(ev.pageY-10)+'px';
    });
    sp.addEventListener('mouseleave',function(){stip.style.opacity=0;});
  });
})();
"""


# ============================================================================
# 7.  BUILD / MAIN
# ============================================================================

def _history(indexes, hist, price):
    common = hist.index
    def col(s): return [None if pd.isna(x) else round(float(x), 1) for x in s.reindex(common).ffill()]
    series = {"Overall": [None if pd.isna(x) else round(float(x), 1) for x in hist]}
    names = {"sp500": "S&P 500", "nasdaq": "NASDAQ", "sp600": "Small Caps"}
    for b in indexes:
        series[names.get(b.key, b.name)] = col(b.score_ts)
    if price is not None:
        pr = price.reindex(common, method="ffill")
        pcol = [None if pd.isna(x) else round(float(x), 2) for x in pr]
    else:
        pcol = [None] * len(common)
    return {"dates": [d.strftime("%Y-%m-%d") for d in common], "series": series, "price": pcol}


def build(sample=None, refresh=False, demo=False, log=True, email_to=None,
          source="yahoo", tiingo_key=None, tv_user=None, tv_pass=None, artifact_path=None, years=2):
    indexes, sectors = [], None
    period = f"{max(int(years), 2)}y"
    if demo:
        print("DEMO mode: synthetic data (no network).")
        for spec, (nn, seed, drift) in zip(INDEX_SPECS, [(120, 1, 0.0009), (160, 2, 0.0006), (140, 3, 0.0003)]):
            b = compute_breadth(demo_closes(nn, seed, drift), spec["name"], spec["sub"])
            b.key, b.color = spec["key"], spec["color"]
            indexes.append(b)
        price = demo_price()
        rng = np.random.default_rng(5)
        secs = ["Information Technology", "Financials", "Health Care", "Consumer Discretionary",
                "Industrials", "Communication Services", "Consumer Staples", "Energy",
                "Utilities", "Materials", "Real Estate"]
        sectors = sorted([{"name": s, "n": int(rng.integers(20, 75)),
                           "p50": float(rng.uniform(25, 90)), "p200": float(rng.uniform(20, 92))}
                          for s in secs], key=lambda r: -r["p200"])
        note = "DEMO / SYNTHETIC DATA — not real market data."
    else:
        print("Fetching constituents ...")
        for spec in INDEX_SPECS:
            try:
                tickers = spec["fetch"]()
                print(f"  {spec['name']}: {len(tickers)} names")
                if sample:
                    import random; random.seed(42)
                    tickers = sorted(random.sample(tickers, min(sample, len(tickers))))
                closes = download_closes(tickers, spec["key"], refresh=refresh, period=period, source=source,
                                         tiingo_key=tiingo_key, tv_user=tv_user, tv_pass=tv_pass)
                b = compute_breadth(closes, spec["name"], spec["sub"])
                b.key, b.color = spec["key"], spec["color"]
                indexes.append(b)
                if spec["key"] == "sp500":
                    sm = fetch_sp500_sectors()
                    if sm:
                        sectors = compute_sector_breadth(closes, sm)
            except Exception as e:
                print(f"  WARN: skipping {spec['name']} — {e}")
        if not indexes:
            raise RuntimeError("Could not build any index — every data source was blocked.")
        gspc = download_series("^GSPC", "gspc", refresh, period=f"{max(int(years) + 1, 3)}y")
        price = gspc.dropna().iloc[-(int(years) * 252 + 140):] if gspc is not None else None
        src = {"yahoo": "Yahoo Finance", "tiingo": "Tiingo", "tradingview": "TradingView"}.get(source, source)
        aux = "; VIX/price via Yahoo." if source in ("tiingo", "tradingview") else "."
        note = (("Sampled universes." if sample else "Full constituent universes.")
                + f" Source: {src} + FRED" + aux)

    risk = build_risk(refresh=refresh, demo=demo)
    aligned = pd.concat([b.score_ts for b in indexes], axis=1).dropna()
    hist = aligned.mean(axis=1)
    history = _history(indexes, hist, price)
    backtest = compute_backtest(history)
    try:
        from zoneinfo import ZoneInfo
        ts = dt.datetime.now(ZoneInfo("America/New_York")).strftime("%a %b %d, %Y %I:%M %p ET")
    except Exception:
        ts = dt.datetime.now().strftime("%a %b %d, %Y %H:%M")
    data_through = history["dates"][-1] if history.get("dates") else "n/a"
    generated = f"Data through<br><b>{data_through} close</b><br>built {ts}"

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(render_full(indexes, risk, history, generated, note, sectors, backtest))
    if artifact_path:
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(render_artifact(indexes, risk, history, generated, note, sectors, backtest))

    overall = round(sum(b.score for b in indexes) / len(indexes), 1)
    print()
    for b in indexes:
        print(f"  {b.name:<22} {b.score:>5}  [{b.regime}]")
    print(f"  {'OVERALL':<22} {overall:>5}  [{regime_for(overall)}]")
    if backtest.get("total"):
        print(f"\n  Regime backtest (fwd S&P return, ~{backtest['total']} days):")
        for r in backtest["order"]:
            c3 = backtest["cells"][r].get("3-month")
            if c3:
                print(f"    {r:<13} 3-mo: {c3['mean']:+5.1f}%  ({c3['hit']:.0f}% up, n={c3['n']})")
    print(f"\nWrote {OUT_HTML}")

    if log and not demo:
        log_history(indexes, overall, risk, os.path.join(HERE, "breadth_history.csv"))
    if email_to:
        lines = [f"Overall breadth: {overall} [{regime_for(overall)}]", ""]
        lines += [f"  {b.name}: {b.score} [{b.regime}]  (%>200d: {b.pct_above_200:.0f}%)" for b in indexes]
        send_email(email_to, OUT_HTML,
                   f"Market Breadth {dt.date.today().isoformat()}: {overall} {regime_for(overall)}",
                   "\n".join(lines) + "\n\nFull dashboard attached.")
    return OUT_HTML


def main():
    ap = argparse.ArgumentParser(description="Market breadth dashboard")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--email", metavar="TO", default=None)
    ap.add_argument("--source", choices=["yahoo", "tiingo", "tradingview"], default="yahoo")
    ap.add_argument("--tiingo-key", default=os.environ.get("TIINGO_API_KEY"))
    ap.add_argument("--tv-user", default=os.environ.get("TV_USERNAME"))
    ap.add_argument("--tv-pass", default=os.environ.get("TV_PASSWORD"))
    ap.add_argument("--artifact", metavar="PATH", default=None,
                    help="also write a head/body-less variant for hosting")
    ap.add_argument("--years", type=int, default=2,
                    help="years of history to pull (bigger = longer backtest, slower download)")
    args = ap.parse_args()
    if args.source == "tiingo" and not args.tiingo_key:
        ap.error("--source tiingo requires --tiingo-key or TIINGO_API_KEY")
    if args.source == "tradingview" and not args.sample:
        print("note: --source tradingview is slow (one call per symbol) and unofficial; "
              "consider --sample 150 for a first run.")
    out = build(sample=args.sample, refresh=args.refresh, demo=args.demo, log=not args.no_log,
                email_to=args.email, source=args.source, tiingo_key=args.tiingo_key,
                tv_user=args.tv_user, tv_pass=args.tv_pass, artifact_path=args.artifact, years=args.years)
    if not args.no_open and not args.email:
        try:
            webbrowser.open("file://" + os.path.abspath(out))
        except Exception:
            pass


if __name__ == "__main__":
    main()
