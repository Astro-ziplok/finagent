"""
src/valuation/inputs.py — Data inputs for valuation, each with its source
==========================================================================
  risk_free_rate()        10-year U.S. Treasury yield: Alpha Vantage (Federal Reserve
                          H.15 series), cached once per day; Yahoo ^TNX as fallback
  company_inputs()        quote, estimates and statements for one company
  peer_multiples()        valuation multiples for a list of peer tickers
  select_peers()          peer group from config (override -> industry -> sector)

Results are cached for the session so a report and its comparison peers do
not download the same data twice.
"""

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (ALPHA_VANTAGE_KEY, VALUATION_ASSUMPTIONS, INDUSTRY_PEERS, TICKER_PEERS,
                    MIN_PEERS, get_shared_dir)

logger = logging.getLogger(__name__)
_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
#  Risk-free rate
# ─────────────────────────────────────────────────────────────────────────────

def risk_free_rate() -> dict:
    """
    Current 10-year Treasury yield as a decimal, with source and date.
    Order: today's cached value -> Alpha Vantage (1 request/day) -> Yahoo ^TNX -> config fallback.
    """
    cache_file = get_shared_dir("macro") / "treasury_10y.json"
    today = str(date.today())
    if cache_file.exists():
        try:
            c = json.loads(cache_file.read_text())
            if c.get("fetched") == today:
                return c
        except (ValueError, OSError):
            pass

    result = None
    if ALPHA_VANTAGE_KEY:
        from src.fundamentals.quota import QuotaManager
        q = QuotaManager()
        if q.remaining > 0:
            try:
                r = requests.get("https://www.alphavantage.co/query", timeout=20, params={
                    "function": "TREASURY_YIELD", "interval": "daily", "maturity": "10year",
                    "apikey": ALPHA_VANTAGE_KEY})
                q.record(1)
                rows = [x for x in r.json().get("data", []) if x.get("value") not in (None, "", ".")]
                if rows:
                    result = {"value": float(rows[0]["value"]) / 100, "as_of": rows[0]["date"],
                              "source": ("10-Year Treasury Constant Maturity Rate, Federal Reserve H.15, "
                                         f"via Alpha Vantage (as of {rows[0]['date']})")}
            except (requests.RequestException, ValueError, KeyError) as e:
                logger.warning(f"Alpha Vantage Treasury yield unavailable: {e}")

    if result is None:
        try:
            import yfinance as yf
            h = yf.download("^TNX", period="10d", progress=False, auto_adjust=True)
            close = h["Close"].dropna()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if len(close):
                as_of = str(close.index[-1].date())
                result = {"value": float(close.iloc[-1]) / 100, "as_of": as_of,
                          "source": f"CBOE 10-Year Treasury Yield index (^TNX), Yahoo Finance (as of {as_of})"}
        except Exception as e:  # network or parsing problems
            logger.warning(f"Yahoo ^TNX unavailable: {e}")

    if result is None:
        fb = VALUATION_ASSUMPTIONS["risk_free_fallback"]
        return {"value": fb["value"], "as_of": today, "source": fb["source"], "fetched": today}

    result["fetched"] = today
    try:
        cache_file.write_text(json.dumps(result))
    except OSError:
        pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Company inputs
# ─────────────────────────────────────────────────────────────────────────────

def _estimates(df) -> dict:
    """Pull 0y / +1y rows from a yfinance estimate table into plain floats."""
    out = {}
    if df is None or getattr(df, "empty", True):
        return out
    for period in ("0y", "+1y"):
        if period in df.index:
            row = df.loc[period]
            out[period] = {k: (None if pd.isna(row.get(k)) else float(row.get(k)))
                           for k in ("avg", "growth", "numberOfAnalysts", "yearAgoRevenue", "yearAgoEps")
                           if k in row.index}
    return out


def company_inputs(ticker: str) -> dict:
    """Quote data, analyst estimates and 5 years of monthly prices for one company."""
    key = f"company_{ticker}"
    if key in _CACHE:
        return _CACHE[key]
    import yfinance as yf
    t = yf.Ticker(ticker)
    info = t.info or {}
    out = {"ticker": ticker, "info": info, "as_of": str(date.today())}
    try:
        out["revenue_est"] = _estimates(t.revenue_estimate)
    except Exception:
        out["revenue_est"] = {}
    try:
        out["eps_est"] = _estimates(t.earnings_estimate)
    except Exception:
        out["eps_est"] = {}
    try:
        # Split-adjusted but not dividend-adjusted, to match reported EPS for P/E history
        m = yf.download(ticker, period="5y", interval="1mo", progress=False, auto_adjust=False)["Close"]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[:, 0]
        out["monthly_close"] = m.dropna()
    except Exception:
        out["monthly_close"] = pd.Series(dtype=float)
    try:
        sp = t.splits
        out["splits"] = sp[sp.index.tz_localize(None) >= pd.Timestamp.today() - pd.DateOffset(years=5)]             if sp is not None and len(sp) else pd.Series(dtype=float)
    except Exception:
        out["splits"] = pd.Series(dtype=float)
    _CACHE[key] = out
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Peers
# ─────────────────────────────────────────────────────────────────────────────

def select_peers(ticker: str, sector: str, industry: str) -> dict:
    """
    Peer group and the reason for it:
      1. hand-picked override in TICKER_PEERS (with its stated rationale)
      2. same Yahoo industry from INDUSTRY_PEERS
      3. widened to the largest companies in the same sector if fewer than MIN_PEERS
    """
    if ticker in TICKER_PEERS:
        peers, why = TICKER_PEERS[ticker]
        return {"peers": [p for p in peers if p != ticker], "basis": "curated", "rationale": why}
    same = [p for p in INDUSTRY_PEERS.get((sector, industry), []) if p != ticker]
    if len(same) >= MIN_PEERS:
        return {"peers": same[:8], "basis": "industry",
                "rationale": f"Companies in the same Yahoo Finance industry ({industry})."}
    sector_pool = []
    for (sec, ind), tickers in INDUSTRY_PEERS.items():
        if sec == sector and ind != industry:
            sector_pool += [p for p in tickers[:2] if p != ticker]
    peers = (same + [p for p in sector_pool if p not in same])[:8]
    if len(peers) < MIN_PEERS:
        return {"peers": peers, "basis": "none",
                "rationale": (f"No curated peer group for {industry or 'this industry'}; add one to "
                              "INDUSTRY_PEERS or TICKER_PEERS in config.py.")}
    return {"peers": peers, "basis": "sector",
            "rationale": (f"Only {len(same)} other company in Yahoo's {industry} industry, so the group "
                          f"was widened to large {sector} companies.")}


PEER_FIELDS = {"forward_pe": "forwardPE", "trailing_pe": "trailingPE", "ev_ebitda": "enterpriseToEbitda",
               "ev_sales": "enterpriseToRevenue", "pb": "priceToBook", "roe": "returnOnEquity",
               "forward_eps": "forwardEps", "trailing_eps": "trailingEps", "market_cap": "marketCap"}


def peer_multiples(peers: list) -> dict:
    """Valuation multiples for each peer from Yahoo Finance quote data (cached)."""
    import yfinance as yf
    out = {}
    for p in peers:
        key = f"peer_{p}"
        if key not in _CACHE:
            try:
                info = yf.Ticker(p).info or {}
            except Exception:
                info = {}
            row = {k: info.get(v) for k, v in PEER_FIELDS.items()}
            row["name"] = info.get("shortName") or p
            fe, te = row.get("forward_eps"), row.get("trailing_eps")
            row["exp_eps_growth"] = (fe / te - 1) if (fe and te and te > 0) else None
            _CACHE[key] = row
        out[p] = _CACHE[key]
    return out
