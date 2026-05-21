"""
src/fundamentals/yfinance_fundamentals.py — Financial Statements via yfinance
==============================================================================
Fetches financial statement data directly from Yahoo Finance.
Completely FREE — no API key, no quota, no rate limits.

Provides:
  1. Income Statement  — revenue, gross profit, net income, EPS (annual + quarterly)
  2. Balance Sheet     — assets, liabilities, equity, cash, debt
  3. Cash Flow         — operating, investing, financing cash flows, free cash flow
  4. Key Metrics       — EPS growth, revenue growth, profit margin trend, FCF yield

All data comes from yfinance which wraps Yahoo Finance's public endpoints.
"""

import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_RAW_DIR, get_ticker_financials_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_val(val, round_digits=2):
    """Safely convert a value to float, return None if invalid."""
    try:
        f = float(val)
        return round(f, round_digits) if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _fmt_millions(val) -> str:
    """Format a large number into readable millions/billions string."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if abs(v) >= 1e9:
            return f"${v/1e9:.2f}B"
        elif abs(v) >= 1e6:
            return f"${v/1e6:.1f}M"
        else:
            return f"${v:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _pct_change(new, old) -> float:
    """Calculate percentage change between two values."""
    try:
        if old and old != 0:
            return round((float(new) - float(old)) / abs(float(old)) * 100, 2)
    except (TypeError, ValueError):
        pass
    return None


def fetch_income_statement(ticker: str) -> dict:
    """
    Fetch annual and quarterly income statement from Yahoo Finance via yfinance.

    Returns:
        Dict with annual and quarterly data, plus YoY growth metrics
    """
    import yfinance as yf
    logger.info(f"  Fetching income statement for {ticker} (Yahoo Finance)...")

    try:
        stock = yf.Ticker(ticker)

        # Annual financials
        annual_df = stock.financials
        # Quarterly financials
        quarterly_df = stock.quarterly_financials

        if annual_df is None or annual_df.empty:
            logger.warning(f"  No income statement data for {ticker}")
            return {}

        def extract_rows(df, max_periods=4):
            """Extract key income statement rows from a DataFrame."""
            rows = []
            cols = df.columns[:max_periods]  # Most recent periods first

            for col in cols:
                row = {"period": str(col.date()) if hasattr(col, 'date') else str(col)}

                def get(key_options):
                    for k in key_options:
                        if k in df.index:
                            return _safe_val(df.loc[k, col])
                    return None

                total_rev    = get(["Total Revenue", "TotalRevenue"])
                gross_profit = get(["Gross Profit", "GrossProfit"])
                op_income    = get(["Operating Income", "OperatingIncome", "EBIT"])
                net_income   = get(["Net Income", "NetIncome"])
                ebitda       = get(["EBITDA", "Ebitda"])
                rd           = get(["Research Development", "ResearchAndDevelopment",
                                    "Research And Development"])

                row.update({
                    "total_revenue":    total_rev,
                    "gross_profit":     gross_profit,
                    "operating_income": op_income,
                    "net_income":       net_income,
                    "ebitda":           ebitda,
                    "rd_expense":       rd,
                    # Calculated margins
                    "gross_margin_pct":    round(gross_profit / total_rev * 100, 2)
                                          if gross_profit and total_rev else None,
                    "net_margin_pct":      round(net_income / total_rev * 100, 2)
                                          if net_income and total_rev else None,
                    "operating_margin_pct":round(op_income / total_rev * 100, 2)
                                          if op_income and total_rev else None,
                    # Formatted for display
                    "revenue_fmt":      _fmt_millions(total_rev),
                    "net_income_fmt":   _fmt_millions(net_income),
                })
                rows.append(row)
            return rows

        annual_rows    = extract_rows(annual_df, max_periods=4)
        quarterly_rows = extract_rows(quarterly_df, max_periods=4) if (
            quarterly_df is not None and not quarterly_df.empty) else []

        # YoY growth metrics (most recent vs prior year)
        yoy_growth = {}
        if len(annual_rows) >= 2:
            curr, prev = annual_rows[0], annual_rows[1]
            yoy_growth = {
                "revenue_growth_pct":    _pct_change(curr["total_revenue"], prev["total_revenue"]),
                "net_income_growth_pct": _pct_change(curr["net_income"], prev["net_income"]),
                "gross_margin_change":   round(
                    (curr["gross_margin_pct"] or 0) - (prev["gross_margin_pct"] or 0), 2
                ),
            }

        result = {
            "ticker":          ticker,
            "annual":          annual_rows,
            "quarterly":       quarterly_rows,
            "yoy_growth":      yoy_growth,
            "source":          "Yahoo Finance (yfinance)",
        }

        # Save to CSV
        # Save to organised financials folder
        # Path: data/raw/TICKER/financials/TICKER_income_annual.csv
        if annual_rows:
            pd.DataFrame(annual_rows).to_csv(
                get_ticker_financials_dir(ticker) / f"{ticker}_income_annual.csv",
                index=False)

        logger.info(f"  ✓ {ticker} income statement: {len(annual_rows)} annual, "
                    f"{len(quarterly_rows)} quarterly periods")
        return result

    except Exception as e:
        logger.error(f"  Failed to fetch income statement for {ticker}: {e}")
        return {}


def fetch_balance_sheet(ticker: str) -> dict:
    """
    Fetch annual and quarterly balance sheet from Yahoo Finance via yfinance.

    Returns:
        Dict with assets, liabilities, equity, and key ratios
    """
    import yfinance as yf
    logger.info(f"  Fetching balance sheet for {ticker} (Yahoo Finance)...")

    try:
        stock = yf.Ticker(ticker)
        annual_df    = stock.balance_sheet
        quarterly_df = stock.quarterly_balance_sheet

        if annual_df is None or annual_df.empty:
            logger.warning(f"  No balance sheet data for {ticker}")
            return {}

        def extract_rows(df, max_periods=4):
            rows = []
            cols = df.columns[:max_periods]

            for col in cols:
                row = {"period": str(col.date()) if hasattr(col, 'date') else str(col)}

                def get(key_options):
                    for k in key_options:
                        if k in df.index:
                            return _safe_val(df.loc[k, col])
                    return None

                total_assets  = get(["Total Assets", "TotalAssets"])
                total_liab    = get(["Total Liabilities Net Minority Interest",
                                     "TotalLiabilitiesNetMinorityInterest",
                                     "Total Liabilities"])
                equity        = get(["Stockholders Equity", "StockholdersEquity",
                                     "Total Stockholder Equity"])
                cash          = get(["Cash And Cash Equivalents", "CashAndCashEquivalents",
                                     "Cash"])
                lt_debt       = get(["Long Term Debt", "LongTermDebt"])
                st_debt       = get(["Current Debt", "CurrentDebt",
                                     "Short Long Term Debt", "ShortLongTermDebt"])
                retained      = get(["Retained Earnings", "RetainedEarnings"])

                # Debt-to-equity ratio
                de_ratio = round(float(lt_debt or 0) / float(equity) * 100, 2) \
                           if equity and equity != 0 else None

                row.update({
                    "total_assets":       total_assets,
                    "total_liabilities":  total_liab,
                    "shareholder_equity": equity,
                    "cash":               cash,
                    "long_term_debt":     lt_debt,
                    "short_term_debt":    st_debt,
                    "retained_earnings":  retained,
                    "de_ratio_pct":       de_ratio,
                    # Formatted
                    "assets_fmt":         _fmt_millions(total_assets),
                    "equity_fmt":         _fmt_millions(equity),
                    "cash_fmt":           _fmt_millions(cash),
                    "lt_debt_fmt":        _fmt_millions(lt_debt),
                })
                rows.append(row)
            return rows

        annual_rows    = extract_rows(annual_df, 4)
        quarterly_rows = extract_rows(quarterly_df, 4) if (
            quarterly_df is not None and not quarterly_df.empty) else []

        result = {
            "ticker":    ticker,
            "annual":    annual_rows,
            "quarterly": quarterly_rows,
            "source":    "Yahoo Finance (yfinance)",
        }

        # Save to organised financials folder
        # Path: data/raw/TICKER/financials/TICKER_balance_annual.csv
        if annual_rows:
            pd.DataFrame(annual_rows).to_csv(
                get_ticker_financials_dir(ticker) / f"{ticker}_balance_annual.csv",
                index=False)

        logger.info(f"  ✓ {ticker} balance sheet: {len(annual_rows)} annual periods")
        return result

    except Exception as e:
        logger.error(f"  Failed to fetch balance sheet for {ticker}: {e}")
        return {}


def fetch_cash_flow(ticker: str) -> dict:
    """
    Fetch annual and quarterly cash flow statement from Yahoo Finance.

    Returns:
        Dict with operating, investing, financing cash flows and free cash flow
    """
    import yfinance as yf
    logger.info(f"  Fetching cash flow for {ticker} (Yahoo Finance)...")

    try:
        stock = yf.Ticker(ticker)
        annual_df    = stock.cashflow
        quarterly_df = stock.quarterly_cashflow

        if annual_df is None or annual_df.empty:
            logger.warning(f"  No cash flow data for {ticker}")
            return {}

        def extract_rows(df, max_periods=4):
            rows = []
            cols = df.columns[:max_periods]

            for col in cols:
                row = {"period": str(col.date()) if hasattr(col, 'date') else str(col)}

                def get(key_options):
                    for k in key_options:
                        if k in df.index:
                            return _safe_val(df.loc[k, col])
                    return None

                op_cf   = get(["Operating Cash Flow", "OperatingCashFlow",
                                "Total Cash From Operating Activities"])
                inv_cf  = get(["Investing Cash Flow", "InvestingCashFlow",
                                "Total Cash From Investing Activities"])
                fin_cf  = get(["Financing Cash Flow", "FinancingCashFlow",
                                "Total Cash From Financing Activities"])
                capex   = get(["Capital Expenditure", "CapitalExpenditure",
                                "Capital Expenditures"])
                fcf     = get(["Free Cash Flow", "FreeCashFlow"])

                # Calculate FCF if not directly available
                if fcf is None and op_cf and capex:
                    fcf = round(op_cf + capex, 2)  # capex is usually negative

                row.update({
                    "operating_cf":  op_cf,
                    "investing_cf":  inv_cf,
                    "financing_cf":  fin_cf,
                    "capex":         capex,
                    "free_cash_flow":fcf,
                    # Formatted
                    "operating_cf_fmt":   _fmt_millions(op_cf),
                    "fcf_fmt":            _fmt_millions(fcf),
                    "capex_fmt":          _fmt_millions(capex),
                })
                rows.append(row)
            return rows

        annual_rows    = extract_rows(annual_df, 4)
        quarterly_rows = extract_rows(quarterly_df, 4) if (
            quarterly_df is not None and not quarterly_df.empty) else []

        # FCF growth
        fcf_growth = None
        if len(annual_rows) >= 2:
            fcf_growth = _pct_change(
                annual_rows[0].get("free_cash_flow"),
                annual_rows[1].get("free_cash_flow")
            )

        result = {
            "ticker":     ticker,
            "annual":     annual_rows,
            "quarterly":  quarterly_rows,
            "fcf_growth_pct": fcf_growth,
            "source":     "Yahoo Finance (yfinance)",
        }

        # Save to organised financials folder
        # Path: data/raw/TICKER/financials/TICKER_cashflow_annual.csv
        if annual_rows:
            pd.DataFrame(annual_rows).to_csv(
                get_ticker_financials_dir(ticker) / f"{ticker}_cashflow_annual.csv",
                index=False)

        logger.info(f"  ✓ {ticker} cash flow: {len(annual_rows)} annual periods | "
                    f"FCF growth: {fcf_growth}%")
        return result

    except Exception as e:
        logger.error(f"  Failed to fetch cash flow for {ticker}: {e}")
        return {}


def fetch_key_metrics(ticker: str) -> dict:
    """
    Build a key metrics dashboard from yfinance info + financial statements.

    Returns:
        Dict with EPS, EPS growth, revenue growth, margins, FCF yield,
        P/E, P/S, beta, dividend yield, analyst target price
    """
    import yfinance as yf
    logger.info(f"  Fetching key metrics for {ticker} (Yahoo Finance)...")

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        def gi(key, default=None):
            v = info.get(key, default)
            return _safe_val(v) if v is not None else default

        # Basic info
        name     = info.get("longName", ticker)
        sector   = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")

        # EPS metrics
        eps_ttm      = gi("trailingEps")
        eps_forward  = gi("forwardEps")
        eps_growth   = _pct_change(eps_forward, eps_ttm) if eps_ttm and eps_forward else None

        # Revenue
        revenue_ttm  = gi("totalRevenue")
        revenue_qoq  = gi("revenueGrowth")   # Already a decimal (e.g. 0.12 = 12%)
        revenue_yoy  = round(float(revenue_qoq) * 100, 2) if revenue_qoq else None

        # Profitability
        profit_margin   = gi("profitMargins")
        gross_margin    = gi("grossMargins")
        operating_margin= gi("operatingMargins")

        # Valuation
        pe_trailing  = gi("trailingPE")
        pe_forward   = gi("forwardPE")
        pb_ratio     = gi("priceToBook")
        ps_ratio     = gi("priceToSalesTrailing12Months")
        peg_ratio    = gi("pegRatio")

        # Balance sheet quick metrics
        current_ratio = gi("currentRatio")
        de_ratio      = gi("debtToEquity")
        beta          = gi("beta")

        # Market data
        market_cap        = gi("marketCap")
        current_price     = gi("currentPrice") or gi("regularMarketPrice")
        target_price      = gi("targetMeanPrice")
        upside_pct        = _pct_change(target_price, current_price) if target_price and current_price else None
        dividend_yield    = gi("dividendYield")

        # FCF yield (FCF / Market Cap)
        fcf               = gi("freeCashflow")
        fcf_yield         = round(float(fcf) / float(market_cap) * 100, 2) \
                            if fcf and market_cap and market_cap != 0 else None

        # Analyst consensus
        rec               = info.get("recommendationKey", "N/A")
        target_high       = gi("targetHighPrice")
        target_low        = gi("targetLowPrice")
        n_analysts        = gi("numberOfAnalystOpinions")

        metrics = {
            "ticker":   ticker,
            "name":     name,
            "sector":   sector,
            "industry": industry,

            "valuation": {
                "current_price":     current_price,
                "market_cap":        market_cap,
                "market_cap_fmt":    _fmt_millions(market_cap),
                "pe_trailing":       pe_trailing,
                "pe_forward":        pe_forward,
                "pb_ratio":          pb_ratio,
                "ps_ratio":          ps_ratio,
                "peg_ratio":         peg_ratio,
                "beta":              beta,
            },
            "earnings": {
                "eps_ttm":           eps_ttm,
                "eps_forward":       eps_forward,
                "eps_growth_pct":    eps_growth,
            },
            "growth": {
                "revenue_yoy_pct":   revenue_yoy,
                "revenue_ttm":       revenue_ttm,
                "revenue_ttm_fmt":   _fmt_millions(revenue_ttm),
            },
            "profitability": {
                "profit_margin_pct":    round(float(profit_margin)*100, 2) if profit_margin else None,
                "gross_margin_pct":     round(float(gross_margin)*100, 2) if gross_margin else None,
                "operating_margin_pct": round(float(operating_margin)*100, 2) if operating_margin else None,
                "fcf_yield_pct":        fcf_yield,
                "fcf_fmt":              _fmt_millions(fcf),
            },
            "financial_health": {
                "current_ratio":     current_ratio,
                "debt_to_equity":    de_ratio,
            },
            "analyst": {
                "recommendation":    rec,
                "target_price":      target_price,
                "target_high":       target_high,
                "target_low":        target_low,
                "upside_pct":        upside_pct,
                "n_analysts":        n_analysts,
            },
            "dividend": {
                "dividend_yield_pct": round(float(dividend_yield)*100, 2) if dividend_yield else None,
            },
            "source": "Yahoo Finance (yfinance)",
        }

        logger.info(f"  ✓ {ticker} key metrics: P/E={pe_trailing}, "
                    f"EPS growth={eps_growth}%, Revenue growth={revenue_yoy}%")
        return metrics

    except Exception as e:
        logger.error(f"  Failed to fetch key metrics for {ticker}: {e}")
        return {}


def fetch_all_yfinance_fundamentals(ticker: str) -> dict:
    """
    Fetch all financial statement data for a ticker using yfinance.
    Completely free — no API key, no quota.

    Args:
        ticker: Stock symbol e.g. "AMZN"

    Returns:
        Dict with keys: income, balance, cashflow, metrics
    """
    logger.info(f"\n── {ticker} (Yahoo Finance Fundamentals) ────────────────")

    result = {
        "income":   fetch_income_statement(ticker),
        "balance":  fetch_balance_sheet(ticker),
        "cashflow": fetch_cash_flow(ticker),
        "metrics":  fetch_key_metrics(ticker),
        "source":   "Yahoo Finance (yfinance) — free, no quota",
    }

    time.sleep(0.5)  # Small delay between tickers
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Run directly to test: python src/fundamentals/yfinance_fundamentals.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    data = fetch_all_yfinance_fundamentals("AMZN")
    print(json.dumps(data["metrics"], indent=2, default=str))
    print(f"\nIncome periods: {len(data['income'].get('annual', []))}")
    print(f"Balance periods: {len(data['balance'].get('annual', []))}")
    print(f"Cash flow periods: {len(data['cashflow'].get('annual', []))}")