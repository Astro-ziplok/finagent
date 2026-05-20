"""
src/fundamentals/alpha_vantage.py — Alpha Vantage Data Module
==============================================================
Responsibility: Fetch financial data from Alpha Vantage API.

Provides 4 data types:
  1. News & Sentiment    → headlines with positive/negative/neutral scores
  2. Financial Statements → income statement, balance sheet, cash flow
  3. Key Ratios          → P/E, ROE, ROA, D/E from company overview
  4. Commodity Prices    → oil (WTI), natural gas, gold, copper

Free tier: 25 requests/day — this module batches carefully to stay within limit.

Usage:
    from src.fundamentals.alpha_vantage import fetch_all_fundamental_data
    data = fetch_all_fundamental_data(["AAPL", "NVDA"])
"""

import time
import logging
import requests
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ALPHA_VANTAGE_KEY, DATA_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"


def _get(params: dict, label: str, feature: str = "unknown", ticker: str = "") -> dict:
    """
    Make a single Alpha Vantage API request with error handling.
    Quota tracking is handled externally by agent.py via QuotaManager.

    Args:
        params:  Query parameters for the API call
        label:   Human-readable label for logging
        feature: Feature name (unused here, kept for signature compatibility)
        ticker:  Stock ticker for logging

    Returns:
        Parsed JSON dict, or empty dict on failure
    """
    if not ALPHA_VANTAGE_KEY:
        logger.warning("ALPHA_VANTAGE_KEY not set in .env — skipping Alpha Vantage call.")
        return {}

    params["apikey"] = ALPHA_VANTAGE_KEY

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Alpha Vantage returns error messages inside the JSON body
        if "Error Message" in data:
            logger.error(f"  Alpha Vantage error for {label}: {data['Error Message']}")
            return {}
        if "Note" in data:
            logger.warning(f"  Rate limit hit for {label}. Wait 60s before retrying.")
            return {}
        if "Information" in data:
            logger.warning(f"  Alpha Vantage info for {label}: {data['Information']}")
            return {}

        return data

    except requests.exceptions.RequestException as e:
        logger.error(f"  Request failed for {label}: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  1. News & Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news_sentiment(ticker: str, limit: int = 10) -> dict:
    """
    Fetch recent news articles with sentiment scores for a ticker.

    Each article includes:
      - title, url, source, published_at
      - overall_sentiment_label: "Bullish" / "Bearish" / "Neutral" / "Somewhat Bullish" etc.
      - overall_sentiment_score: float from -1 (very bearish) to +1 (very bullish)
      - ticker_sentiment_score: sentiment specifically about this ticker

    Args:
        ticker: Stock symbol e.g. "AAPL"
        limit:  Max number of articles (max 50 on free tier)

    Returns:
        Dict with keys: articles (list), avg_sentiment_score, sentiment_label, summary
    """
    logger.info(f"  Fetching news & sentiment for {ticker}...")

    data = _get({
        "function": "NEWS_SENTIMENT",
        "tickers":  ticker,
        "limit":    limit,
        "sort":     "LATEST",
    }, label=f"news_{ticker}", feature="news", ticker=ticker)

    if not data or "feed" not in data:
        return {"articles": [], "avg_sentiment_score": 0, "sentiment_label": "Neutral", "summary": "No news data available."}

    articles = []
    sentiment_scores = []

    for item in data.get("feed", [])[:limit]:
        # Find this ticker's specific sentiment score in the article
        ticker_sentiment = 0.0
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                try:
                    ticker_sentiment = float(ts.get("ticker_sentiment_score", 0))
                except:
                    ticker_sentiment = 0.0
                break

        articles.append({
            "title":           item.get("title", ""),
            "source":          item.get("source", ""),
            "published_at":    item.get("time_published", "")[:10],  # Date only
            "url":             item.get("url", ""),
            "sentiment_label": item.get("overall_sentiment_label", "Neutral"),
            "sentiment_score": round(ticker_sentiment, 4),
            "summary":         item.get("summary", "")[:200],  # First 200 chars
        })
        sentiment_scores.append(ticker_sentiment)

    avg_score = round(sum(sentiment_scores) / len(sentiment_scores), 4) if sentiment_scores else 0

    # Convert average score to a readable label
    if avg_score >= 0.35:
        label = "Bullish"
    elif avg_score >= 0.15:
        label = "Somewhat Bullish"
    elif avg_score <= -0.35:
        label = "Bearish"
    elif avg_score <= -0.15:
        label = "Somewhat Bearish"
    else:
        label = "Neutral"

    result = {
        "ticker":              ticker,
        "article_count":       len(articles),
        "avg_sentiment_score": avg_score,
        "sentiment_label":     label,
        "articles":            articles,
    }

    logger.info(f"  ✓ {ticker} news: {len(articles)} articles | avg sentiment: {avg_score} ({label})")

    # Save to CSV
    if articles:
        df = pd.DataFrame(articles)
        df.to_csv(DATA_RAW_DIR / f"{ticker}_news_sentiment.csv", index=False)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  2. Company Overview & Key Ratios
# ─────────────────────────────────────────────────────────────────────────────

def fetch_company_overview(ticker: str) -> dict:
    """
    Fetch company overview and key financial ratios from Alpha Vantage.

    Returns key metrics including:
      - P/E Ratio, Forward P/E
      - ROE, ROA
      - Debt/Equity Ratio
      - EPS, Revenue TTM, Profit Margin
      - Market Cap, 52-week high/low
      - Analyst target price

    Args:
        ticker: Stock symbol e.g. "AAPL"

    Returns:
        Dict of cleaned financial ratios and company info
    """
    logger.info(f"  Fetching company overview for {ticker}...")

    data = _get({"function": "OVERVIEW", "symbol": ticker}, label=f"overview_{ticker}", feature="overview", ticker=ticker)

    if not data:
        return {}

    def safe_float(val, default=None):
        """Convert to float safely, return default if invalid."""
        try:
            f = float(val)
            return round(f, 4) if f else default
        except (TypeError, ValueError):
            return default

    def safe_int(val, default=None):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default

    result = {
        # Company info
        "name":            data.get("Name", ticker),
        "sector":          data.get("Sector", "N/A"),
        "industry":        data.get("Industry", "N/A"),
        "description":     data.get("Description", "")[:300],
        "exchange":        data.get("Exchange", "N/A"),
        "market_cap":      safe_int(data.get("MarketCapitalization")),

        # Valuation ratios
        "pe_ratio":        safe_float(data.get("PERatio")),
        "forward_pe":      safe_float(data.get("ForwardPE")),
        "pb_ratio":        safe_float(data.get("PriceToBookRatio")),
        "ps_ratio":        safe_float(data.get("PriceToSalesRatioTTM")),
        "peg_ratio":       safe_float(data.get("PEGRatio")),

        # Profitability
        "roe":             safe_float(data.get("ReturnOnEquityTTM")),
        "roa":             safe_float(data.get("ReturnOnAssetsTTM")),
        "profit_margin":   safe_float(data.get("ProfitMargin")),
        "operating_margin":safe_float(data.get("OperatingMarginTTM")),

        # Financial health
        "debt_to_equity":  safe_float(data.get("DebtToEquityRatio")),
        "current_ratio":   safe_float(data.get("CurrentRatio")),
        "quick_ratio":     safe_float(data.get("QuickRatio")),

        # Earnings & revenue
        "eps":             safe_float(data.get("EPS")),
        "revenue_ttm":     safe_int(data.get("RevenueTTM")),
        "revenue_per_share":safe_float(data.get("RevenuePerShareTTM")),
        "gross_profit_ttm":safe_int(data.get("GrossProfitTTM")),
        "ebitda":          safe_int(data.get("EBITDA")),

        # Dividend
        "dividend_yield":  safe_float(data.get("DividendYield")),
        "dividend_per_share": safe_float(data.get("DividendPerShare")),

        # Analyst estimates
        "analyst_target_price": safe_float(data.get("AnalystTargetPrice")),
        "analyst_rating_strong_buy": safe_int(data.get("AnalystRatingStrongBuy")),
        "analyst_rating_buy":        safe_int(data.get("AnalystRatingBuy")),
        "analyst_rating_hold":       safe_int(data.get("AnalystRatingHold")),
        "analyst_rating_sell":       safe_int(data.get("AnalystRatingSell")),

        # 52-week range
        "52w_high": safe_float(data.get("52WeekHigh")),
        "52w_low":  safe_float(data.get("52WeekLow")),

        # Beta
        "beta": safe_float(data.get("Beta")),
    }

    logger.info(f"  ✓ {ticker} overview: P/E={result['pe_ratio']}, ROE={result['roe']}, D/E={result['debt_to_equity']}")

    # Save to JSON
    import json
    with open(DATA_RAW_DIR / f"{ticker}_overview.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  3. Financial Statements
# ─────────────────────────────────────────────────────────────────────────────

def fetch_income_statement(ticker: str) -> dict:
    """
    Fetch the last 4 quarters of income statement data.

    Returns key line items:
      - Total Revenue, Gross Profit, Operating Income, Net Income
      - EPS (basic and diluted)
    """
    logger.info(f"  Fetching income statement for {ticker}...")
    time.sleep(1)  # Respect rate limit

    data = _get({"function": "INCOME_STATEMENT", "symbol": ticker}, label=f"income_{ticker}", feature="income", ticker=ticker)

    if not data or "quarterlyReports" not in data:
        return {}

    quarters = []
    for report in data["quarterlyReports"][:4]:  # Last 4 quarters
        def sv(key):
            try:
                v = report.get(key, "0")
                return int(float(v)) if v and v != "None" else 0
            except:
                return 0

        quarters.append({
            "fiscal_date":      report.get("fiscalDateEnding", ""),
            "total_revenue":    sv("totalRevenue"),
            "gross_profit":     sv("grossProfit"),
            "operating_income": sv("operatingIncome"),
            "net_income":       sv("netIncome"),
            "ebit":             sv("ebit"),
            "rd_expense":       sv("researchAndDevelopment"),
        })

    result = {"ticker": ticker, "quarterly_income": quarters}
    logger.info(f"  ✓ {ticker} income statement: {len(quarters)} quarters")

    df = pd.DataFrame(quarters)
    df.to_csv(DATA_RAW_DIR / f"{ticker}_income_statement.csv", index=False)

    return result


def fetch_balance_sheet(ticker: str) -> dict:
    """
    Fetch the latest balance sheet data.

    Returns key line items:
      - Total Assets, Total Liabilities
      - Shareholder Equity, Cash, Long-term Debt
    """
    logger.info(f"  Fetching balance sheet for {ticker}...")
    time.sleep(1)

    data = _get({"function": "BALANCE_SHEET", "symbol": ticker}, label=f"balance_{ticker}", feature="balance", ticker=ticker)

    if not data or "quarterlyReports" not in data:
        return {}

    quarters = []
    for report in data["quarterlyReports"][:4]:
        def sv(key):
            try:
                v = report.get(key, "0")
                return int(float(v)) if v and v != "None" else 0
            except:
                return 0

        quarters.append({
            "fiscal_date":        report.get("fiscalDateEnding", ""),
            "total_assets":       sv("totalAssets"),
            "total_liabilities":  sv("totalLiabilities"),
            "shareholder_equity": sv("totalShareholderEquity"),
            "cash":               sv("cashAndCashEquivalentsAtCarryingValue"),
            "long_term_debt":     sv("longTermDebt"),
            "short_term_debt":    sv("shortTermDebt"),
            "retained_earnings":  sv("retainedEarnings"),
        })

    result = {"ticker": ticker, "quarterly_balance": quarters}
    logger.info(f"  ✓ {ticker} balance sheet: {len(quarters)} quarters")

    df = pd.DataFrame(quarters)
    df.to_csv(DATA_RAW_DIR / f"{ticker}_balance_sheet.csv", index=False)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  4. Commodity Prices (Macro Indicators)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_commodity_prices() -> dict:
    """
    Fetch commodity prices using Yahoo Finance futures tickers.
    This is completely FREE and uses NO Alpha Vantage quota.

    Commodities fetched:
      - WTI Crude Oil  → CL=F
      - Natural Gas    → NG=F
      - Gold           → GC=F
      - Copper         → HG=F

    Returns a dict with latest price, unit, 1M/3M change, and trend for each.
    """
    import yfinance as yf

    logger.info("  Fetching commodity prices from Yahoo Finance (free, no quota)...")

    # Yahoo Finance futures tickers for each commodity
    commodities = {
        "WTI_Oil":     {"ticker": "CL=F", "unit": "USD per barrel"},
        "Natural_Gas": {"ticker": "NG=F", "unit": "USD per MMBtu"},
        "Gold":        {"ticker": "GC=F", "unit": "USD per troy oz"},
        "Copper":      {"ticker": "HG=F", "unit": "USD per pound"},
    }

    results = {}

    for name, info in commodities.items():
        yf_ticker = info["ticker"]
        unit      = info["unit"]

        try:
            # Download 3 months of daily data
            df = yf.download(yf_ticker, period="3mo", interval="1d",
                             progress=False, auto_adjust=True)

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or "Close" not in df.columns:
                logger.warning(f"  ⚠️  No data returned for {name} ({yf_ticker})")
                results[name] = {
                    "latest_price": None, "trend": "N/A",
                    "unit": unit, "source": "Yahoo Finance"
                }
                continue

            close = df["Close"].dropna()

            latest = round(float(close.iloc[-1]), 2)
            # 1-month ago (~21 trading days)
            price_1m = float(close.iloc[-21]) if len(close) >= 21 else float(close.iloc[0])
            # 3-months ago (start of data)
            price_3m = float(close.iloc[0])

            pct_1m = round((latest / price_1m - 1) * 100, 2)
            pct_3m = round((latest / price_3m - 1) * 100, 2)
            trend  = "Rising" if pct_1m > 1 else ("Falling" if pct_1m < -1 else "Flat")
            date   = str(close.index[-1].date())

            results[name] = {
                "latest_price":  latest,
                "date":          date,
                "unit":          unit,
                "1m_change_pct": pct_1m,
                "3m_change_pct": pct_3m,
                "trend":         trend,
                "source":        "Yahoo Finance",
            }

            logger.info(f"  ✓ {name} ({yf_ticker}): ${latest} {unit} | "
                        f"1M: {pct_1m:+.1f}% | 3M: {pct_3m:+.1f}% | {trend}")

            time.sleep(0.3)  # Small delay between requests

        except Exception as e:
            logger.error(f"  Failed to fetch {name} ({yf_ticker}): {e}")
            results[name] = {
                "latest_price": None, "trend": "N/A",
                "unit": unit, "source": "Yahoo Finance"
            }

    # Save to CSV
    if results:
        df_out = pd.DataFrame(results).T
        df_out.to_csv(DATA_RAW_DIR / "commodity_prices.csv")
        logger.info("  💾 Commodity prices saved to data/raw/commodity_prices.csv")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Master function: fetch all fundamental data for a list of tickers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_fundamental_data(tickers: list, features: list = None) -> dict:
    """
    Fetch Alpha Vantage data for the given tickers.
    Only fetches the features specified — saves quota when not everything is needed.

    Args:
        tickers:  List of stock symbols
        features: List of features to fetch. Options: "news", "overview", "income", "balance"
                  Defaults to all four if not specified.

    Returns:
        Dict structured as:
        {
          "AAPL": {"news": {...}, "overview": {...}, "income": {...}, "balance": {...}},
          "commodities": {...}
        }
    """
    if not ALPHA_VANTAGE_KEY:
        logger.warning("ALPHA_VANTAGE_KEY not set — skipping all Alpha Vantage data.")
        return {}

    # Default to all features if not specified
    if features is None:
        features = ["news", "overview", "income", "balance"]

    logger.info("=" * 60)
    logger.info("ALPHA VANTAGE: FUNDAMENTALS & NEWS")
    logger.info(f"Fetching {features} for {tickers}")
    logger.info("=" * 60)

    all_data = {}

    for ticker in tickers:
        logger.info(f"\n── {ticker} ──────────────────────────────────────────")
        all_data[ticker] = {}

        if "news" in features:
            all_data[ticker]["news"] = fetch_news_sentiment(ticker)
            time.sleep(1)

        if "overview" in features:
            all_data[ticker]["overview"] = fetch_company_overview(ticker)
            time.sleep(1)

        if "income" in features:
            all_data[ticker]["income"] = fetch_income_statement(ticker)
            time.sleep(1)

        if "balance" in features:
            all_data[ticker]["balance"] = fetch_balance_sheet(ticker)
            time.sleep(2)

    logger.info(f"\n✅ Alpha Vantage data collection complete for {tickers}")
    return all_data


# ─────────────────────────────────────────────────────────────────────────────
#  Run directly to test: python src/fundamentals/alpha_vantage.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    data = fetch_all_fundamental_data(["AAPL"])
    print(json.dumps(data, indent=2, default=str))