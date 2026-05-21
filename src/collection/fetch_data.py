"""
src/collection/fetch_data.py — Data Collection Module
======================================================
Responsibility: Download raw financial data from external sources.

Two data sources are used (satisfying the ≥2 requirement):
  1. Yahoo Finance  → stock OHLCV prices via yfinance
  2. NewsAPI        → latest financial news headlines

All raw data is saved to data/raw/ as CSV files.
"""

import time
import logging
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path

# Import our project settings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TICKERS, PERIOD, INTERVAL, DATA_RAW_DIR, NEWS_API_KEY, \
                   get_ticker_prices_dir, get_shared_dir

# ── Logging Setup ─────────────────────────────────────────────────────────────
# This prints timestamped messages to the terminal so you can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 1: Yahoo Finance — Stock Prices
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_prices(tickers: list = None,
                       period: str = PERIOD,
                       interval: str = INTERVAL,
                       start: str = None,
                       end: str = None) -> dict:
    """
    Download daily OHLCV (Open, High, Low, Close, Volume) stock data
    from Yahoo Finance for each ticker in the list.

    Args:
        tickers:  List of stock symbols, e.g. ["AAPL", "MSFT"]
        period:   How far back to go, e.g. "1y", "2y", "5y" (ignored if start/end given)
        interval: Bar size, e.g. "1d" = daily
        start:    Optional start date string "YYYY-MM-DD" for historical range
        end:      Optional end date string "YYYY-MM-DD" for historical range

    Returns:
        A dictionary: { "AAPL": DataFrame, "MSFT": DataFrame, ... }
        Each DataFrame has columns: Open, High, Low, Close, Volume
    """
    if tickers is None:
        tickers = TICKERS
    results = {}

    for ticker in tickers:
        # Sanitise ticker — strip $, spaces, force uppercase
        ticker = ticker.strip().upper().replace("$", "").replace(" ", "")
        logger.info(f"Fetching stock data for {ticker}...")

        try:
            # Use start/end dates if provided, otherwise use period
            if start:
                df = yf.download(
                    ticker, start=start, end=end, interval=interval,
                    progress=False, auto_adjust=True,
                )
                logger.info(f"Fetching stock data for {ticker} ({start} → {end or 'today'})...")
            else:
                logger.info(f"Fetching stock data for {ticker}...")
                df = yf.download(
                    ticker, period=period, interval=interval,
                    progress=False, auto_adjust=True,
                )

            if df.empty:
                logger.warning(f"No data returned for {ticker}. Skipping.")
                continue

            # Flatten MultiIndex columns (yfinance 1.x returns these)
            # e.g. ("Close", "TSLA") → "Close"
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep only the standard OHLCV columns
            available = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                         if c in df.columns]
            if not available:
                logger.warning(f"No OHLCV columns found for {ticker}. Skipping.")
                continue
            df = df[available].copy()

            # Ensure proper DatetimeIndex with no timezone
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Drop fully empty rows
            df.dropna(how="all", inplace=True)

            # Log how many rows we got
            logger.info(f"  ✓ {ticker}: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")

            # Save raw data to CSV
            # Save raw price data to organised ticker subfolder
            # Path: data/raw/TICKER/prices/TICKER_raw.csv
            output_path = get_ticker_prices_dir(ticker) / f"{ticker}_raw.csv"
            df.to_csv(output_path)
            logger.info(f"  💾 Saved to {output_path}")

            results[ticker] = df

            # Be polite — small delay between API calls to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            # If anything goes wrong, log it but keep going with other tickers
            logger.error(f"Failed to fetch {ticker}: {e}")

    logger.info(f"\nStock data collection complete. Got data for: {list(results.keys())}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 2: NewsAPI — Financial News Headlines
# ─────────────────────────────────────────────────────────────────────────────

def fetch_financial_news(tickers: list[str] = TICKERS,
                         max_articles: int = 5) -> pd.DataFrame:
    """
    Fetch recent financial news headlines for our tracked stocks from NewsAPI.
    Free tier allows 100 requests/day — we keep queries minimal.

    Args:
        tickers:      List of stock symbols to search news for
        max_articles: Max articles to fetch per ticker

    Returns:
        A DataFrame with columns: ticker, title, source, published_at, url
    """
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set in .env — skipping news collection.")
        return pd.DataFrame()  # Return empty DataFrame if no key

    all_articles = []
    base_url = "https://newsapi.org/v2/everything"

    for ticker in tickers:
        logger.info(f"Fetching news for {ticker}...")

        # Build the request parameters
        params = {
            "q": ticker,           # Search query
            "language": "en",      # English articles only
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": NEWS_API_KEY,
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()  # Raise error if HTTP status is 4xx/5xx

            data = response.json()
            articles = data.get("articles", [])

            for article in articles:
                all_articles.append({
                    "ticker":       ticker,
                    "title":        article.get("title", ""),
                    "source":       article.get("source", {}).get("name", ""),
                    "published_at": article.get("publishedAt", ""),
                    "url":          article.get("url", ""),
                })

            logger.info(f"  ✓ {ticker}: {len(articles)} articles fetched")
            time.sleep(1)  # Respect rate limit

        except requests.exceptions.RequestException as e:
            logger.error(f"News API error for {ticker}: {e}")

    if not all_articles:
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    df["published_at"] = pd.to_datetime(df["published_at"])

    # Save to CSV
    # Save news data to shared news folder
    # Path: data/raw/shared/news/news_raw.csv
    output_path = get_shared_dir("news") / "news_raw.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n💾 News data saved to {output_path} ({len(df)} articles total)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience function to run both sources at once
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_data() -> tuple[dict, pd.DataFrame]:
    """
    Run both data collection sources and return the results.

    Returns:
        (stock_data_dict, news_df)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 60)

    stock_data = fetch_stock_prices()
    news_data  = fetch_financial_news()

    return stock_data, news_data


# ─────────────────────────────────────────────────────────────────────────────
#  Run this file directly to test: python src/collection/fetch_data.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stocks, news = collect_all_data()
    print(f"\nCollected {len(stocks)} stocks and {len(news)} news articles.")


# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 3: Free RSS News (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_rss_news(tickers: list = None, max_articles: int = 5) -> pd.DataFrame:
    """
    Fetch financial news from free RSS feeds — no API key required.

    Sources used:
      - Yahoo Finance RSS (per ticker)
      - Google News RSS (per ticker, filtered to finance)

    Args:
        tickers:      List of stock symbols
        max_articles: Max articles per ticker per source

    Returns:
        DataFrame with columns: ticker, title, source, published_at, url, sentiment_hint
    """
    if tickers is None:
        tickers = TICKERS

    # Try to import feedparser — graceful fallback if not installed
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser not installed — skipping RSS news. Run: pip install feedparser")
        return pd.DataFrame()

    all_articles = []

    for ticker in tickers:
        logger.info(f"  Fetching RSS news for {ticker}...")

        # RSS feed URLs for this ticker
        feeds = [
            {
                "url":    f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
                "source": "Yahoo Finance RSS",
            },
            {
                "url":    f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
                "source": "Google News RSS",
            },
        ]

        ticker_count = 0
        for feed_info in feeds:
            if ticker_count >= max_articles:
                break
            try:
                feed = feedparser.parse(feed_info["url"])
                entries = feed.entries[:max_articles]

                for entry in entries:
                    if ticker_count >= max_articles:
                        break

                    # Parse published date
                    pub_date = None
                    if hasattr(entry, "published"):
                        try:
                            pub_date = pd.to_datetime(entry.published, utc=True)
                        except Exception:
                            pub_date = None

                    title = entry.get("title", "")

                    # Simple keyword sentiment hint (not a score, just directional)
                    title_lower = title.lower()
                    if any(w in title_lower for w in ["surge", "soar", "gain", "beat", "record", "rise", "up"]):
                        sentiment_hint = "Positive"
                    elif any(w in title_lower for w in ["fall", "drop", "miss", "loss", "down", "cut", "warn"]):
                        sentiment_hint = "Negative"
                    else:
                        sentiment_hint = "Neutral"

                    all_articles.append({
                        "ticker":         ticker,
                        "title":          title,
                        "source":         feed_info["source"],
                        "published_at":   pub_date,
                        "url":            entry.get("link", ""),
                        "sentiment_hint": sentiment_hint,
                    })
                    ticker_count += 1

            except Exception as e:
                logger.error(f"  RSS error for {ticker} ({feed_info['source']}): {e}")
                continue

        logger.info(f"  ✓ {ticker}: {ticker_count} RSS articles fetched")
        time.sleep(0.5)

    if not all_articles:
        logger.warning("  No RSS articles collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)

    # Save to CSV
    # Save RSS news to shared news folder
    # Path: data/raw/shared/news/rss_news.csv
    output_path = get_shared_dir("news") / "rss_news.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n💾 RSS news saved to {output_path} ({len(df)} articles)")

    return df