"""
config.py — Global Configuration for FinAgent
==============================================
Change settings here to customise what the agent tracks and analyses.
All other modules import from this file — you only need to edit one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from .env file automatically
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_RAW_DIR   = BASE_DIR / "data" / "raw"
DATA_PROC_DIR  = BASE_DIR / "data" / "processed"

# ── Organised Reports Structure ───────────────────────────────────────────────
# Charts and analysis are now saved in separate subfolders per ticker
REPORTS_DIR         = BASE_DIR / "reports"
REPORTS_CHARTS_DIR  = REPORTS_DIR / "charts"      # reports/charts/TICKER/
REPORTS_ANALYSIS_DIR= REPORTS_DIR / "analysis"    # reports/analysis/
REPORTS_DATA_DIR    = REPORTS_DIR / "data_snapshots" # reports/data_snapshots/

# Create all folders if they don't exist yet
for folder in [DATA_RAW_DIR, DATA_PROC_DIR, REPORTS_DIR,
               REPORTS_CHARTS_DIR, REPORTS_ANALYSIS_DIR, REPORTS_DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
# ── Organised Raw Data Structure ─────────────────────────────────────────────
# Raw data is now organised into subfolders per ticker and data type:
#   data/raw/AAPL/prices/AAPL_raw.csv
#   data/raw/AAPL/financials/AAPL_income_annual.csv
#   data/raw/AAPL/financials/AAPL_balance_annual.csv
#   data/raw/AAPL/financials/AAPL_cashflow_annual.csv
#   data/raw/AAPL/news/AAPL_news_sentiment.csv
#   data/raw/shared/commodities/commodity_prices.csv
#   data/raw/shared/news/rss_news.csv
#   data/raw/shared/quota/av_quota.json
 
def get_ticker_prices_dir(ticker: str) -> Path:
    """
    Returns the folder path for a ticker's price data.
    Creates the folder automatically if it does not exist.
    Example: data/raw/AAPL/prices/
    """
    d = DATA_RAW_DIR / ticker / "prices"
    d.mkdir(parents=True, exist_ok=True)
    return d
 
 
def get_ticker_financials_dir(ticker: str) -> Path:
    """
    Returns the folder path for a ticker's financial statement data.
    Creates the folder automatically if it does not exist.
    Example: data/raw/AAPL/financials/
    Stores: income_annual.csv, balance_annual.csv, cashflow_annual.csv,
            overview.json, income_statement.csv, balance_sheet.csv
    """
    d = DATA_RAW_DIR / ticker / "financials"
    d.mkdir(parents=True, exist_ok=True)
    return d
 
 
def get_ticker_news_dir(ticker: str) -> Path:
    """
    Returns the folder path for a ticker's news and sentiment data.
    Creates the folder automatically if it does not exist.
    Example: data/raw/AAPL/news/
    Stores: news_sentiment.csv
    """
    d = DATA_RAW_DIR / ticker / "news"
    d.mkdir(parents=True, exist_ok=True)
    return d
 
 
def get_shared_dir(category: str) -> Path:
    """
    Returns the folder path for shared (non-ticker-specific) data.
    Creates the folder automatically if it does not exist.
    Example: data/raw/shared/commodities/
             data/raw/shared/news/
             data/raw/shared/quota/
    Categories: 'commodities', 'news', 'quota'
    """
    d = DATA_RAW_DIR / "shared" / category
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── Assets to Track ───────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

# ── Date Range ────────────────────────────────────────────────────────────────
PERIOD   = "1y"   # Options: "1mo", "3mo", "6mo", "1y", "2y"
INTERVAL = "1d"   # Daily data

# ── Feature Engineering Settings ─────────────────────────────────────────────
ROLLING_SHORT    = 7
ROLLING_LONG     = 30
BOLLINGER_WINDOW = 20
BOLLINGER_STD    = 2

# ── API Keys (loaded from .env) ───────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
NEWS_API_KEY      = os.getenv("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# ── LLM Settings ─────────────────────────────────────────────────────────────
GROQ_MODEL    = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_TOKENS    = 1500

# ── Chart Settings ────────────────────────────────────────────────────────────
CHART_DPI     = 150
CHART_STYLE   = "seaborn-v0_8"
CHART_FIGSIZE = (14, 6)