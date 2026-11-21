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

# ── Risk / Benchmark Settings ─────────────────────────────────────────────────
# Annual risk-free rate used in the Sharpe ratio (approx. 3-month US T-bill yield).
# Update this occasionally to reflect current rates.
RISK_FREE_RATE   = 0.04
TRADING_DAYS     = 252
# Market benchmark used for excess return, beta and correlation.
BENCHMARK_TICKER = "SPY"

# ── Valuation Assumptions ─────────────────────────────────────────────────────
# Every assumption below is printed with its source in each research report.
# Change a value here and the report's citation table changes with it.
VALUATION_ASSUMPTIONS = {
    "equity_risk_premium": {
        "value": 0.050,
        "source": ("FinAgent default. Consistent with Kroll's recommended U.S. equity risk premium "
                   "of 5.0% (2024) and within Damodaran's implied ERP range for the S&P 500 "
                   "(about 4-5% in recent years)."),
    },
    "terminal_growth": {
        "value": 0.025,
        "source": ("FinAgent default. Conservative versus long-run U.S. nominal GDP growth of about "
                   "3.8% (Federal Reserve 2% inflation objective + CBO long-term potential real "
                   "growth of roughly 1.8%)."),
    },
    "forecast_years": {
        "value": 10,
        "source": ("FinAgent default. Ten-year explicit forecast (as used by Damodaran for firms with "
                   "durable growth); years 1-2 follow analyst consensus, later years fade linearly to "
                   "terminal growth, avoiding an abrupt drop to terminal growth."),
    },
    "beta_adjustment": {
        "value": 0.67,
        "source": ("Blume (1971) adjustment toward the market beta of 1.0, as reported by Bloomberg: "
                   "adjusted beta = 0.67 x raw beta + 0.33."),
    },
    "risk_free_fallback": {
        "value": 0.045,
        "source": "FinAgent fallback, used only if the live 10-year Treasury yield cannot be fetched.",
    },
    "credit_spread_fallback": {
        "value": 0.010,
        "source": ("FinAgent fallback: risk-free rate + 1.0% (typical investment-grade spread), used "
                   "when interest expense is not reported separately."),
    },
    "tax_rate_fallback": {
        "value": 0.21,
        "source": "U.S. federal statutory corporate tax rate (Tax Cuts and Jobs Act, 2017).",
    },
    "scenario_growth_shift": {
        "value": 0.03,
        "source": "FinAgent default: bull/bear cases add/subtract 3 percentage points of revenue growth each year.",
    },
    "scenario_margin_shift": {
        "value": 0.10,
        "source": "FinAgent default: bull/bear cases raise/lower the free-cash-flow margin by 10% (relative).",
    },
}

# Which methods value which kind of company, and how they are weighted in the
# fair-value estimate. Weights are re-scaled if a method cannot be computed.
VALUATION_WEIGHTS = {
    "cash_generative": {"dcf": 0.5, "comps": 0.5},
    "financial":       {"ri": 0.5, "comps": 0.5},
    "dividend":        {"ddm": 0.4, "dcf": 0.3, "comps": 0.3},
    "high_growth":     {"comps": 1.0},
}
DIVIDEND_MIN_YIELD  = 0.020   # dividend payer = yield >= 2.0% ...
DIVIDEND_MIN_PAYOUT = 0.40    # ... and payout ratio >= 40%
FINANCIAL_INDUSTRY_KEYWORDS = ("Banks", "Insurance", "Capital Markets", "Asset Management", "Mortgage")

# ── Peer Groups for Comparable-Company Valuation ──────────────────────────────
# Default: companies in the same Yahoo Finance industry (labels verified Nov 2026,
# listed largest first). If an industry has fewer than 3 peers, the peer set is
# widened to the largest companies in the same sector. Edit freely.
INDUSTRY_PEERS = {
    ("Technology", "Semiconductors"): ["NVDA", "TSM", "AVGO", "AMD", "MU", "INTC", "TXN", "ADI", "ARM", "QCOM", "MRVL"],
    ("Technology", "Semiconductor Equipment & Materials"): ["ASML"],
    ("Technology", "Software - Infrastructure"): ["MSFT", "PLTR", "ORCL", "PANW", "CRWD", "FTNT", "SNPS"],
    ("Technology", "Software - Application"): ["SAP", "SHOP", "UBER", "CRM", "NOW", "ADBE", "CDNS", "INTU"],
    ("Technology", "Consumer Electronics"): ["AAPL", "SONY"],
    ("Technology", "Computer Hardware"): ["ANET", "DELL", "HPQ", "SMCI"],
    ("Technology", "Communication Equipment"): ["CSCO", "HPE"],
    ("Technology", "Information Technology Services"): ["IBM", "FIS"],
    ("Communication Services", "Internet Content & Information"): ["GOOGL", "META", "SPOT"],
    ("Communication Services", "Entertainment"): ["NFLX", "DIS", "WBD", "ROKU"],
    ("Communication Services", "Telecom Services"): ["T", "VZ", "TMUS", "CMCSA", "CHTR"],
    ("Consumer Cyclical", "Internet Retail"): ["AMZN", "BABA", "PDD", "MELI", "EBAY", "JD", "ETSY"],
    ("Consumer Cyclical", "Auto Manufacturers"): ["TSLA", "TM", "GM", "F", "HMC", "RIVN", "STLA"],
    ("Consumer Cyclical", "Restaurants"): ["MCD", "SBUX", "CMG"],
    ("Consumer Cyclical", "Home Improvement Retail"): ["HD", "LOW"],
    ("Consumer Cyclical", "Travel Services"): ["BKNG", "ABNB"],
    ("Consumer Defensive", "Discount Stores"): ["WMT", "COST", "TGT"],
    ("Consumer Defensive", "Beverages - Non-Alcoholic"): ["KO", "PEP"],
    ("Consumer Defensive", "Household & Personal Products"): ["PG", "CL"],
    ("Consumer Defensive", "Tobacco"): ["PM", "MO"],
    ("Financial Services", "Banks - Diversified"): ["JPM", "BAC", "WFC", "C"],
    ("Financial Services", "Banks - Regional"): ["USB", "PNC", "TFC"],
    ("Financial Services", "Capital Markets"): ["MS", "GS", "SCHW"],
    ("Financial Services", "Credit Services"): ["V", "MA", "AXP", "COF", "PYPL", "SYF"],
    ("Healthcare", "Drug Manufacturers - General"): ["LLY", "JNJ", "ABBV", "MRK", "AMGN", "PFE"],
    ("Energy", "Oil & Gas Integrated"): ["XOM", "CVX", "SHEL", "BP"],
    ("Industrials", "Aerospace & Defense"): ["GE", "RTX", "LMT", "BA"],
    ("Industrials", "Farm & Heavy Construction Machinery"): ["CAT", "DE"],
    ("Utilities", "Utilities - Regulated Electric"): ["NEE", "SO", "DUK", "AEP", "D"],
}

# Hand-picked peer sets where the Yahoo industry would mislead, with the reason
# printed in the report.
TICKER_PEERS = {
    "AAPL": (["SONY", "DELL", "HPQ", "MSFT", "GOOGL", "META"],
             "Yahoo's Consumer Electronics industry has only one comparable US-listed peer (SONY); "
             "widened to hardware makers (DELL, HPQ) and mega-cap consumer technology platforms."),
    "GOOGL": (["META", "SPOT", "MSFT", "AMZN", "NFLX"],
              "Internet Content has only two other large peers; widened to mega-cap platforms "
              "competing for digital advertising, cloud and consumer attention."),
    "MSFT": (["ORCL", "SAP", "CRM", "ADBE", "NOW", "GOOGL", "AMZN"],
             "Mature large-cap software and cloud platforms. High-growth security names in Yahoo's "
             "Software - Infrastructure industry (PLTR, CRWD, PANW) are excluded: their multiples "
             "reflect much faster expected growth."),
    "META": (["GOOGL", "SPOT", "SNAP", "PINS", "MSFT"],
             "Digital-advertising platforms (Internet Content) plus MSFT as a mega-cap platform."),
    "MA": (["V", "AXP", "PYPL"],
           "Payment networks and processors. Card lenders (COF, SYF) are excluded: they carry "
           "credit risk on their balance sheet and trade at very different multiples."),
    "V": (["MA", "AXP", "PYPL"],
          "Payment networks and processors. Card lenders (COF, SYF) are excluded: they carry "
          "credit risk on their balance sheet and trade at very different multiples."),
}
MIN_PEERS = 3

# ── API Keys (loaded from .env) ───────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
NEWS_API_KEY      = os.getenv("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# ── LLM Settings ─────────────────────────────────────────────────────────────
# llama-3.3-70b-versatile has been retired by Groq. gpt-oss-120b is a reasoning
# model: its hidden reasoning counts toward max_tokens, so every call adds
# REASONING_TOKEN_BUFFER on top of the visible answer budget.
# Override the model by setting GROQ_MODEL=... in .env.
GROQ_MODEL    = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_TOKENS    = 1500
REASONING_TOKEN_BUFFER = 1024
# Extra request options for reasoning models ("low" keeps latency and token use down).
GROQ_EXTRA_BODY = {"reasoning_effort": "low"} if "gpt-oss" in GROQ_MODEL else {}
# Groq tokens-per-minute cap for your account tier (free tier: 8000). The research
# report sizes its requests to fit under it; raise it in .env if you upgrade.
GROQ_TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "8000"))

# ── Chart Settings ────────────────────────────────────────────────────────────
CHART_DPI     = 150
CHART_STYLE   = "seaborn-v0_8"
CHART_FIGSIZE = (14, 6)