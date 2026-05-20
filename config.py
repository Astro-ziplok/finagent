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