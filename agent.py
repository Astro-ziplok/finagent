"""
agent.py — FinAgent Interactive Terminal Chat
=============================================
Primary entry point. Type natural language prompts and the agent
fetches data, generates charts, and produces AI analysis.

Data sources (all free):
  - Yahoo Finance (yfinance) → prices, financial statements, cash flow
  - Yahoo Finance RSS        → real-time news headlines (no API key)
  - Alpha Vantage            → news sentiment + company overview (25/day quota)
  - Groq                     → LLM analysis (free)

Supported queries:
  "Give me a full report on Nvidia"
  "Compare Apple and Microsoft"
  "What is the risk of TSLA?"
  "Show me fundamentals for AAPL"
  "Analyse income statement of Nvidia 2025 vs 2024"
  "Show me the balance sheet for MSFT"
  "Show me cash flow of Amazon"
  "Latest news for GOOGL"
  "Show me commodity prices"
  "What sectors should I invest in right now?"

Run with:
  python agent.py
"""

import json
import logging
import sys
import re
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from config import (GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL,
                    REPORTS_ANALYSIS_DIR, REPORTS_CHARTS_DIR)
from src.collection.fetch_data import fetch_stock_prices
from src.cleaning.clean_data import clean_stock_dataframe
from src.visualization.charts import (plot_price_and_volume, plot_bollinger_bands,
                                       plot_correlation_heatmap, plot_return_distributions,
                                       plot_comparative_returns, plot_fundamental_ratios,
                                       plot_financial_dashboard, plot_candlestick,
                                       generate_comparison_charts)
from src.analysis.ai_analysis import (generate_trend_summary, generate_anomaly_commentary,
                                       generate_risk_commentary, generate_comparative_analysis,
                                       generate_market_context, generate_fundamental_analysis,
                                       generate_news_sentiment_analysis, generate_commodity_context,
                                       generate_sector_analysis, fetch_sector_etf_data,
                                       generate_yfinance_income_analysis,
                                       generate_yfinance_balance_analysis,
                                       generate_yfinance_cashflow_analysis,
                                       generate_key_metrics_dashboard,
                                       generate_rss_news_analysis,
                                       _build_data_context, _call_groq)
from src.fundamentals.alpha_vantage import fetch_all_fundamental_data, fetch_commodity_prices
from src.fundamentals.yfinance_fundamentals import fetch_all_yfinance_fundamentals
from src.fundamentals.quota import QuotaManager
from src.export.report_exporter import export_report
from src.collection.fetch_data import fetch_rss_news

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

# ── In-memory cache — avoids re-downloading in the same session ───────────────
DATA_CACHE:       dict = {}   # ticker → cleaned DataFrame
FUNDAMENTAL_CACHE:dict = {}   # ticker → Alpha Vantage data (news, overview)
FINANCIAL_CACHE:  dict = {}   # ticker → yfinance financials (income, balance, cashflow)

# ── Quota manager — tracks Alpha Vantage daily usage ──────────────────────────
quota = QuotaManager()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1: Parse what the user wants
# ─────────────────────────────────────────────────────────────────────────────

def parse_user_intent(user_message: str) -> dict:
    """Use the LLM to extract tickers, intent, and optional year range from a message."""
    system_prompt = """
You are a financial assistant that extracts structured information from user messages.
Return ONLY a JSON object with:
- "tickers": list of stock ticker symbols. Convert company names:
    Apple→AAPL, Tesla→TSLA, Microsoft→MSFT, Google/Alphabet→GOOGL,
    Amazon→AMZN, Meta→META, Netflix→NFLX, Nvidia→NVDA, AMD→AMD,
    Intel→INTC, Samsung→005930.KS, Berkshire→BRK-B
- "intent": pick the BEST match from this list:
    "analyse"      → general stock analysis
    "compare"      → comparing two or more stocks side by side
    "risk"         → risk, volatility, drawdown, downside questions
    "trend"        → price trend, momentum, moving averages
    "full"         → full or complete report on a stock
    "news"         → news headlines, sentiment, recent coverage
    "fundamentals" → P/E, ROE, ROA, D/E, valuation ratios
    "income"       → income statement, revenue, profit, earnings, EPS, year-over-year
    "balance"      → balance sheet, assets, liabilities, debt, equity
    "cashflow"     → cash flow statement, free cash flow, FCF, operating cash flow
    "commodities"  → oil, gas, gold, copper, commodity prices (no ticker needed)
    "sectors"      → sector rotation, trending sectors, what to invest in
    "unknown"      → cannot determine
- "start_year": integer year if user specifies a historical start (e.g. 2020), else null
- "end_year": integer year if user specifies a historical end (e.g. 2023), else null

Classification rules (follow strictly):
- "what sectors", "sector rotation", "trending sectors" → "sectors" with empty tickers []
- "commodity", "oil price", "gold price", "gas price" → "commodities" with empty tickers []
- "cash flow", "cashflow", "free cash flow", "FCF", "operating cash" → "cashflow"
- "income statement", "revenue", "earnings", "profit 2024 vs 2025", "EPS" → "income"
- "balance sheet", "total assets", "liabilities", "debt levels" → "balance"
- "P/E ratio", "ROE", "ROA", "valuation" → "fundamentals"
- IMPORTANT: "cashflow" must NEVER be classified as "balance"
- Year range examples: "2020 to 2023" → start_year=2020, end_year=2023
                       "from 2019" → start_year=2019, end_year=null
                       "last 5 years" → start_year=2020 (current year minus 5), end_year=null

Return ONLY valid JSON. No markdown, no explanation.
Example: {"tickers": ["AAPL", "TSLA"], "intent": "compare", "start_year": 2020, "end_year": 2023}
"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=200,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        parsed["tickers"]    = [t.upper() for t in parsed.get("tickers", [])]
        parsed["intent"]     = parsed.get("intent", "unknown")
        parsed["start_year"] = parsed.get("start_year")
        parsed["end_year"]   = parsed.get("end_year")
        return parsed
    except Exception:
        # Fallback: extract capitalised words that look like tickers
        # Exclude common English words that are not tickers
        EXCLUDED = {"OF", "VS", "TO", "IN", "AT", "BY", "FOR", "AND", "OR",
                    "THE", "A", "AN", "FROM", "WITH", "ON", "IS", "IT",
                    "BE", "AS", "DO", "IF", "SO", "WE", "MY", "ME", "US"}
        raw_tickers = re.findall(r'\b[A-Z]{2,5}\b', user_message.upper())
        tickers = [t for t in raw_tickers if t not in EXCLUDED]

        # Simple keyword intent fallback
        msg_lower = user_message.lower()
        if any(w in msg_lower for w in ["income", "revenue", "earnings", "profit", "eps"]):
            fallback_intent = "income"
        elif any(w in msg_lower for w in ["cash flow", "cashflow", "fcf"]):
            fallback_intent = "cashflow"
        elif any(w in msg_lower for w in ["balance sheet", "assets", "liabilities", "debt"]):
            fallback_intent = "balance"
        elif any(w in msg_lower for w in ["compare", "vs", "versus"]):
            fallback_intent = "compare"
        elif any(w in msg_lower for w in ["risk", "volatility"]):
            fallback_intent = "risk"
        else:
            fallback_intent = "analyse"

        return {"tickers": tickers, "intent": fallback_intent,
                "start_year": None, "end_year": None}


def _year_range_to_period(start_year, end_year) -> str:
    """
    Convert a user-specified year range to a yfinance period string or
    start/end date tuple.

    Returns a dict with keys 'period' OR 'start'+'end' for yfinance.
    """
    from datetime import date
    current_year = date.today().year

    if start_year and end_year:
        return {
            "start": f"{start_year}-01-01",
            "end":   f"{end_year}-12-31",
        }
    elif start_year:
        years_back = current_year - start_year
        if years_back <= 1:   return {"period": "1y"}
        elif years_back <= 2: return {"period": "2y"}
        elif years_back <= 5: return {"period": "5y"}
        else:                 return {"period": "10y"}
    else:
        return {"period": "1y"}  # Default


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: Fetch data
# ─────────────────────────────────────────────────────────────────────────────

def get_price_data(tickers: list, start_year=None, end_year=None) -> dict:
    """
    Fetch and clean stock price data.
    Supports optional year range (e.g. start_year=2020, end_year=2023).
    Cache key includes the date range so different periods are cached separately.
    """
    date_params = _year_range_to_period(start_year, end_year)
    cache_suffix = f"_{start_year}_{end_year}" if (start_year or end_year) else ""

    result = {}
    for raw_ticker in tickers:
        ticker    = raw_ticker.strip().upper().replace("$", "")
        cache_key = f"{ticker}{cache_suffix}"

        if cache_key in DATA_CACHE:
            print(f"  📦 {ticker}: using cached data")
            result[ticker] = DATA_CACHE[cache_key]
        else:
            if start_year or end_year:
                print(f"  🌐 {ticker}: downloading data ({start_year or 'start'} → {end_year or 'now'})...")
            else:
                print(f"  🌐 {ticker}: downloading price data (1 year)...")

            raw = fetch_stock_prices([ticker], **date_params)
            if ticker not in raw:
                print(f"  ⚠️  No data for '{ticker}'. Check the symbol or turn on VPN.")
                continue
            cleaned = clean_stock_dataframe(raw[ticker], ticker)
            DATA_CACHE[cache_key] = cleaned
            result[ticker] = cleaned
            rng = f"{cleaned.index[0].date()} → {cleaned.index[-1].date()}"
            print(f"  ✅ {ticker}: {len(cleaned)} trading days ({rng})")
    return result


def get_fundamental_data(tickers: list, intent: str = "full") -> dict:
    """
    Fetch Alpha Vantage data with smart quota management.

    Before fetching, checks the remaining daily quota and only fetches
    features that can be afforded. Skips gracefully if quota is exhausted.
    """
def get_fundamental_data(tickers: list, intent: str = "full") -> dict:
    """
    Fetch Alpha Vantage data with smart quota management.
    Shows quota status before and after every call.
    Automatically skips features that would exceed the daily limit.
    """
    # Always show quota before any Alpha Vantage call
    print(quota.status_line())

    if quota.remaining == 0:
        print("  ❌ Alpha Vantage quota exhausted for today (25/25 used).")
        print("     ↳ Trend analysis and charts still work fine (Yahoo Finance).")
        print("     ↳ Quota resets at midnight UTC.\n")
        return {}

    # Use cache for already-fetched tickers
    uncached = [t for t in tickers if t not in FUNDAMENTAL_CACHE]
    if not uncached:
        print(f"  📦 Using cached fundamental data for {tickers}")
        result = {t: FUNDAMENTAL_CACHE[t] for t in tickers if t in FUNDAMENTAL_CACHE}
        if "commodities" in FUNDAMENTAL_CACHE:
            result["commodities"] = FUNDAMENTAL_CACHE["commodities"]
        return result

    # Decide which features we can afford
    features = quota.what_can_afford(n_tickers=len(uncached), intent=intent)

    if not features:
        print(f"  ⚠️  Only {quota.remaining} request(s) left — not enough for any features.")
        print("     ↳ Try again tomorrow, or register a new free key at alphavantage.co\n")
        return {}

    cost = len(uncached) * len(features)
    print(f"  🔍 Fetching {features} for {uncached} — costs {cost} request(s)...")

    new_data = fetch_all_fundamental_data(uncached, features=features)

    # Record usage and cache results
    quota.record(cost)
    for ticker in uncached:
        if ticker in new_data:
            FUNDAMENTAL_CACHE[ticker] = new_data[ticker]
    if "commodities" in new_data:
        FUNDAMENTAL_CACHE["commodities"] = new_data["commodities"]

    # Show updated quota after the call
    print(quota.status_line())

    result = {t: FUNDAMENTAL_CACHE[t] for t in tickers if t in FUNDAMENTAL_CACHE}
    if "commodities" in FUNDAMENTAL_CACHE:
        result["commodities"] = FUNDAMENTAL_CACHE["commodities"]
    return result


def get_yfinance_fundamentals(tickers: list) -> dict:
    """
    Fetch financial statements from Yahoo Finance (free, no quota).
    Includes income statement, balance sheet, cash flow, and key metrics.
    Uses cache to avoid re-fetching in the same session.
    """
    result = {}
    for ticker in tickers:
        cache_key = f"yf_{ticker}"
        if cache_key in FUNDAMENTAL_CACHE:
            print(f"  📦 {ticker}: using cached yfinance fundamentals")
            result[ticker] = FUNDAMENTAL_CACHE[cache_key]
        else:
            print(f"  🌐 {ticker}: fetching financial statements from Yahoo Finance...")
            data = fetch_all_yfinance_fundamentals(ticker)
            FUNDAMENTAL_CACHE[cache_key] = data
            result[ticker] = data
            has = [k for k in ["income", "balance", "cashflow", "metrics"]
                   if data.get(k)]
            print(f"  ✅ {ticker}: loaded {', '.join(has)}")
    return result


def get_rss_news(tickers: list):
    """Fetch free RSS news for the given tickers (no API key needed)."""
    cache_key = f"rss_{'_'.join(tickers)}"
    if cache_key in FUNDAMENTAL_CACHE:
        print(f"  📦 RSS news: using cached data")
        return FUNDAMENTAL_CACHE[cache_key]
    print(f"  🌐 Fetching RSS news for {tickers} (free, no key needed)...")
    df = fetch_rss_news(tickers, max_articles=5)
    if df is not None and not df.empty:
        FUNDAMENTAL_CACHE[cache_key] = df
        print(f"  ✅ RSS news: {len(df)} articles fetched")
    return df


def get_commodity_data() -> dict:
    """
    Fetch commodity prices using Yahoo Finance futures (CL=F, NG=F, GC=F, HG=F).
    Completely FREE — uses no Alpha Vantage quota whatsoever.
    Uses in-memory cache so repeated queries don't re-download.
    """
    if "commodities" in FUNDAMENTAL_CACHE:
        print("  📦 Commodities: using cached data")
        return FUNDAMENTAL_CACHE["commodities"]

    print("  🌐 Fetching commodity prices from Yahoo Finance (free, no quota)...")
    data = fetch_commodity_prices()

    if data:
        FUNDAMENTAL_CACHE["commodities"] = data
        available = [k for k, v in data.items() if v.get("latest_price") is not None]
        print(f"  ✅ Commodity data loaded: {', '.join(available)}")
    else:
        print("  ⚠️  Could not fetch commodity data. Check your internet connection or VPN.")

    return data or {}

    return data or {}


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: Generate charts
# ─────────────────────────────────────────────────────────────────────────────

def generate_charts(cleaned_data: dict, fundamental_data: dict = None,
                    intent: str = "") -> list:
    """Generate all charts and save to reports/charts/."""
    if not cleaned_data:
        return []

    saved = []

    # Per-ticker charts — always generated
    for ticker, df in cleaned_data.items():
        saved.append(plot_price_and_volume(df, ticker))
        saved.append(plot_bollinger_bands(df, ticker))
        # Bonus: candlestick chart (last 90 days)
        path = plot_candlestick(df, ticker)
        if path:
            saved.append(path)

    # Cross-ticker standard charts
    if len(cleaned_data) > 1:
        saved.append(plot_correlation_heatmap(cleaned_data))
        saved.append(plot_comparative_returns(cleaned_data))

    saved.append(plot_return_distributions(cleaned_data))

    # Fundamental ratios chart (Alpha Vantage overview data)
    if fundamental_data:
        av_data = fundamental_data.get("av", {})
        if av_data and isinstance(av_data, dict):
            path = plot_fundamental_ratios(av_data)
            if path:
                saved.append(path)

    # Financial dashboard — year-by-year for income/cashflow/balance/full
    if intent in ("income", "balance", "cashflow", "full", "fundamentals"):
        yf_data = (fundamental_data or {}).get("yfinance", {})
        for ticker, td in yf_data.items():
            if isinstance(td, dict):
                path = plot_financial_dashboard(ticker, td)
                if path:
                    saved.append(path)

    # Comparison charts — only for compare intent with 2+ tickers
    if intent == "compare" and len(cleaned_data) >= 2:
        yf_data = (fundamental_data or {}).get("yfinance", {})
        print("  📊 Generating comparison charts (C1–C4)...")
        comparison_paths = generate_comparison_charts(cleaned_data, yf_data)
        saved.extend(comparison_paths)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4: Run AI analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(cleaned_data: dict, intent: str,
                 fundamental_data: dict = None,
                 start_year: int = None, end_year: int = None) -> str:
    """Route the query to the correct analysis functions based on intent."""
    output = []

    # Defensive guard — ensure fundamental_data is always a dict with safe defaults
    if not isinstance(fundamental_data, dict):
        fundamental_data = {}
    fundamental_data.setdefault("yfinance", {})
    fundamental_data.setdefault("av", {})
    # rss_news can be None or a DataFrame — leave it as-is

    # ── COMMODITIES ───────────────────────────────────────────────────────────
    if intent == "commodities":
        output.append("\n── COMMODITY PRICES & MACRO CONTEXT ────────────────")
        commodity_data = (fundamental_data or {}).get("commodities")
        if commodity_data:
            output.append(generate_commodity_context(commodity_data))
        else:
            output.append("⚠️  No commodity data available. Check ALPHA_VANTAGE_KEY or quota.")
        return "\n".join(output)

    # ── SECTORS ───────────────────────────────────────────────────────────────
    if intent == "sectors":
        output.append("\n── SECTOR ROTATION & TRENDING ANALYSIS ─────────────")
        print("  📊 Fetching live sector ETF data (Yahoo Finance, ~30 seconds)...")
        sector_data    = fetch_sector_etf_data()
        commodity_data = (fundamental_data or {}).get("commodities")
        output.append(generate_sector_analysis(sector_data, commodity_data))
        return "\n".join(output)

    # ── All other intents require price data ──────────────────────────────────
    if not cleaned_data:
        return "❌ No price data available. Check ticker symbol or internet connection."

    context = _build_data_context(cleaned_data)
    tickers = list(cleaned_data.keys())

    # ── FULL / ANALYSE / UNKNOWN ──────────────────────────────────────────────
    if intent in ("analyse", "full", "unknown"):
        yf_data  = (fundamental_data or {}).get("yfinance", {})
        rss_news = (fundamental_data or {}).get("rss_news")

        for ticker in tickers:
            output.append(f"\n── {ticker} TREND ANALYSIS ──────────────────────────")
            output.append(generate_trend_summary(ticker, context))
            output.append(f"\n── {ticker} MARKET CONTEXT ──────────────────────────")
            output.append(generate_market_context(ticker, context))
            output.append(generate_anomaly_commentary(ticker, context))

            # Key metrics dashboard (yfinance — always free)
            yf_ticker = yf_data.get(ticker, {})
            if yf_ticker.get("metrics"):
                output.append(f"\n── {ticker} KEY METRICS DASHBOARD ──────────────")
                output.append(generate_key_metrics_dashboard(ticker, yf_ticker["metrics"]))

            # Full financial statements (yfinance — always free)
            if intent == "full":
                if yf_ticker.get("income"):
                    output.append(f"\n── {ticker} INCOME STATEMENT ──────────────────")
                    output.append(generate_yfinance_income_analysis(ticker, yf_ticker["income"]))
                if yf_ticker.get("balance"):
                    output.append(f"\n── {ticker} BALANCE SHEET ─────────────────────")
                    output.append(generate_yfinance_balance_analysis(ticker, yf_ticker["balance"]))
                if yf_ticker.get("cashflow"):
                    output.append(f"\n── {ticker} CASH FLOW STATEMENT ───────────────")
                    output.append(generate_yfinance_cashflow_analysis(ticker, yf_ticker["cashflow"]))

            # Alpha Vantage news sentiment (if available)
            av_data = (fundamental_data or {}).get("av", {})
            if av_data.get(ticker, {}).get("news", {}).get("articles"):
                output.append(f"\n── {ticker} NEWS SENTIMENT (Alpha Vantage) ───────")
                output.append(generate_news_sentiment_analysis(
                    ticker, av_data[ticker]["news"]))

            # RSS news (free, always available)
            if rss_news is not None:
                output.append(f"\n── {ticker} MARKET NEWS (RSS) ────────────────────")
                output.append(generate_rss_news_analysis(ticker, rss_news))

        output.append("\n── RISK COMMENTARY ─────────────────────────────────")
        output.append(generate_risk_commentary(context))
        output.append("\n── COMPARATIVE / SECTOR ANALYSIS ───────────────────")
        output.append(generate_comparative_analysis(context))

        if (fundamental_data or {}).get("commodities"):
            output.append("\n── COMMODITY MACRO CONTEXT ─────────────────────────")
            sector_info = {}
            for t in tickers:
                m = (fundamental_data or {}).get("yfinance", {}).get(t, {}).get("metrics", {})
                sector_info[t] = m.get("sector", "Unknown")
            output.append(generate_commodity_context(
                fundamental_data["commodities"],
                tickers=tickers,
                sector_info=sector_info,
            ))

    # ── TREND ─────────────────────────────────────────────────────────────────
    elif intent == "trend":
        for ticker in tickers:
            output.append(f"\n── {ticker} TREND ────────────────────────────────")
            output.append(generate_trend_summary(ticker, context))

    # ── RISK ──────────────────────────────────────────────────────────────────
    elif intent == "risk":
        output.append("\n── RISK COMMENTARY ───────────────────────────────")
        output.append(generate_risk_commentary(context))
        for ticker in tickers:
            output.append(f"\n── {ticker} ANOMALIES ─────────────────────────────")
            output.append(generate_anomaly_commentary(ticker, context))

    # ── COMPARE ───────────────────────────────────────────────────────────────
    elif intent == "compare":
        if len(tickers) < 2:
            output.append("⚠️  Please mention at least 2 stocks to compare.")
            output.append("   Example: 'Compare Apple and Microsoft'")
        else:
            output.append("\n── COMPARATIVE ANALYSIS ──────────────────────────")
            output.append(generate_comparative_analysis(context))
            output.append("\n── RISK COMPARISON ───────────────────────────────")
            output.append(generate_risk_commentary(context))

    # ── FUNDAMENTALS ──────────────────────────────────────────────────────────
    elif intent == "fundamentals":
        if not fundamental_data:
            output.append("⚠️  No fundamental data available.")
        else:
            for ticker in tickers:
                td = fundamental_data.get(ticker, {})
                if td.get("overview"):
                    output.append(f"\n── {ticker} FUNDAMENTALS ──────────────────────")
                    output.append(generate_fundamental_analysis(
                        ticker, td.get("overview", {}),
                        td.get("income", {}), td.get("balance", {})))
                else:
                    output.append(f"⚠️  No fundamental data for {ticker}.")

    # ── NEWS ──────────────────────────────────────────────────────────────────
    elif intent == "news":
        if not fundamental_data:
            output.append("⚠️  No news data available.")
        else:
            for ticker in tickers:
                td = fundamental_data.get(ticker, {})
                if td.get("news", {}).get("articles"):
                    output.append(f"\n── {ticker} NEWS SENTIMENT ──────────────────")
                    output.append(generate_news_sentiment_analysis(ticker, td["news"]))
                else:
                    output.append(f"⚠️  No news data for {ticker}.")

    # ── INCOME STATEMENT ──────────────────────────────────────────────────────
    elif intent == "income":
        yf_data = fundamental_data.get("yfinance", {}) if fundamental_data else {}
        for ticker in tickers:
            td = yf_data.get(ticker, {})
            income = td.get("income", {})
            if not income:
                output.append(f"⚠️  No income statement data for {ticker}.")
                continue
            # Filter annual data to requested year range
            if start_year or end_year:
                income = dict(income)  # shallow copy
                annual = income.get("annual", [])
                filtered = [r for r in annual if
                            (not start_year or int(r["period"][:4]) >= start_year) and
                            (not end_year   or int(r["period"][:4]) <= end_year)]
                if filtered:
                    income = {**income, "annual": filtered}
                    # Recalculate YoY with filtered data
                    if len(filtered) >= 2:
                        from src.fundamentals.yfinance_fundamentals import _pct_change
                        income["yoy_growth"] = {
                            "revenue_growth_pct":    _pct_change(
                                filtered[0].get("total_revenue"),
                                filtered[1].get("total_revenue")),
                            "net_income_growth_pct": _pct_change(
                                filtered[0].get("net_income"),
                                filtered[1].get("net_income")),
                        }
            label = f"{start_year}–{end_year}" if (start_year and end_year) else ticker
            output.append(f"\n── {ticker} INCOME STATEMENT ({label}) ────────────")
            output.append(generate_yfinance_income_analysis(ticker, income))

    # ── BALANCE SHEET ─────────────────────────────────────────────────────────
    elif intent == "balance":
        yf_data = fundamental_data.get("yfinance", {}) if fundamental_data else {}
        for ticker in tickers:
            td = yf_data.get(ticker, {})
            balance = td.get("balance", {})
            if not balance:
                output.append(f"⚠️  No balance sheet data for {ticker}.")
                continue
            output.append(f"\n── {ticker} BALANCE SHEET ─────────────────────────")
            output.append(generate_yfinance_balance_analysis(ticker, balance))

    # ── CASH FLOW ─────────────────────────────────────────────────────────────
    elif intent == "cashflow":
        yf_data = fundamental_data.get("yfinance", {}) if fundamental_data else {}
        for ticker in tickers:
            td = yf_data.get(ticker, {})
            cashflow = td.get("cashflow", {})
            if not cashflow:
                output.append(f"⚠️  No cash flow data for {ticker}.")
                continue
            output.append(f"\n── {ticker} CASH FLOW STATEMENT ───────────────────")
            output.append(generate_yfinance_cashflow_analysis(ticker, cashflow))

    return "\n".join(output)


# ─────────────────────────────────────────────────────────────────────────────
#  Financial Statement Analysis Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_income_statement(ticker: str, income: dict, overview: dict) -> str:
    quarters = income.get("quarterly_income", [])
    if not quarters:
        return f"⚠️  No quarterly income data available for {ticker}."

    system_prompt = """
You are a financial analyst writing an income statement review for a university finance course.
Use ONLY the exact numbers provided. Never invent figures.
Format large numbers clearly (e.g. $2.3B, $450M). Write professionally and specifically.
"""
    prompt = f"""
Analyse the income statement for {ticker} using the quarterly data below.

QUARTERLY INCOME DATA (most recent first):
{json.dumps(quarters, indent=2)}

CONTEXT: Sector={overview.get('sector','N/A')}, Profit Margin={overview.get('profit_margin','N/A')}, Operating Margin={overview.get('operating_margin','N/A')}

Write using EXACTLY these section headers:

**Revenue Trend**
Compare total_revenue across all quarters. Calculate the QoQ change between the two most recent quarters.

**Profitability Analysis**
Compare gross_profit and net_income. Calculate gross profit margin for the latest quarter.
Is net income growing in line with revenue?

**Operating Performance**
Discuss operating_income trend. Comment on R&D spend relative to revenue if available.

**Year-over-Year Comparison**
Compare the most recent quarter to the same quarter one year ago (if 4 quarters available).

**Key Takeaway**
2-3 sentences summarising financial health. Reference at least 2 specific numbers.
"""
    return _call_groq(prompt, system_prompt, max_tokens=900)


def _analyse_balance_sheet(ticker: str, balance: dict, overview: dict) -> str:
    quarters = balance.get("quarterly_balance", [])
    if not quarters:
        return f"⚠️  No quarterly balance sheet data available for {ticker}."

    system_prompt = """
You are a financial analyst writing a balance sheet review for a university finance course.
Use ONLY the exact numbers provided. Never invent figures.
Format large numbers clearly (e.g. $2.3B, $450M). Write professionally and specifically.
"""
    prompt = f"""
Analyse the balance sheet for {ticker} using the quarterly data below.

QUARTERLY BALANCE SHEET DATA (most recent first):
{json.dumps(quarters, indent=2)}

CONTEXT: D/E={overview.get('debt_to_equity','N/A')}, Current Ratio={overview.get('current_ratio','N/A')}, Quick Ratio={overview.get('quick_ratio','N/A')}

Write using EXACTLY these section headers:

**Asset Base**
State total_assets for the most recent quarter. Is it growing? What does that suggest?

**Debt & Liabilities**
State total_liabilities and long_term_debt. Calculate debt-to-assets ratio. Is debt rising or falling?

**Shareholder Equity**
State shareholder_equity and compare to prior quarters. Reference retained_earnings if available.

**Liquidity & Stability**
Use current_ratio and quick_ratio to assess short-term solvency. Comment on cash vs debt.

**Key Takeaway**
2-3 sentences summarising financial strength. Reference at least 2 specific numbers.
"""
    return _call_groq(prompt, system_prompt, max_tokens=900)


# ─────────────────────────────────────────────────────────────────────────────
#  Save analysis report
# ─────────────────────────────────────────────────────────────────────────────

def save_analysis(tickers: list, analysis_text: str) -> Path:
    """
    Save analysis to reports/analysis/analysis_TICKER.txt
    Wraps long lines at 100 characters so the file is readable
    in any text editor without horizontal scrolling.
    """
    import textwrap

    label    = "_".join(tickers) if tickers else "no_ticker"
    filename = f"analysis_{label}.txt"
    path     = REPORTS_ANALYSIS_DIR / filename

    # Wrap each line individually so section headers and short lines are preserved
    wrapped_lines = []
    for line in analysis_text.split("\n"):
        # Don't wrap section dividers, headers, or short lines
        if (line.startswith("──") or line.startswith("==") or
                line.startswith("**") or len(line) <= 100):
            wrapped_lines.append(line)
        else:
            # Wrap long paragraph lines at 100 chars, preserving indentation
            indent = len(line) - len(line.lstrip())
            wrapped = textwrap.fill(
                line.strip(),
                width=100,
                initial_indent=" " * indent,
                subsequent_indent=" " * indent,
            )
            wrapped_lines.append(wrapped)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(wrapped_lines))

    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 60)
    print("  📈 FINAGENT — AI-Powered Financial Agent")
    print("=" * 60)
    print("  Examples:")
    print('    "Give me a full report on Nvidia"')
    print('    "Compare Apple and Microsoft"')
    print('    "What is the risk of TSLA?"')
    print('    "Show me fundamentals for AAPL"')
    print('    "Analyse income statement of Nvidia 2025 vs 2024"')
    print('    "Show me the balance sheet for MSFT"')
    print('    "Show me cash flow for Amazon"')
    print('    "Latest news for GOOGL"')
    print('    "Show me commodity prices"')
    print('    "What sectors should I invest in right now?"')
    print('    "Compare AAPL and MSFT from 2020 to 2023"')
    print('    "Analyse TSLA from 2019"')
    print("\n  Type 'help' to see this again. Type 'quit' to exit.")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main Chat Loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_banner()

    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY not found in .env file.")
        print("   Open .env and add: GROQ_API_KEY=your_key_here")
        sys.exit(1)

    # Show quota on startup so user knows their budget for the session
    print(quota.status_line() + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye! Results saved to reports/")
                break
            if user_input.lower() in ("help", "?"):
                print_banner()
                continue
            if user_input.lower() in ("quota", "quota status"):
                print(quota.status_line() + "\n")
                continue

            print("\n🤖 Agent: Let me look into that...\n")

            # ── Step 1: Parse intent ───────────────────────────────────────
            intent_data = parse_user_intent(user_input)
            tickers     = intent_data.get("tickers", [])
            intent      = intent_data.get("intent", "unknown")
            start_year  = intent_data.get("start_year")
            end_year    = intent_data.get("end_year")

            year_label = ""
            if start_year or end_year:
                year_label = f" | Range: {start_year or 'start'} → {end_year or 'now'}"

            print(f"  📋 Intent: {intent}"
                  + (f" | Tickers: {tickers}" if tickers else "")
                  + year_label + "\n")

            # ── Step 2: Handle no-ticker intents ──────────────────────────
            if intent in ("commodities", "sectors") and not tickers:
                if intent == "commodities":
                    commodity_data = get_commodity_data()
                    fd = {"commodities": commodity_data} if commodity_data else {}
                else:
                    commodity_data = FUNDAMENTAL_CACHE.get("commodities")
                    fd = {"commodities": commodity_data} if commodity_data else {}

                analysis = run_analysis({}, intent, fd)
                print("=" * 60)
                print(analysis)
                print("=" * 60 + "\n")
                save_analysis([], analysis)
                continue

            # ── Step 3: Validate tickers ───────────────────────────────────
            if not tickers:
                print("🤖 Agent: I couldn't find any stock names in your message.")
                print("   Try: 'Analyse Apple' or 'Compare TSLA and MSFT'\n")
                continue

            # ── Step 4: Fetch price data (with optional year range) ────────
            cleaned_data = get_price_data(tickers,
                                          start_year=start_year,
                                          end_year=end_year)
            if not cleaned_data:
                print("🤖 Agent: Sorry, I couldn't retrieve price data for those tickers.")
                print("   If you're in Vietnam, try turning on a VPN.\n")
                continue

            # ── Step 5: Fetch data from all free sources ───────────────────
            fundamental_data = {"yfinance": {}, "av": {}, "rss_news": None}

            # yfinance financial statements — always free, no quota
            if intent in ("full", "fundamentals", "income", "balance",
                          "cashflow", "analyse", "unknown", "compare"):
                print("  📊 Fetching financial statements (Yahoo Finance, free)...")
                fundamental_data["yfinance"] = get_yfinance_fundamentals(tickers)

            # RSS news — always free, no API key needed
            if intent in ("full", "news", "analyse", "unknown"):
                fundamental_data["rss_news"] = get_rss_news(tickers)

            # Alpha Vantage — only for news sentiment and overview (quota-limited)
            if intent in ("full", "news") and quota.remaining >= len(tickers):
                fundamental_data["av"] = get_fundamental_data(tickers, intent="news")

            # Commodities — Yahoo Finance futures, always free
            if intent == "full":
                fundamental_data["commodities"] = get_commodity_data()

            # ── Step 6: Generate charts ────────────────────────────────────
            print("  📊 Generating charts...")
            chart_paths = generate_charts(cleaned_data, fundamental_data, intent=intent)
            print(f"  ✅ {len(chart_paths)} chart(s) saved to reports/charts/\n")

            # ── Step 7: Run AI analysis ────────────────────────────────────
            print("  🧠 Running AI analysis...\n")
            analysis = run_analysis(cleaned_data, intent, fundamental_data,
                                    start_year=start_year, end_year=end_year)

            # ── Step 8: Print and save ─────────────────────────────────────
            print("=" * 60)
            print(analysis)
            print("=" * 60 + "\n")

            saved_path = save_analysis(tickers, analysis)
            print(f"  Saved to reports/analysis/{saved_path.name}\n")

            # ── Step 9: Offer PDF export ───────────────────────────────────
            # Ask user if they want to export the analysis + charts as a PDF
            print("  Export this analysis as a PDF report? (yes/no): ", end="")
            try:
                export_choice = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                export_choice = "no"

            if export_choice in ("yes", "y"):
                print("  Generating PDF report (this may take 15-30 seconds)...")
                pdf_path = export_report(
                    tickers=tickers,
                    analysis_text=analysis,
                    chart_paths=chart_paths,
                )
                if pdf_path:
                    ext = "PDF" if pdf_path.endswith(".pdf") else "Word document"
                    print(f"  {ext} saved: {Path(pdf_path).name}\n")
                else:
                    print("  Export failed. Check that python-docx is installed.\n")
            else:
                print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Something went wrong: {e}")
            print("   Try again with a different query.\n")


if __name__ == "__main__":
    main()