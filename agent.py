"""
agent.py - FinAgent Interactive Terminal Chat
=============================================
Primary entry point. Type natural language prompts and the agent
fetches data, generates charts, and produces AI analysis.
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
from src.collection.fetch_data import fetch_stock_prices, fetch_rss_news
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

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

DATA_CACHE:        dict = {}
FUNDAMENTAL_CACHE: dict = {}
quota = QuotaManager()


def parse_user_intent(user_message: str) -> dict:
    system_prompt = """
You are a financial assistant extracting structured info from user messages.
Return ONLY a JSON object with:
- "tickers": list of ticker symbols (convert Apple->AAPL, Tesla->TSLA, Microsoft->MSFT, Google->GOOGL, Amazon->AMZN, Meta->META, Netflix->NFLX, Nvidia->NVDA)
- "intent": one of: analyse, compare, risk, trend, full, news, fundamentals, income, balance, cashflow, commodities, sectors, unknown
- "start_year": integer or null
- "end_year": integer or null

Rules:
- "cash flow","cashflow","FCF" -> "cashflow" (NEVER "balance")
- "income statement","revenue","earnings" -> "income"
- "balance sheet","assets","liabilities" -> "balance"
- "commodity","oil price","gold" -> "commodities" with empty tickers []
- "sectors","sector rotation" -> "sectors" with empty tickers []

Return ONLY valid JSON. No markdown.
"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": user_message}],
            max_tokens=200, temperature=0,
        )
        raw    = response.choices[0].message.content.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        parsed["tickers"]    = [t.upper() for t in parsed.get("tickers", [])]
        parsed["intent"]     = parsed.get("intent", "unknown")
        parsed["start_year"] = parsed.get("start_year")
        parsed["end_year"]   = parsed.get("end_year")
        return parsed
    except Exception:
        EXCLUDED = {"OF","VS","TO","IN","AT","BY","FOR","AND","OR","THE","A","AN",
                    "FROM","WITH","ON","IS","IT","BE","AS","DO","IF","SO","WE","MY","ME","US"}
        raw_tickers = re.findall(r'\b[A-Z]{2,5}\b', user_message.upper())
        tickers     = [t for t in raw_tickers if t not in EXCLUDED]
        msg = user_message.lower()
        if any(w in msg for w in ["income","revenue","earnings","profit","eps"]):
            fi = "income"
        elif any(w in msg for w in ["cash flow","cashflow","fcf"]):
            fi = "cashflow"
        elif any(w in msg for w in ["balance sheet","assets","liabilities"]):
            fi = "balance"
        elif any(w in msg for w in ["compare","vs","versus"]):
            fi = "compare"
        elif any(w in msg for w in ["risk","volatility"]):
            fi = "risk"
        else:
            fi = "analyse"
        return {"tickers": tickers, "intent": fi, "start_year": None, "end_year": None}


def _year_range_to_period(start_year, end_year) -> dict:
    from datetime import date
    cy = date.today().year
    if start_year and end_year:
        return {"start": f"{start_year}-01-01", "end": f"{end_year}-12-31"}
    elif start_year:
        yb = cy - start_year
        if yb <= 1:   return {"period": "1y"}
        elif yb <= 2: return {"period": "2y"}
        elif yb <= 5: return {"period": "5y"}
        else:         return {"period": "10y"}
    return {"period": "1y"}


def get_price_data(tickers, start_year=None, end_year=None):
    date_params  = _year_range_to_period(start_year, end_year)
    cache_suffix = f"_{start_year}_{end_year}" if (start_year or end_year) else ""
    result = {}
    for raw_ticker in tickers:
        ticker    = raw_ticker.strip().upper().replace("$", "")
        cache_key = f"{ticker}{cache_suffix}"
        if cache_key in DATA_CACHE:
            print(f"  Cached: {ticker}")
            result[ticker] = DATA_CACHE[cache_key]
        else:
            print(f"  Fetching {ticker}...")
            raw = fetch_stock_prices([ticker], **date_params)
            if ticker not in raw:
                print(f"  No data for '{ticker}'. Check symbol or turn on VPN.")
                continue
            cleaned = clean_stock_dataframe(raw[ticker], ticker)
            DATA_CACHE[cache_key] = cleaned
            result[ticker] = cleaned
            rng = f"{cleaned.index[0].date()} to {cleaned.index[-1].date()}"
            print(f"  {ticker}: {len(cleaned)} trading days ({rng})")
    return result


def get_yfinance_fundamentals(tickers):
    result = {}
    for ticker in tickers:
        cache_key = f"yf_{ticker}"
        if cache_key in FUNDAMENTAL_CACHE:
            result[ticker] = FUNDAMENTAL_CACHE[cache_key]
        else:
            print(f"  Fetching financial statements for {ticker}...")
            data = fetch_all_yfinance_fundamentals(ticker)
            FUNDAMENTAL_CACHE[cache_key] = data
            result[ticker] = data
            has = [k for k in ["income","balance","cashflow","metrics"] if data.get(k)]
            print(f"  {ticker}: loaded {', '.join(has)}")
    return result


def get_rss_news(tickers):
    cache_key = f"rss_{'_'.join(tickers)}"
    if cache_key in FUNDAMENTAL_CACHE:
        return FUNDAMENTAL_CACHE[cache_key]
    print(f"  Fetching RSS news for {tickers}...")
    df = fetch_rss_news(tickers, max_articles=5)
    if df is not None and not df.empty:
        FUNDAMENTAL_CACHE[cache_key] = df
        print(f"  RSS news: {len(df)} articles")
    return df


def get_commodity_data():
    if "commodities" in FUNDAMENTAL_CACHE:
        return FUNDAMENTAL_CACHE["commodities"]
    print("  Fetching commodity prices (Yahoo Finance)...")
    data = fetch_commodity_prices()
    if data:
        FUNDAMENTAL_CACHE["commodities"] = data
        available = [k for k,v in data.items() if v.get("latest_price") is not None]
        print(f"  Commodities: {', '.join(available)}")
    return data or {}


def get_fundamental_data(tickers, intent="full"):
    print(quota.status_line())
    if quota.remaining == 0:
        print("  Alpha Vantage quota exhausted. Resets at 7 AM Vietnam time.")
        return {}
    uncached = [t for t in tickers if t not in FUNDAMENTAL_CACHE]
    if not uncached:
        return {t: FUNDAMENTAL_CACHE[t] for t in tickers if t in FUNDAMENTAL_CACHE}
    features = quota.what_can_afford(n_tickers=len(uncached), intent=intent)
    if not features:
        print(f"  Only {quota.remaining} request(s) left.")
        return {}
    cost = len(uncached) * len(features)
    print(f"  Fetching {features} for {uncached} ({cost} requests)...")
    new_data = fetch_all_fundamental_data(uncached, features=features)
    quota.record(cost)
    for ticker in uncached:
        if ticker in new_data:
            FUNDAMENTAL_CACHE[ticker] = new_data[ticker]
    print(quota.status_line())
    return {t: FUNDAMENTAL_CACHE[t] for t in tickers if t in FUNDAMENTAL_CACHE}


def generate_charts(cleaned_data, fundamental_data=None, intent=""):
    if not cleaned_data:
        return []
    saved = []
    for ticker, df in cleaned_data.items():
        saved.append(plot_price_and_volume(df, ticker))
        saved.append(plot_bollinger_bands(df, ticker))
        path = plot_candlestick(df, ticker)
        if path:
            saved.append(path)
    if len(cleaned_data) > 1:
        saved.append(plot_correlation_heatmap(cleaned_data))
        saved.append(plot_comparative_returns(cleaned_data))
    saved.append(plot_return_distributions(cleaned_data))
    if fundamental_data:
        av_data = fundamental_data.get("av", {})
        if av_data and isinstance(av_data, dict):
            path = plot_fundamental_ratios(av_data)
            if path:
                saved.append(path)
    if intent in ("income","balance","cashflow","full","fundamentals"):
        yf_data = (fundamental_data or {}).get("yfinance", {})
        for ticker, td in yf_data.items():
            if isinstance(td, dict):
                path = plot_financial_dashboard(ticker, td)
                if path:
                    saved.append(path)
    if intent == "compare" and len(cleaned_data) >= 2:
        yf_data = (fundamental_data or {}).get("yfinance", {})
        print("  Generating comparison charts (C1-C4)...")
        saved.extend(generate_comparison_charts(cleaned_data, yf_data))
    return [p for p in saved if p]


def run_analysis(cleaned_data, intent, fundamental_data=None,
                 start_year=None, end_year=None):
    output = []
    if not isinstance(fundamental_data, dict):
        fundamental_data = {}
    fundamental_data.setdefault("yfinance", {})
    fundamental_data.setdefault("av", {})

    if intent == "commodities":
        output.append("\n-- COMMODITY PRICES & MACRO CONTEXT --")
        cd = fundamental_data.get("commodities")
        output.append(generate_commodity_context(cd) if cd else "No commodity data available.")
        return "\n".join(output)

    if intent == "sectors":
        output.append("\n-- SECTOR ROTATION & TRENDING ANALYSIS --")
        print("  Fetching live sector ETF data (~30 seconds)...")
        output.append(generate_sector_analysis(fetch_sector_etf_data(),
                                               fundamental_data.get("commodities")))
        return "\n".join(output)

    if not cleaned_data:
        return "No price data available."

    context = _build_data_context(cleaned_data)
    tickers = list(cleaned_data.keys())

    if intent in ("analyse", "full", "unknown"):
        yf_data  = fundamental_data.get("yfinance", {})
        rss_news = fundamental_data.get("rss_news")
        for ticker in tickers:
            output.append(f"\n-- {ticker} TREND ANALYSIS --")
            output.append(generate_trend_summary(ticker, context))
            output.append(f"\n-- {ticker} MARKET CONTEXT --")
            output.append(generate_market_context(ticker, context))
            output.append(generate_anomaly_commentary(ticker, context))
            yf_ticker = yf_data.get(ticker, {})
            if yf_ticker.get("metrics"):
                output.append(f"\n-- {ticker} KEY METRICS DASHBOARD --")
                output.append(generate_key_metrics_dashboard(ticker, yf_ticker["metrics"]))
            if intent == "full":
                if yf_ticker.get("income"):
                    output.append(f"\n-- {ticker} INCOME STATEMENT --")
                    output.append(generate_yfinance_income_analysis(ticker, yf_ticker["income"]))
                if yf_ticker.get("balance"):
                    output.append(f"\n-- {ticker} BALANCE SHEET --")
                    output.append(generate_yfinance_balance_analysis(ticker, yf_ticker["balance"]))
                if yf_ticker.get("cashflow"):
                    output.append(f"\n-- {ticker} CASH FLOW STATEMENT --")
                    output.append(generate_yfinance_cashflow_analysis(ticker, yf_ticker["cashflow"]))
            av_data = fundamental_data.get("av", {})
            if av_data.get(ticker, {}).get("news", {}).get("articles"):
                output.append(f"\n-- {ticker} NEWS SENTIMENT --")
                output.append(generate_news_sentiment_analysis(ticker, av_data[ticker]["news"]))
            if rss_news is not None:
                output.append(f"\n-- {ticker} MARKET NEWS (RSS) --")
                output.append(generate_rss_news_analysis(ticker, rss_news))
        output.append("\n-- RISK COMMENTARY --")
        output.append(generate_risk_commentary(context))
        output.append("\n-- COMPARATIVE / SECTOR ANALYSIS --")
        output.append(generate_comparative_analysis(context))
        if fundamental_data.get("commodities"):
            output.append("\n-- COMMODITY MACRO CONTEXT --")
            sector_info = {t: (fundamental_data.get("yfinance",{}).get(t,{}).get("metrics",{}) or {}).get("sector","Unknown") for t in tickers}
            output.append(generate_commodity_context(fundamental_data["commodities"],
                                                     tickers=tickers, sector_info=sector_info))

    elif intent == "trend":
        for ticker in tickers:
            output.append(f"\n-- {ticker} TREND --")
            output.append(generate_trend_summary(ticker, context))

    elif intent == "risk":
        output.append("\n-- RISK COMMENTARY --")
        output.append(generate_risk_commentary(context))
        for ticker in tickers:
            output.append(f"\n-- {ticker} ANOMALIES --")
            output.append(generate_anomaly_commentary(ticker, context))

    elif intent == "compare":
        if len(tickers) < 2:
            output.append("Please mention at least 2 stocks to compare.")
        else:
            output.append("\n-- COMPARATIVE ANALYSIS --")
            output.append(generate_comparative_analysis(context))
            output.append("\n-- RISK COMPARISON --")
            output.append(generate_risk_commentary(context))

    elif intent == "fundamentals":
        yf_data = fundamental_data.get("yfinance", {})
        for ticker in tickers:
            td = yf_data.get(ticker, {})
            if td.get("metrics"):
                output.append(f"\n-- {ticker} KEY METRICS --")
                output.append(generate_key_metrics_dashboard(ticker, td["metrics"]))

    elif intent == "news":
        rss_news = fundamental_data.get("rss_news")
        av_data  = fundamental_data.get("av", {})
        for ticker in tickers:
            if av_data.get(ticker, {}).get("news", {}).get("articles"):
                output.append(f"\n-- {ticker} NEWS SENTIMENT --")
                output.append(generate_news_sentiment_analysis(ticker, av_data[ticker]["news"]))
            if rss_news is not None:
                output.append(f"\n-- {ticker} RSS NEWS --")
                output.append(generate_rss_news_analysis(ticker, rss_news))

    elif intent == "income":
        yf_data = fundamental_data.get("yfinance", {})
        for ticker in tickers:
            income = yf_data.get(ticker, {}).get("income", {})
            if not income:
                output.append(f"No income data for {ticker}.")
                continue
            if start_year or end_year:
                annual   = income.get("annual", [])
                filtered = [r for r in annual if
                            (not start_year or int(r["period"][:4]) >= start_year) and
                            (not end_year   or int(r["period"][:4]) <= end_year)]
                if filtered:
                    income = {**income, "annual": filtered}
            label = f"{start_year}-{end_year}" if (start_year and end_year) else ticker
            output.append(f"\n-- {ticker} INCOME STATEMENT ({label}) --")
            output.append(generate_yfinance_income_analysis(ticker, income))

    elif intent == "balance":
        yf_data = fundamental_data.get("yfinance", {})
        for ticker in tickers:
            balance = yf_data.get(ticker, {}).get("balance", {})
            if not balance:
                output.append(f"No balance sheet data for {ticker}.")
                continue
            output.append(f"\n-- {ticker} BALANCE SHEET --")
            output.append(generate_yfinance_balance_analysis(ticker, balance))

    elif intent == "cashflow":
        yf_data = fundamental_data.get("yfinance", {})
        for ticker in tickers:
            cashflow = yf_data.get(ticker, {}).get("cashflow", {})
            if not cashflow:
                output.append(f"No cash flow data for {ticker}.")
                continue
            output.append(f"\n-- {ticker} CASH FLOW STATEMENT --")
            output.append(generate_yfinance_cashflow_analysis(ticker, cashflow))

    return "\n".join(output)


def save_analysis(tickers, analysis_text):
    import textwrap
    label    = "_".join(tickers) if tickers else "no_ticker"
    path     = REPORTS_ANALYSIS_DIR / f"analysis_{label}.txt"
    lines    = []
    for line in analysis_text.split("\n"):
        if len(line) <= 100 or line.startswith(("--","==","**")):
            lines.append(line)
        else:
            indent = len(line) - len(line.lstrip())
            lines.append(textwrap.fill(line.strip(), width=100,
                                       initial_indent=" "*indent,
                                       subsequent_indent=" "*indent))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def print_banner():
    print("\n" + "=" * 60)
    print("  FINAGENT - AI-Powered Financial Agent")
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
    print("\n  Type 'help' to see this again. Type 'quit' to exit.")
    print("=" * 60 + "\n")


def main():
    print_banner()

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in .env file.")
        sys.exit(1)

    print(quota.status_line() + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! Results saved to reports/")
                break
            if user_input.lower() in ("help", "?"):
                print_banner()
                continue
            if user_input.lower() in ("quota", "quota status"):
                print(quota.status_line() + "\n")
                continue

            print("\nAgent: Let me look into that...\n")

            # Step 1: Parse intent
            intent_data = parse_user_intent(user_input)
            tickers     = intent_data.get("tickers", [])
            intent      = intent_data.get("intent", "unknown")
            start_year  = intent_data.get("start_year")
            end_year    = intent_data.get("end_year")

            year_label = (f" | Range: {start_year or 'start'} to {end_year or 'now'}"
                          if (start_year or end_year) else "")
            print(f"  Intent: {intent}"
                  + (f" | Tickers: {tickers}" if tickers else "")
                  + year_label + "\n")

            # Step 2: No-ticker intents
            if intent in ("commodities", "sectors") and not tickers:
                if intent == "commodities":
                    fd = {"commodities": get_commodity_data()}
                else:
                    fd = {"commodities": FUNDAMENTAL_CACHE.get("commodities")}
                analysis = run_analysis({}, intent, fd)
                print("=" * 60)
                print(analysis)
                print("=" * 60 + "\n")
                save_analysis([], analysis)
                continue

            # Step 3: Validate tickers
            if not tickers:
                print("Agent: Could not find any stock names. Try: 'Analyse Apple'\n")
                continue

            # Step 4: Price data
            cleaned_data = get_price_data(tickers, start_year=start_year, end_year=end_year)
            if not cleaned_data:
                print("Agent: Could not retrieve price data. Turn on VPN and try again.\n")
                continue

            # Step 5: All data sources
            fundamental_data = {"yfinance": {}, "av": {}, "rss_news": None}

            if intent in ("full","fundamentals","income","balance",
                          "cashflow","analyse","unknown","compare"):
                print("  Fetching financial statements (Yahoo Finance, free)...")
                fundamental_data["yfinance"] = get_yfinance_fundamentals(tickers)

            if intent in ("full","news","analyse","unknown"):
                fundamental_data["rss_news"] = get_rss_news(tickers)

            if intent in ("full","news","fundamentals") and quota.remaining >= len(tickers):
                fundamental_data["av"] = get_fundamental_data(tickers, intent=intent)

            if intent == "full":
                fundamental_data["commodities"] = get_commodity_data()

            # Step 6: Charts
            print("  Generating charts...")
            chart_paths = generate_charts(cleaned_data, fundamental_data, intent=intent)
            print(f"  {len(chart_paths)} chart(s) saved to reports/charts/\n")

            # Step 7: Analysis
            print("  Running AI analysis...\n")
            analysis = run_analysis(cleaned_data, intent, fundamental_data,
                                    start_year=start_year, end_year=end_year)

            # Step 8: Output
            print("=" * 60)
            print(analysis)
            print("=" * 60 + "\n")

            saved_path = save_analysis(tickers, analysis)
            print(f"  Saved to reports/analysis/{saved_path.name}\n")

            # Step 9: PDF export
            print("  Export this analysis as a PDF report? (yes/no): ", end="")
            try:
                export_choice = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                export_choice = "no"

            if export_choice in ("yes", "y"):
                print("  Generating PDF report (15-30 seconds)...")
                pdf_path = export_report(
                    tickers=tickers,
                    analysis_text=analysis,
                    chart_paths=chart_paths,
                )
                if pdf_path:
                    ext = "PDF" if pdf_path.endswith(".pdf") else "Word document"
                    print(f"  {ext} saved: {Path(pdf_path).name}\n")
                else:
                    print("  Export failed. Check python-docx is installed.\n")
            else:
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nSomething went wrong: {e}")
            print("Try again with a different query.\n")


if __name__ == "__main__":
    main()