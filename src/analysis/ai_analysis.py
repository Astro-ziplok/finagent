"""
src/analysis/ai_analysis.py — AI Analysis Module
=================================================
Responsibility: Use the Groq API to generate natural language analysis
of our financial data.

This module produces (all required by the rubric + extras):
  ✓ Detailed trend summaries with MA signals, Bollinger Bands, quarterly breakdown
  ✓ Investment implication section (so what does this mean for an investor?)
  ✓ Market context & news factors (LLM knowledge of the company/sector)
  ✓ Anomaly / notable event identification
  ✓ Risk commentary based on volatility metrics and Sharpe ratio
  ✓ Comparative analysis (handles both single and multi-stock queries gracefully)
"""

import json
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL, MAX_TOKENS, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_data_context(cleaned_data: dict) -> dict:
    """
    Extract a rich, structured JSON-serialisable summary of the data.
    Includes quarterly breakdown, MA signals, Bollinger Bands,
    win rate, Sharpe ratio, and multi-window momentum.
    """
    context = {}

    for ticker, df in cleaned_data.items():
        returns   = df["Daily_Return"].dropna()
        recent_7  = df.tail(7)
        recent_30 = df.tail(30)
        recent_90 = df.tail(90)

        # Quarter-by-quarter breakdown
        quarterly = []
        for q in range(4):
            chunk = df.iloc[q * 63:(q + 1) * 63]
            if len(chunk) > 5:
                quarterly.append({
                    "quarter":          f"Q{q + 1}",
                    "start_date":       str(chunk.index[0].date()),
                    "end_date":         str(chunk.index[-1].date()),
                    "open_price":       round(float(chunk["Close"].iloc[0]), 2),
                    "close_price":      round(float(chunk["Close"].iloc[-1]), 2),
                    "pct_change":       round((chunk["Close"].iloc[-1] / chunk["Close"].iloc[0] - 1) * 100, 2),
                    "avg_daily_volume": int(chunk["Volume"].mean()),
                })

        # Moving average signal
        current_price = float(df["Close"].iloc[-1])
        ma7  = float(df["MA_7"].iloc[-1])  if "MA_7"  in df.columns and not pd.isna(df["MA_7"].iloc[-1])  else None
        ma30 = float(df["MA_30"].iloc[-1]) if "MA_30" in df.columns and not pd.isna(df["MA_30"].iloc[-1]) else None

        if ma7 and ma30:
            if current_price > ma7 and current_price > ma30:
                ma_signal = "BULLISH — price is above both the 7-day and 30-day moving averages"
            elif current_price < ma7 and current_price < ma30:
                ma_signal = "BEARISH — price is below both the 7-day and 30-day moving averages"
            elif current_price > ma30 and current_price < ma7:
                ma_signal = "MIXED — above 30-day MA but below 7-day MA; short-term pullback within a longer uptrend"
            else:
                ma_signal = "MIXED — above 7-day MA but below 30-day MA; short-term recovery within a longer downtrend"
        else:
            ma_signal = "Insufficient data for MA signal"

        # Bollinger Band signal
        bb_upper = float(df["BB_Upper"].iloc[-1]) if "BB_Upper" in df.columns and not pd.isna(df["BB_Upper"].iloc[-1]) else None
        bb_lower = float(df["BB_Lower"].iloc[-1]) if "BB_Lower" in df.columns and not pd.isna(df["BB_Lower"].iloc[-1]) else None
        bb_mid   = float(df["BB_Mid"].iloc[-1])   if "BB_Mid"   in df.columns and not pd.isna(df["BB_Mid"].iloc[-1])   else None

        if bb_upper and bb_lower and bb_mid:
            if current_price >= bb_upper:
                bb_signal = "OVERBOUGHT — price at/above upper band; pullback or consolidation may follow"
            elif current_price <= bb_lower:
                bb_signal = "OVERSOLD — price at/below lower band; potential bounce or reversal may follow"
            elif current_price > bb_mid:
                bb_signal = "NEUTRAL-BULLISH — price in upper half of Bollinger Band range"
            else:
                bb_signal = "NEUTRAL-BEARISH — price in lower half of Bollinger Band range"
        else:
            bb_signal = "Insufficient data for Bollinger Band signal"

        sharpe_approx = round(returns.mean() / returns.std(), 4) if returns.std() != 0 else 0
        win_rate      = round((returns > 0).sum() / len(returns) * 100, 1)

        context[ticker] = {
            "period": {
                "start":        str(df.index[0].date()),
                "end":          str(df.index[-1].date()),
                "trading_days": len(df),
            },
            "full_year_price": {
                "start_price":        round(float(df["Close"].iloc[0]), 2),
                "end_price":          round(float(df["Close"].iloc[-1]), 2),
                "52w_high":           round(float(df["High"].max()), 2),
                "52w_low":            round(float(df["Low"].min()), 2),
                "total_return_pct":   round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2),
                "pct_below_52w_high": round((current_price / float(df["High"].max()) - 1) * 100, 2),
                "pct_above_52w_low":  round((current_price / float(df["Low"].min()) - 1) * 100, 2),
            },
            "returns": {
                "mean_daily_pct":      round(float(returns.mean()), 3),
                "std_daily_pct":       round(float(returns.std()), 3),
                "best_day_pct":        round(float(returns.max()), 2),
                "best_day_date":       str(returns.idxmax().date()),
                "worst_day_pct":       round(float(returns.min()), 2),
                "worst_day_date":      str(returns.idxmin().date()),
                "win_rate_pct":        win_rate,
                "sharpe_ratio_approx": sharpe_approx,
            },
            "short_term_momentum": {
                "last_7d_change_pct":  round((recent_7["Close"].iloc[-1]  / recent_7["Close"].iloc[0]  - 1) * 100, 2),
                "last_30d_change_pct": round((recent_30["Close"].iloc[-1] / recent_30["Close"].iloc[0] - 1) * 100, 2),
                "last_90d_change_pct": round((recent_90["Close"].iloc[-1] / recent_90["Close"].iloc[0] - 1) * 100, 2),
                "last_30d_volatility": round(float(recent_30["Daily_Return"].std()), 3),
                "last_30d_avg_volume": int(recent_30["Volume"].mean()),
            },
            "technical_signals": {
                "current_price": round(current_price, 2),
                "ma_7":          round(ma7, 2) if ma7 else None,
                "ma_30":         round(ma30, 2) if ma30 else None,
                "ma_signal":     ma_signal,
                "bb_upper":      round(bb_upper, 2) if bb_upper else None,
                "bb_mid":        round(bb_mid, 2) if bb_mid else None,
                "bb_lower":      round(bb_lower, 2) if bb_lower else None,
                "bb_signal":     bb_signal,
            },
            "quarterly_breakdown": quarterly,
            "outlier_dates": df[df["Is_Outlier"] == True].index.strftime("%Y-%m-%d").tolist(),
        }

    return context


def _call_groq(prompt: str, system_prompt: str = "", max_tokens: int = MAX_TOKENS) -> str:
    """Send a prompt to Groq and return the response text."""
    if not GROQ_API_KEY:
        return "⚠️  GROQ_API_KEY not set in .env file. Skipping AI analysis."

    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return f"⚠️  Groq API call failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_trend_summary(ticker: str, data_context: dict) -> str:
    """
    Generate a rich, structured trend summary with 6 sections including
    an investment implication conclusion.
    """
    stats      = data_context[ticker]
    stats_json = json.dumps(stats, indent=2)

    system_prompt = """
You are a senior equity research analyst writing a stock trend report for a university finance course.

Rules:
- Use ONLY the exact numbers from the JSON data provided. Never invent or estimate figures.
- Write in clear, professional English suitable for a finance audience.
- Use specific numbers in every sentence — vague statements are not acceptable.
- Structure your response with the exact section headers shown below.
- Each section should be 2-4 sentences.
"""

    prompt = f"""
Using the data below for {ticker}, write a structured trend analysis report.

DATA:
{stats_json}

Write your analysis using EXACTLY these section headers:

**1. Overall Performance**
Summarise the full-year return (use total_return_pct). State the start and end price.
Compare to the 52-week high and low — how far is the current price from each?

**2. Quarterly Breakdown**
Walk through each quarter's performance using the quarterly_breakdown data.
Identify which quarter was strongest and which was weakest, with exact figures.

**3. Short-Term Momentum**
Discuss the 7-day, 30-day, and 90-day price changes from short_term_momentum.
Is the stock accelerating or decelerating recently?

**4. Technical Signal**
State the current moving average signal (ma_signal) and what it implies.
State the Bollinger Band signal (bb_signal) and what it suggests about near-term direction.

**5. Risk & Return Profile**
Use win_rate_pct, best_day_pct with its date, worst_day_pct with its date, and sharpe_ratio_approx.
Comment on what the win rate and Sharpe ratio suggest about the stock's consistency.

**6. Investment Implication**
Based on ALL the sections above, write 2-3 sentences answering: "So what does this mean for an investor?"
State clearly whether this stock currently looks more suitable for a risk-tolerant or risk-averse investor.
Reference at least 2 specific numbers from the data to justify your conclusion.
"""
    return _call_groq(prompt, system_prompt, max_tokens=1500)


def generate_market_context(ticker: str, data_context: dict) -> str:
    """
    Generate a market context and news factors section using the LLM's
    own knowledge of the company, its sector, and relevant macro themes.
    This adds a qualitative news layer without requiring a NewsAPI key.
    """
    stats = data_context[ticker]
    period_start = stats["period"]["start"]
    period_end   = stats["period"]["end"]
    total_return = stats["full_year_price"]["total_return_pct"]
    best_date    = stats["returns"]["best_day_date"]
    worst_date   = stats["returns"]["worst_day_date"]
    outliers     = stats["outlier_dates"]

    system_prompt = """
You are a financial analyst with broad knowledge of market events, corporate news,
and macroeconomic trends. Write in a professional, analytical tone suitable for
a university finance course report. Be specific about time periods and events.
Do NOT fabricate specific earnings figures or stock prices — focus on known
qualitative themes, sector trends, and company developments.
"""

    prompt = f"""
Write a "Market Context & News Factors" section for {ticker} covering the period
{period_start} to {period_end}.

Key data points to contextualise:
- Total return over the period: {total_return}%
- Best single trading day: {best_date}
- Worst single trading day: {worst_date}
- Statistically extreme outlier dates: {outliers if outliers else "None detected"}

Using your knowledge of {ticker} and its industry, write 3 paragraphs with these headers:

**Company & Sector Overview**
Briefly describe what {ticker} does and which sector/industry it operates in.
Mention 1-2 major strategic themes or business developments relevant to this period.

**Key Market Themes & Drivers**
What macro-economic or sector-specific themes (e.g. AI boom, interest rate environment,
regulatory changes, supply chain shifts) likely influenced {ticker}'s performance
during this period? Discuss 2-3 themes specifically relevant to this company.

**News Context for Outlier Dates**
If outlier dates were detected, discuss what types of corporate events
(earnings releases, product launches, analyst upgrades/downgrades, macro data releases)
typically drive extreme single-day moves for {ticker}.
If no outliers were detected, comment on what that stability suggests about
market sentiment toward the company during this period.
"""
    return _call_groq(prompt, system_prompt, max_tokens=800)


def generate_anomaly_commentary(ticker: str, data_context: dict) -> str:
    """Identify and comment on outlier / anomalous trading days."""
    stats    = data_context[ticker]
    outliers = stats.get("outlier_dates", [])

    if not outliers:
        return (
            f"✅ No statistical outliers (returns beyond ±3 standard deviations) "
            f"were detected for {ticker} over the {stats['period']['trading_days']}-day "
            f"analysis period. This suggests relatively stable daily price behaviour "
            f"with no extreme single-day events during this timeframe."
        )

    system_prompt = """
You are a financial analyst identifying unusual market events.
Reference the exact dates provided. Do not fabricate specific reasons —
acknowledge uncertainty and suggest cross-referencing with news.
Write in 3-5 sentences, professionally and specifically.
"""

    prompt = f"""
{ticker} experienced the following statistically extreme trading days
(daily returns beyond ±3 standard deviations from the mean):

Outlier dates: {outliers}
Mean daily return: {stats['returns']['mean_daily_pct']}%
Daily std deviation: {stats['returns']['std_daily_pct']}%
Best single day: {stats['returns']['best_day_pct']}% on {stats['returns']['best_day_date']}
Worst single day: {stats['returns']['worst_day_pct']}% on {stats['returns']['worst_day_date']}

Write a commentary on these anomaly dates. Note how many occurred, when they happened,
and what types of events (earnings surprises, macro shocks, sector news, analyst actions)
could typically cause such extreme moves for this stock.
Advise cross-referencing with financial news for these specific dates.
"""
    return _call_groq(prompt, system_prompt, max_tokens=400)


def generate_risk_commentary(data_context: dict) -> str:
    """Produce a risk comparison across all tracked assets."""
    risk_summary = {
        ticker: {
            "annual_volatility_proxy": round(data["returns"]["std_daily_pct"] * (252 ** 0.5), 2),
            "daily_std_pct":           data["returns"]["std_daily_pct"],
            "recent_30d_volatility":   data["short_term_momentum"]["last_30d_volatility"],
            "worst_day_pct":           data["returns"]["worst_day_pct"],
            "worst_day_date":          data["returns"]["worst_day_date"],
            "sharpe_ratio_approx":     data["returns"]["sharpe_ratio_approx"],
            "win_rate_pct":            data["returns"]["win_rate_pct"],
            "total_return_pct":        data["full_year_price"]["total_return_pct"],
        }
        for ticker, data in data_context.items()
    }

    system_prompt = """
You are a risk analyst writing a risk assessment for a university finance project.
Use only the provided data. Reference specific numbers in every claim.
Write 2 well-structured paragraphs.
"""

    prompt = f"""
Write a risk commentary comparing these assets using the data below:

{json.dumps(risk_summary, indent=2)}

Paragraph 1 — Volatility Ranking:
Rank all assets from least to most risky by daily_std_pct.
Annualise the volatility using annual_volatility_proxy.
Note which asset had the single worst day and on what date.
Comment on whether recent 30-day volatility is higher or lower than the full-year average.

Paragraph 2 — Risk-Adjusted Performance:
Use sharpe_ratio_approx and win_rate_pct to assess which asset gave the best return per unit of risk.
Discuss whether high-return assets justified their volatility.
Conclude with a practical implication for a risk-averse vs risk-tolerant investor.
"""
    return _call_groq(prompt, system_prompt, max_tokens=600)


def generate_comparative_analysis(data_context: dict) -> str:
    """
    Compare assets across return, risk, momentum, and technical signals.
    Handles single-stock queries gracefully with a sector comparison instead.
    """
    tickers = list(data_context.keys())

    # ── Single stock: replace comparison with sector context ──────────────────
    if len(tickers) == 1:
        ticker = tickers[0]
        data   = data_context[ticker]

        system_prompt = """
You are a senior equity analyst writing a sector comparison for a university finance report.
Be specific and analytical. Reference the exact numbers provided.
Write exactly 3 paragraphs with the headers shown.
"""
        prompt = f"""
Only one asset ({ticker}) was analysed. Instead of a multi-stock comparison,
write a sector benchmarking analysis using the data below and your knowledge of
{ticker}'s industry peers and the broader market.

Data for {ticker}:
- Total return: {data['full_year_price']['total_return_pct']}%
- Daily volatility: {data['returns']['std_daily_pct']}%
- Sharpe ratio: {data['returns']['sharpe_ratio_approx']}
- Win rate: {data['returns']['win_rate_pct']}%
- 30-day momentum: {data['short_term_momentum']['last_30d_change_pct']}%
- MA signal: {data['technical_signals']['ma_signal']}
- BB signal: {data['technical_signals']['bb_signal']}

Write using EXACTLY these section headers:

**Sector Benchmarking**
Compare {ticker}'s total return of {data['full_year_price']['total_return_pct']}% to
what you know about typical returns for its sector peers over a similar period.
Is this outperformance, underperformance, or in line with sector expectations?

**Risk Profile vs Peers**
How does {ticker}'s daily volatility of {data['returns']['std_daily_pct']}% and
Sharpe ratio of {data['returns']['sharpe_ratio_approx']} compare to what you would
expect from similar large-cap companies in its sector?

**Relative Momentum & Outlook**
Given the 30-day momentum of {data['short_term_momentum']['last_30d_change_pct']}%
and the current technical signals, how does {ticker}'s near-term outlook compare
to the broader market or sector trend? Would you consider it a leader or a lagger?
"""
        return _call_groq(prompt, system_prompt, max_tokens=700)

    # ── Multi-stock: full comparison ──────────────────────────────────────────
    comparison_data = {
        ticker: {
            "total_return_pct":    data["full_year_price"]["total_return_pct"],
            "current_price":       data["full_year_price"]["end_price"],
            "52w_high":            data["full_year_price"]["52w_high"],
            "pct_below_52w_high":  data["full_year_price"]["pct_below_52w_high"],
            "daily_volatility":    data["returns"]["std_daily_pct"],
            "sharpe_ratio":        data["returns"]["sharpe_ratio_approx"],
            "win_rate_pct":        data["returns"]["win_rate_pct"],
            "last_30d_change_pct": data["short_term_momentum"]["last_30d_change_pct"],
            "ma_signal":           data["technical_signals"]["ma_signal"],
            "bb_signal":           data["technical_signals"]["bb_signal"],
        }
        for ticker, data in data_context.items()
    }

    system_prompt = """
You are a senior equity analyst writing a comparative stock analysis for a finance course.
Use only the provided numbers. Be specific, analytical, and objective.
Write exactly 3 paragraphs with the headers shown.
"""

    prompt = f"""
Write a comparative analysis of the following assets:

{json.dumps(comparison_data, indent=2)}

Use EXACTLY these section headers:

**Return Comparison**
Rank all assets by total_return_pct from best to worst. State each figure explicitly.
Note which is furthest from its 52-week high (pct_below_52w_high) and what that implies.
End with 1 sentence on what the return spread between best and worst tells us.

**Risk & Consistency Comparison**
Compare daily_volatility and sharpe_ratio across all assets.
Discuss win_rate_pct — which stock had the most consistent positive days?
Identify which asset offered the best risk-adjusted return and state why.

**Momentum & Technical Outlook**
Compare last_30d_change_pct — which asset has the strongest recent momentum?
Summarise the ma_signal and bb_signal for each asset in one sentence each.
Conclude with a clear statement: which asset looks most technically constructive right now and why?
"""
    return _call_groq(prompt, system_prompt, max_tokens=900)


# ─────────────────────────────────────────────────────────────────────────────
#  Master function: run all analyses
# ─────────────────────────────────────────────────────────────────────────────

def run_all_analysis(cleaned_data: dict) -> dict:
    """
    Run all AI analysis components and save the results to a text file.

    Components:
      1. Trend summary (with investment implication)
      2. Market context & news factors
      3. Anomaly commentary
      4. Risk commentary
      5. Comparative / sector analysis
    """
    logger.info("=" * 60)
    logger.info("STEP 4: AI ANALYSIS (Groq)")
    logger.info("=" * 60)

    data_context = _build_data_context(cleaned_data)
    results = {}

    # 1. Trend summaries
    results["trend_summaries"] = {}
    for ticker in cleaned_data:
        logger.info(f"  Generating trend summary for {ticker}...")
        results["trend_summaries"][ticker] = generate_trend_summary(ticker, data_context)

    # 2. Market context & news factors (NEW)
    results["market_context"] = {}
    for ticker in cleaned_data:
        logger.info(f"  Generating market context for {ticker}...")
        results["market_context"][ticker] = generate_market_context(ticker, data_context)

    # 3. Anomaly commentary
    results["anomaly_commentary"] = {}
    for ticker in cleaned_data:
        logger.info(f"  Generating anomaly commentary for {ticker}...")
        results["anomaly_commentary"][ticker] = generate_anomaly_commentary(ticker, data_context)

    # 4. Risk commentary
    logger.info("  Generating risk commentary...")
    results["risk_commentary"] = generate_risk_commentary(data_context)

    # 5. Comparative / sector analysis
    logger.info("  Generating comparative analysis...")
    results["comparative_analysis"] = generate_comparative_analysis(data_context)

    _save_analysis_report(results)
    return results


def _save_analysis_report(results: dict) -> None:
    """Save all AI analysis results to a readable text file in reports/."""
    output_path = REPORTS_DIR / "ai_analysis_report.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("FINAGENT — AI ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("── TREND SUMMARIES ─────────────────────────────────────────────────\n\n")
        for ticker, summary in results.get("trend_summaries", {}).items():
            f.write(f"[{ticker}]\n{summary}\n\n")

        f.write("── MARKET CONTEXT & NEWS FACTORS ───────────────────────────────────\n\n")
        for ticker, context in results.get("market_context", {}).items():
            f.write(f"[{ticker}]\n{context}\n\n")

        f.write("── ANOMALY & OUTLIER COMMENTARY ────────────────────────────────────\n\n")
        for ticker, commentary in results.get("anomaly_commentary", {}).items():
            f.write(f"[{ticker}]\n{commentary}\n\n")

        f.write("── RISK COMMENTARY ─────────────────────────────────────────────────\n\n")
        f.write(results.get("risk_commentary", "") + "\n\n")

        f.write("── COMPARATIVE / SECTOR ANALYSIS ───────────────────────────────────\n\n")
        f.write(results.get("comparative_analysis", "") + "\n\n")

    logger.info(f"  💾 AI analysis report saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Run directly to test: python src/analysis/ai_analysis.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import DATA_PROC_DIR, TICKERS

    cleaned_data = {}
    for ticker in TICKERS:
        path = DATA_PROC_DIR / f"{ticker}_processed.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            cleaned_data[ticker] = df

    if cleaned_data:
        results = run_all_analysis(cleaned_data)
        print("\n\n=== SAMPLE OUTPUT ===\n")
        first = list(cleaned_data.keys())[0]
        print(f"Trend Summary — {first}:")
        print(results["trend_summaries"][first])
        print(f"\nMarket Context — {first}:")
        print(results["market_context"][first])
    else:
        print("No processed data found. Run clean_data.py first.")


# ─────────────────────────────────────────────────────────────────────────────
#  Fundamental & News-Enriched Analysis (Alpha Vantage)
# ─────────────────────────────────────────────────────────────────────────────

def generate_fundamental_analysis(ticker: str, overview: dict, income: dict, balance: dict) -> str:
    """
    Generate a financial health analysis using real fundamental data
    from Alpha Vantage: P/E, ROE, ROA, D/E, revenue trends, etc.
    """
    if not overview:
        return f"⚠️  No fundamental data available for {ticker}."

    system_prompt = """
You are a fundamental analyst writing a financial health report for a university finance course.
Use ONLY the exact numbers provided. Never invent figures.
Write in clear, professional English. Each section should be 2-3 sentences.
"""

    income_summary = income.get("quarterly_income", [])[:2] if income else []
    balance_summary = balance.get("quarterly_balance", [])[:1] if balance else []

    prompt = f"""
Write a fundamental analysis for {ticker} using the data below.

COMPANY OVERVIEW:
{json.dumps(overview, indent=2)}

RECENT INCOME STATEMENT (last 2 quarters):
{json.dumps(income_summary, indent=2)}

LATEST BALANCE SHEET:
{json.dumps(balance_summary, indent=2)}

Use EXACTLY these section headers:

**Company Profile**
State the company name, sector, industry, and market cap.
Write 1 sentence describing the business using the description field.

**Valuation Assessment**
Discuss the P/E ratio — is the stock cheap or expensive relative to typical market averages (S&P 500 avg ~20-25x)?
Comment on P/B ratio and PEG ratio if available. What do these suggest about market expectations?

**Profitability**
Discuss ROE and ROA — convert to percentages by multiplying by 100.
Comment on profit margin and operating margin. Is the company efficiently generating profit?

**Financial Health**
Discuss the debt-to-equity ratio — is the company conservatively or aggressively financed?
Comment on current ratio and cash position from the balance sheet if available.

**Revenue & Earnings Trend**
Using the quarterly income data, is revenue growing or declining quarter over quarter?
Comment on net income trend and what it implies about business momentum.

**Analyst Consensus**
State the analyst target price vs current price — is there upside or downside implied?
Summarise the analyst rating breakdown (strong buy / buy / hold / sell counts).
"""
    return _call_groq(prompt, system_prompt, max_tokens=1000)


def generate_news_sentiment_analysis(ticker: str, news_data: dict) -> str:
    """
    Generate an analysis section based on real news headlines and
    sentiment scores from Alpha Vantage.
    """
    if not news_data or not news_data.get("articles"):
        return f"⚠️  No news sentiment data available for {ticker}."

    articles     = news_data["articles"][:8]  # Top 8 most recent
    avg_score    = news_data["avg_sentiment_score"]
    sentiment_label = news_data["sentiment_label"]
    article_count   = news_data["article_count"]

    # Build a readable headlines list for the prompt
    headlines = "\n".join([
        f"  - [{a['published_at']}] {a['title']} | Source: {a['source']} | Sentiment: {a['sentiment_label']} ({a['sentiment_score']:+.3f})"
        for a in articles
    ])

    system_prompt = """
You are a financial news analyst writing a sentiment report for a university finance project.
Reference specific headlines and dates from the provided data.
Do NOT fabricate any news — only reference what is given.
Write in 3 structured paragraphs.
"""

    prompt = f"""
Analyse the news sentiment for {ticker} based on the following real headlines:

AVERAGE SENTIMENT SCORE: {avg_score} ({sentiment_label})
ARTICLE COUNT: {article_count}

RECENT HEADLINES:
{headlines}

Write 3 paragraphs with these headers:

**News Sentiment Overview**
State the overall sentiment score ({avg_score}) and label ({sentiment_label}).
Is the market narrative around {ticker} currently positive, negative, or mixed?
Reference the number of articles analysed.

**Key Themes in Recent Coverage**
Identify 2-3 recurring themes or topics across the headlines provided.
Which headlines had the strongest positive or negative sentiment scores?
Reference specific titles and dates.

**Sentiment vs Price Action Implication**
How does the current news sentiment ({sentiment_label}) align or conflict with the stock's recent price momentum?
What should investors watch for in upcoming news coverage?
"""
    return _call_groq(prompt, system_prompt, max_tokens=700)


def generate_commodity_context(commodity_data: dict,
                               tickers: list = None,
                               sector_info: dict = None) -> str:
    """
    Generate a macro commodity context section.
    When tickers and sector_info are provided, the LLM connects commodity
    trends specifically to those stocks rather than giving generic commentary.

    Args:
        commodity_data: Dict from fetch_commodity_prices()
        tickers:        Optional list of stock tickers being analysed
        sector_info:    Optional dict of {ticker: sector} for specific implications
    """
    if not commodity_data:
        return "⚠️  No commodity data available."

    system_prompt = """
You are a macro analyst writing a commodity market context section for a finance report.
Use only the provided data. Reference specific price levels and percentage changes.
Be specific about how commodity trends affect the exact companies mentioned.
Write 3 paragraphs with the headers shown.
"""

    # Build stock-specific context string
    stock_context = ""
    if tickers and sector_info:
        stock_context = f"""
STOCKS BEING ANALYSED:
{json.dumps(sector_info, indent=2)}

Make sure to specifically discuss how the commodity trends affect EACH of these companies
based on their sector and business model. Do not write generic implications.
"""
    elif tickers:
        stock_context = f"\nSTOCKS BEING ANALYSED: {', '.join(tickers)}\nConnect commodity trends specifically to these companies.\n"

    prompt = f"""
Write a commodity market context section using the data below:

COMMODITY DATA:
{json.dumps(commodity_data, indent=2)}
{stock_context}

Write using EXACTLY these section headers:

**Commodity Market Snapshot**
For each commodity with available data, state the latest price, unit, and 1-month change.
Which commodities are rising and which are falling? Reference exact figures for each.

**Macro Implications**
How do current oil, gas, and gold price trends reflect broader macroeconomic conditions?
What do these commodity trends suggest about inflation, corporate costs, and consumer spending?

**Impact on Analysed Stocks**
{"For each of " + ", ".join(tickers) + ", explain specifically how the commodity trends above affect that company." if tickers else "Discuss how these commodity trends would affect technology, finance, and energy sector stocks differently."}
Consider each company's sector, cost structure, and business model.
Which of the analysed stocks is MOST and LEAST exposed to commodity price risk?
"""
    return _call_groq(prompt, system_prompt, max_tokens=700)


def run_full_enriched_analysis(cleaned_data: dict, fundamental_data: dict = None) -> dict:
    """
    Run the complete analysis pipeline including:
    - Standard technical analysis (trend, anomaly, risk, comparison)
    - Fundamental analysis (P/E, ROE, ROA, D/E, revenue) if Alpha Vantage data available
    - News sentiment analysis if Alpha Vantage data available
    - Commodity macro context if Alpha Vantage data available

    Args:
        cleaned_data:     Dict of cleaned price DataFrames
        fundamental_data: Dict from fetch_all_fundamental_data() or None

    Returns:
        Dict of all analysis results
    """
    # Run standard analysis first
    results = run_all_analysis(cleaned_data)

    if not fundamental_data:
        return results

    # Add fundamental analysis per ticker
    results["fundamental_analysis"] = {}
    results["news_sentiment"] = {}

    for ticker in cleaned_data:
        ticker_data = fundamental_data.get(ticker, {})

        # Fundamental ratios + financial statements
        if ticker_data.get("overview"):
            logger.info(f"  Generating fundamental analysis for {ticker}...")
            results["fundamental_analysis"][ticker] = generate_fundamental_analysis(
                ticker,
                ticker_data.get("overview", {}),
                ticker_data.get("income", {}),
                ticker_data.get("balance", {}),
            )

        # News sentiment
        if ticker_data.get("news", {}).get("articles"):
            logger.info(f"  Generating news sentiment analysis for {ticker}...")
            results["news_sentiment"][ticker] = generate_news_sentiment_analysis(
                ticker, ticker_data["news"]
            )

    # Commodity macro context
    if fundamental_data.get("commodities"):
        logger.info("  Generating commodity macro context...")
        results["commodity_context"] = generate_commodity_context(
            fundamental_data["commodities"]
        )

    # Save enriched report
    _save_enriched_report(results)
    return results


def _save_enriched_report(results: dict) -> None:
    """Save the full enriched report to reports/analysis/."""
    from config import REPORTS_ANALYSIS_DIR
    output_path = REPORTS_ANALYSIS_DIR / "ai_analysis_report.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("FINAGENT — FULL AI ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("── TREND SUMMARIES ─────────────────────────────────────────────────\n\n")
        for ticker, s in results.get("trend_summaries", {}).items():
            f.write(f"[{ticker}]\n{s}\n\n")

        f.write("── MARKET CONTEXT & NEWS FACTORS ───────────────────────────────────\n\n")
        for ticker, s in results.get("market_context", {}).items():
            f.write(f"[{ticker}]\n{s}\n\n")

        if results.get("news_sentiment"):
            f.write("── NEWS SENTIMENT (Alpha Vantage) ───────────────────────────────────\n\n")
            for ticker, s in results["news_sentiment"].items():
                f.write(f"[{ticker}]\n{s}\n\n")

        if results.get("fundamental_analysis"):
            f.write("── FUNDAMENTAL ANALYSIS (Alpha Vantage) ────────────────────────────\n\n")
            for ticker, s in results["fundamental_analysis"].items():
                f.write(f"[{ticker}]\n{s}\n\n")

        f.write("── ANOMALY & OUTLIER COMMENTARY ────────────────────────────────────\n\n")
        for ticker, s in results.get("anomaly_commentary", {}).items():
            f.write(f"[{ticker}]\n{s}\n\n")

        f.write("── RISK COMMENTARY ─────────────────────────────────────────────────\n\n")
        f.write(results.get("risk_commentary", "") + "\n\n")

        f.write("── COMPARATIVE / SECTOR ANALYSIS ───────────────────────────────────\n\n")
        f.write(results.get("comparative_analysis", "") + "\n\n")

        if results.get("commodity_context"):
            f.write("── COMMODITY MACRO CONTEXT (Alpha Vantage) ─────────────────────────\n\n")
            f.write(results["commodity_context"] + "\n\n")

    logger.info(f"  💾 Enriched report saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Sector Trending Analysis
# ─────────────────────────────────────────────────────────────────────────────

# US Sector ETFs — each represents one market sector
SECTOR_ETFS = {
    "Technology":        "XLK",
    "Healthcare":        "XLV",
    "Financials":        "XLF",
    "Energy":            "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":  "XLP",
    "Industrials":       "XLI",
    "Materials":         "XLB",
    "Real Estate":       "XLRE",
    "Utilities":         "XLU",
    "Communication":     "XLC",
}


def fetch_sector_etf_data() -> dict:
    """
    Download 3-month price data for all 11 US sector ETFs from Yahoo Finance.
    Returns a dict of {sector_name: performance_metrics}.
    """
    import yfinance as yf
    import time

    logger.info("Fetching sector ETF data from Yahoo Finance...")
    results = {}

    for sector, etf in SECTOR_ETFS.items():
        try:
            df = yf.download(etf, period="3mo", interval="1d",
                             progress=False, auto_adjust=True)

            if df.empty:
                continue

            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df["Close"].dropna()
            if len(close) < 2:
                continue

            # Calculate performance metrics
            ret_1w  = round((close.iloc[-1] / close.iloc[-5]  - 1) * 100, 2) if len(close) >= 5  else None
            ret_1m  = round((close.iloc[-1] / close.iloc[-21] - 1) * 100, 2) if len(close) >= 21 else None
            ret_3m  = round((close.iloc[-1] / close.iloc[0]   - 1) * 100, 2)
            vol_30d = round(float(close.pct_change().tail(21).std() * 100), 3)

            results[sector] = {
                "etf":          etf,
                "current_price":round(float(close.iloc[-1]), 2),
                "1w_return_pct":ret_1w,
                "1m_return_pct":ret_1m,
                "3m_return_pct":ret_3m,
                "volatility_30d":vol_30d,
                "trend":        "Rising" if (ret_1m or 0) > 1 else ("Falling" if (ret_1m or 0) < -1 else "Flat"),
            }
            logger.info(f"  ✓ {sector} ({etf}): 3M={ret_3m:+.1f}%")
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"  Failed to fetch {etf}: {e}")
            continue

    return results


def generate_sector_analysis(sector_data: dict, commodity_data: dict = None) -> str:
    """
    Generate a sector trending analysis using real ETF performance data.
    Tells the user which sectors are hot, which are cooling, and why.

    Args:
        sector_data:    Dict from fetch_sector_etf_data()
        commodity_data: Optional commodity prices for macro context

    Returns:
        Formatted analysis string
    """
    if not sector_data:
        return "⚠️  Could not fetch sector ETF data. Check your internet connection or VPN."

    # Sort sectors by 1-month return for easy ranking
    sorted_sectors = sorted(
        sector_data.items(),
        key=lambda x: x[1].get("1m_return_pct") or 0,
        reverse=True
    )

    system_prompt = """
You are a senior market strategist writing a sector rotation report for a university finance course.
Use ONLY the exact ETF performance numbers provided. Never invent figures.
Be specific, analytical, and actionable. Write for a finance-literate audience.
"""

    # Build a clean summary table for the prompt
    sector_summary = {
        sector: {
            "etf":           data["etf"],
            "1w_return_%":   data.get("1w_return_pct"),
            "1m_return_%":   data.get("1m_return_pct"),
            "3m_return_%":   data.get("3m_return_pct"),
            "volatility_%":  data.get("volatility_30d"),
            "trend":         data.get("trend"),
        }
        for sector, data in sorted_sectors
    }

    commodity_context = ""
    if commodity_data:
        oil   = commodity_data.get("WTI_Oil", {})
        gold  = commodity_data.get("Gold", {})
        commodity_context = f"""
COMMODITY CONTEXT (for macro backdrop):
- WTI Oil: ${oil.get('latest_price', 'N/A')} ({oil.get('trend', 'N/A')}, {oil.get('1m_change_pct', 'N/A'):+.1f}% 1M)
- Gold: ${gold.get('latest_price', 'N/A')} ({gold.get('trend', 'N/A')}, {gold.get('1m_change_pct', 'N/A'):+.1f}% 1M)
"""

    prompt = f"""
Write a sector rotation and trending analysis using the real ETF data below.

SECTOR ETF PERFORMANCE (sorted by 1-month return, best to worst):
{json.dumps(sector_summary, indent=2)}
{commodity_context}

Write using EXACTLY these section headers:

**Top Performing Sectors Right Now**
Name the top 3 sectors by 1-month return. State their exact 1M and 3M return figures.
What macro or economic themes are driving their outperformance?
Are these sectors showing momentum (also strong over 3M) or just a short-term bounce (strong 1M but weak 3M)?

**Sectors to Watch — Potential Opportunities**
Identify 2 sectors that have strong 3M returns but recent short-term momentum slowing (potential consolidation).
Also flag any sector with negative 1M returns but improving — could be a recovery play.
Reference specific ETF return figures.

**Weakest Sectors — Risks or Contrarian Opportunities**
Name the bottom 2-3 sectors by 1-month performance with exact figures.
What is likely driving their underperformance?
Is there a contrarian case for any of them, or are the headwinds structural?

**Sector Rotation Insight**
Based on the full picture, what does the current sector performance pattern suggest about
where we are in the economic/market cycle? (e.g. early recovery, late cycle, risk-off)
Which 2 sectors would you currently favour for a balanced portfolio and why?
Reference at least 3 specific data points in your conclusion.
"""
    return _call_groq(prompt, system_prompt, max_tokens=1000)


# ─────────────────────────────────────────────────────────────────────────────
#  yfinance Financial Statement Analysis (free, no quota)
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_financial_analysis(ticker: str, financial_data: dict) -> str:
    """
    Generate a comprehensive financial analysis using yfinance data.
    Covers income statement, balance sheet, cash flow, and key metrics dashboard.

    Args:
        ticker:         Stock symbol
        financial_data: Dict from fetch_all_financials()
    """
    income   = financial_data.get("income", {})
    balance  = financial_data.get("balance", {})
    cashflow = financial_data.get("cashflow", {})
    metrics  = financial_data.get("metrics", {})

    if not income.get("annual") and not income.get("quarterly"):
        return f"⚠️  No financial statement data available for {ticker}."

    system_prompt = """
You are a senior financial analyst writing a comprehensive financial statement review
for a university finance course. Use ONLY the exact numbers provided in the data.
Never invent or estimate figures. Format large numbers clearly (e.g. $2.3B, $450M).
Write professionally and specifically. Reference specific periods and figures.
"""

    prompt = f"""
Write a comprehensive financial analysis for {ticker} using the data below.
All monetary values are in millions USD (M) unless stated otherwise.

KEY METRICS DASHBOARD:
{json.dumps(metrics, indent=2)}

ANNUAL INCOME STATEMENT (most recent first):
{json.dumps(income.get('annual', [])[:3], indent=2)}

QUARTERLY INCOME STATEMENT (most recent first):
{json.dumps(income.get('quarterly', [])[:4], indent=2)}

ANNUAL BALANCE SHEET (most recent first):
{json.dumps(balance.get('annual', [])[:2], indent=2)}

ANNUAL CASH FLOW (most recent first):
{json.dumps(cashflow.get('annual', [])[:3], indent=2)}

Write using EXACTLY these section headers:

**📊 Key Metrics Dashboard**
State revenue_growth_yoy_pct and eps_growth_yoy_pct — is the company growing?
State gross_margin_trend — are margins expanding or compressing?
State roe_approx_pct — how efficiently is management using equity?
Give a 1-sentence overall health verdict based on these metrics.

**💰 Revenue & Profitability**
Walk through the annual revenue figures for all available years. Is growth accelerating or decelerating?
Compare gross_margin_pct and net_margin_pct across years — any compression or expansion?
Reference the most recent quarter's revenue and net income with exact figures.

**🏦 Balance Sheet Health**
State total_assets_M, total_liabilities_M, and equity_M for the most recent period.
Discuss debt_to_equity and current_ratio — is the company conservatively financed?
Is cash_M sufficient relative to short_term_debt_M and long_term_debt_M?

**💵 Cash Flow Analysis**
Compare operating_cashflow_M across years — is the company generating more or less cash?
Calculate and discuss free_cashflow_M — is the company cash generative after investments?
Comment on capex_M trend — is the company investing heavily for growth or cutting back?

**📈 Year-over-Year Comparison**
Using the quarterly data, compare the most recent quarter to the same quarter last year.
Which metric showed the strongest improvement? Which is a concern?

**⚠️ Key Risks & Red Flags**
Based on the data, identify 2-3 specific financial risks or warning signs.
Reference exact numbers that support each concern (e.g. rising debt, margin compression).

**✅ Investment Implication**
2-3 sentences: what does this financial picture mean for an investor?
Is the company's financial trajectory improving or deteriorating?
Reference at least 3 specific metrics from the data.
"""
    return _call_groq(prompt, system_prompt, max_tokens=1500)


def generate_rss_news_analysis(ticker: str, articles: list) -> str:
    """
    Generate a news sentiment analysis from RSS feed articles.
    No Alpha Vantage quota used — completely free via Yahoo Finance RSS.

    Args:
        ticker:   Stock symbol
        articles: List of article dicts from fetch_rss_news()
    """
    if not articles:
        return f"⚠️  No RSS news articles found for {ticker}."

    # Build headline list for the prompt
    headlines = "\n".join([
        f"  [{a['published'][:10]}] {a['title']} — {a['summary'][:100]}"
        for a in articles[:8]
    ])

    system_prompt = """
You are a financial news analyst writing a news review for a university finance course.
Reference specific headlines and dates from the provided list.
Do NOT fabricate any news — only reference what is given. Write professionally.
"""

    prompt = f"""
Analyse the following recent news headlines for {ticker} from Yahoo Finance RSS:

{headlines}

Write using EXACTLY these section headers:

**📰 News Overview**
How many articles were found and what is the general tone — mostly positive, negative, or mixed?
What are the 2-3 dominant themes across these headlines?

**🔍 Key Stories**
Highlight the 2-3 most significant headlines. Reference the exact title and date.
What do these stories suggest about the company's current situation?

**📊 Sentiment & Market Implication**
Based on the headlines, is current news sentiment likely to be a tailwind or headwind for the stock?
What should investors watch for in upcoming news coverage?
"""
    return _call_groq(prompt, system_prompt, max_tokens=600)


# ─────────────────────────────────────────────────────────────────────────────
#  yfinance Financial Statement Analysis (Free, No Quota)
# ─────────────────────────────────────────────────────────────────────────────

def generate_yfinance_income_analysis(ticker: str, income: dict) -> str:
    """Analyse income statement data fetched from yfinance."""
    annual = income.get("annual", [])
    quarterly = income.get("quarterly", [])
    yoy = income.get("yoy_growth", {})

    if not annual:
        return f"⚠️  No income statement data available for {ticker}."

    system_prompt = """
You are a financial analyst writing an income statement review for a university finance course.
Use ONLY the exact numbers provided. Format large numbers clearly ($2.3B, $450M).
Never invent figures. Write professionally and specifically.
"""
    prompt = f"""
Analyse the income statement for {ticker} using the data below.

ANNUAL DATA (most recent first):
{json.dumps(annual[:4], indent=2, default=str)}

QUARTERLY DATA (most recent 4 quarters):
{json.dumps(quarterly[:4], indent=2, default=str)}

YEAR-OVER-YEAR GROWTH:
{json.dumps(yoy, indent=2)}

Write using EXACTLY these section headers:

**Revenue Trend**
State annual revenue for each year using revenue_fmt values.
Calculate and state the year-over-year revenue growth from yoy_growth.
Is revenue accelerating or decelerating?

**Profitability Analysis**
Compare gross_margin_pct, operating_margin_pct, and net_margin_pct across years.
Are margins expanding or compressing? What does this mean for the business?

**Quarterly Momentum**
Using the quarterly data, what is the most recent quarter's revenue and net income?
How does it compare to the prior quarter? Is there short-term acceleration?

**R&D Investment**
If rd_expense data is available, comment on R&D spending as a % of revenue.
What does this suggest about the company's investment in future growth?

**Key Takeaway**
2-3 sentences summarising the income statement story. Reference at least 3 specific numbers.
"""
    return _call_groq(prompt, system_prompt, max_tokens=1000)


def generate_yfinance_balance_analysis(ticker: str, balance: dict) -> str:
    """Analyse balance sheet data fetched from yfinance."""
    annual = balance.get("annual", [])

    if not annual:
        return f"⚠️  No balance sheet data available for {ticker}."

    system_prompt = """
You are a financial analyst writing a balance sheet review for a university finance course.
Use ONLY the exact numbers provided. Format large numbers clearly ($2.3B, $450M).
Never invent figures. Write professionally and specifically.
"""
    prompt = f"""
Analyse the balance sheet for {ticker} using the data below.

ANNUAL BALANCE SHEET DATA (most recent first):
{json.dumps(annual[:4], indent=2, default=str)}

Write using EXACTLY these section headers:

**Asset Base & Growth**
State total_assets for each year using assets_fmt. Is the asset base growing?
What does this say about business expansion and capital investment?

**Debt & Leverage**
State long_term_debt and de_ratio_pct for the most recent year.
Is debt increasing or decreasing? Is the leverage level sustainable?

**Equity & Retained Earnings**
State shareholder_equity using equity_fmt and compare across years.
Is equity growing (good) or shrinking (concern)? Reference retained_earnings if available.

**Cash Position & Liquidity**
State cash using cash_fmt. How does cash compare to long_term_debt?
Can the company comfortably service its debt from its cash position?

**Key Takeaway**
2-3 sentences on financial strength and stability. Reference at least 3 specific numbers.
"""
    return _call_groq(prompt, system_prompt, max_tokens=900)


def generate_yfinance_cashflow_analysis(ticker: str, cashflow: dict) -> str:
    """Analyse cash flow statement fetched from yfinance."""
    annual = cashflow.get("annual", [])
    fcf_growth = cashflow.get("fcf_growth_pct")

    if not annual:
        return f"⚠️  No cash flow data available for {ticker}."

    system_prompt = """
You are a financial analyst writing a cash flow review for a university finance course.
Use ONLY the exact numbers provided. Format large numbers clearly ($2.3B, $450M).
Free cash flow analysis is especially important — focus on it.
"""
    prompt = f"""
Analyse the cash flow statement for {ticker} using the data below.

ANNUAL CASH FLOW DATA (most recent first):
{json.dumps(annual[:4], indent=2, default=str)}

FCF YEAR-OVER-YEAR GROWTH: {fcf_growth}%

Write using EXACTLY these section headers:

**Operating Cash Flow**
State operating_cf using operating_cf_fmt for each year. Is the company generating
more or less cash from operations? What does the trend suggest?

**Free Cash Flow (FCF)**
State free_cash_flow using fcf_fmt for each year.
FCF growth is {fcf_growth}% year-over-year — is this strong or concerning?
FCF is the truest measure of a company's cash-generating ability — explain why.

**Capital Expenditure**
State capex using capex_fmt. Is the company investing heavily in infrastructure?
Is capex rising or falling relative to operating cash flow?

**Financing Activities**
Comment on financing_cf — is the company raising debt, repurchasing shares, or paying dividends?
What does this say about management's capital allocation priorities?

**Key Takeaway**
2-3 sentences on cash flow quality and sustainability. Reference at least 3 specific numbers.
FCF is king — make sure to highlight whether this company is truly cash-generative.
"""
    return _call_groq(prompt, system_prompt, max_tokens=900)


def generate_key_metrics_dashboard(ticker: str, metrics: dict) -> str:
    """
    Generate a key metrics dashboard analysis including EPS growth,
    revenue growth, FCF yield, valuation ratios, and analyst consensus.
    """
    if not metrics:
        return f"⚠️  No key metrics data available for {ticker}."

    system_prompt = """
You are an equity analyst writing a metrics dashboard summary for a university finance report.
Use ONLY the exact numbers provided. Be specific and analytical.
Compare each metric to typical market benchmarks where relevant (S&P 500 avg P/E ~20-25x,
good FCF yield >3%, healthy gross margin varies by sector).
"""
    prompt = f"""
Write a key metrics dashboard for {ticker} using the data below.

METRICS DATA:
{json.dumps(metrics, indent=2, default=str)}

Write using EXACTLY these section headers:

**Valuation Snapshot**
State current P/E (trailing and forward), P/B, P/S, and PEG ratio.
Is the stock cheap or expensive vs typical market averages?
What does the difference between trailing and forward P/E suggest about earnings expectations?

**Growth Profile**
State EPS TTM, forward EPS, and EPS growth %.
State revenue YoY growth %. Is this company growing faster or slower than expected?

**Profitability & Cash Generation**
State gross margin, operating margin, and net margin as percentages.
State FCF yield % — is the company generating strong free cash flow relative to its market cap?

**Financial Health**
State current ratio and debt-to-equity. Is the balance sheet strong or stretched?
State beta — is this a high or low volatility stock vs the market?

**Analyst Consensus**
State the analyst recommendation, target price, and upside % from current price.
How many analysts cover this stock? Does the market agree or disagree with analysts?
"""
    return _call_groq(prompt, system_prompt, max_tokens=900)


def generate_rss_news_analysis(ticker: str, news_df) -> str:
    """
    Generate a news analysis section from free RSS feed articles.
    Uses keyword-based sentiment hints from the RSS fetcher.
    """
    import pandas as pd

    if news_df is None or (hasattr(news_df, 'empty') and news_df.empty):
        return f"⚠️  No RSS news data available for {ticker}."

    # Filter to this ticker
    if 'ticker' in news_df.columns:
        ticker_news = news_df[news_df['ticker'] == ticker].head(8)
    else:
        ticker_news = news_df.head(8)

    if ticker_news.empty:
        return f"⚠️  No news articles found for {ticker}."

    # Count sentiment hints
    pos = (ticker_news['sentiment_hint'] == 'Positive').sum() if 'sentiment_hint' in ticker_news.columns else 0
    neg = (ticker_news['sentiment_hint'] == 'Negative').sum() if 'sentiment_hint' in ticker_news.columns else 0
    neu = len(ticker_news) - pos - neg

    # Build headlines string for prompt
    headlines = "\n".join([
        f"  [{row.get('published_at', 'N/A')} | {row.get('source', 'N/A')} | {row.get('sentiment_hint', 'N/A')}] "
        f"{row.get('title', '')}"
        for _, row in ticker_news.iterrows()
    ])

    system_prompt = """
You are a financial news analyst writing a market news section for a university finance report.
Reference specific headlines and dates from the data. Do NOT fabricate any news.
Write in 3 structured paragraphs with the headers shown.
"""
    prompt = f"""
Analyse the following news articles for {ticker}:

SENTIMENT SUMMARY: {pos} Positive | {neg} Negative | {neu} Neutral articles

HEADLINES:
{headlines}

Write using EXACTLY these section headers:

**News Overview**
Summarise the overall news sentiment ({pos} positive, {neg} negative, {neu} neutral).
What is the dominant narrative around {ticker} in recent coverage?

**Key Themes**
Identify 2-3 recurring themes across the headlines.
Reference specific article titles and their sentiment hints.
Which story is getting the most attention?

**Implication for Investors**
How does the current news flow align with or contradict the stock's recent price action?
What should investors watch for in upcoming news coverage of {ticker}?
"""
    return _call_groq(prompt, system_prompt, max_tokens=700)