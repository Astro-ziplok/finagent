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
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL, MAX_TOKENS, REPORTS_DIR,
                    RISK_FREE_RATE, TRADING_DAYS, BENCHMARK_TICKER,
                    REASONING_TOKEN_BUFFER, GROQ_EXTRA_BODY)

# Shared rule appended to every system prompt: numbers and comparisons are
# pre-computed in Python, so the LLM must not redo arithmetic or flip directions.
NUMERIC_RULES = """
Numeric rules (mandatory):
- Every comparison, ranking and direction (higher/lower, increased/decreased, beat/lagged)
  is already computed for you in fields such as *_direction, *_ranking, strongest_*,
  outperformed_*, or *_interpretation. Repeat those conclusions; never recompute or contradict them.
- Ratios ending in _x are multiples (0.8x), fields ending in _pct are percentages.
- Sharpe ratios are ANNUALISED: below 0.5 is weak, 0.5-1 moderate, 1-2 good, above 2 very strong.
- If a value is null/None, say it is not available. Do not estimate it.
"""

DISCLAIMER = ("Disclaimer: This report is generated automatically for educational purposes "
              "only and is not investment advice.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _annualised_sharpe(returns_pct: pd.Series) -> float:
    """
    Annualised Sharpe ratio from daily % returns.
    Formula: (mean daily return - daily risk-free rate) / daily std * sqrt(252)
    """
    r = returns_pct.dropna() / 100
    if len(r) < 2 or r.std() == 0:
        return 0.0
    rf_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    return round(float((r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS)), 2)


def _sharpe_label(sharpe: float) -> str:
    """Plain-English interpretation of an annualised Sharpe ratio."""
    if sharpe < 0:
        return "negative: the stock returned less than the risk-free rate"
    if sharpe < 0.5:
        return "weak risk-adjusted return"
    if sharpe < 1.0:
        return "moderate risk-adjusted return"
    if sharpe < 2.0:
        return "good risk-adjusted return"
    if sharpe < 3.0:
        return "very strong risk-adjusted return"
    return "exceptional risk-adjusted return (rarely sustained over long periods)"


def _pct_over(close: pd.Series, n_days: int):
    """% change over the last n trading days (None if not enough data)."""
    if len(close) <= n_days:
        return None
    return round((float(close.iloc[-1]) / float(close.iloc[-1 - n_days]) - 1) * 100, 2)


def _calendar_breakdown(df: pd.DataFrame, freq: str, max_periods: int) -> list:
    """
    Calendar-period (quarter or year) performance breakdown.
    Each period's return is measured from the PREVIOUS period's last close,
    so period returns chain together correctly. Periods with too few trading
    days are flagged as partial.
    """
    close = df["Close"]
    groups = list(df.groupby(df.index.to_period(freq)))
    full_len = 60 if freq == "Q" else 240
    rows = []
    prev_close = None
    for period, chunk in groups:
        if chunk.empty:
            continue
        base = prev_close if prev_close is not None else float(chunk["Close"].iloc[0])
        end  = float(chunk["Close"].iloc[-1])
        rows.append({
            "period":           str(period).replace("Q", "-Q") if freq == "Q" else str(period),
            "start_date":       str(chunk.index[0].date()),
            "end_date":         str(chunk.index[-1].date()),
            "start_price":      round(base, 2),
            "end_price":        round(end, 2),
            "pct_change":       round((end / base - 1) * 100, 2),
            "trading_days":     len(chunk),
            "partial_period":   len(chunk) < full_len,
            "avg_daily_volume": int(chunk["Volume"].mean()),
        })
        prev_close = end
    return rows[-max_periods:]


def _benchmark_stats(df: pd.DataFrame, bench_df: pd.DataFrame) -> dict:
    """Compare a stock to a market benchmark over the overlapping dates."""
    joined = pd.concat(
        [df["Close"].rename("stock"), bench_df["Close"].rename("bench")], axis=1
    ).dropna()
    if len(joined) < 20:
        return {}
    rets = joined.pct_change().dropna()
    stock_ret = (joined["stock"].iloc[-1] / joined["stock"].iloc[0] - 1) * 100
    bench_ret = (joined["bench"].iloc[-1] / joined["bench"].iloc[0] - 1) * 100
    beta = rets["stock"].cov(rets["bench"]) / rets["bench"].var() if rets["bench"].var() else None
    corr = rets["stock"].corr(rets["bench"])
    return {
        "benchmark":                BENCHMARK_TICKER,
        "benchmark_return_pct":     round(float(bench_ret), 2),
        "stock_return_pct":         round(float(stock_ret), 2),
        "excess_return_pct":        round(float(stock_ret - bench_ret), 2),
        "outperformed_benchmark":   bool(stock_ret > bench_ret),
        "beta_vs_benchmark":        round(float(beta), 2) if beta is not None else None,
        "beta_basis":               "daily returns over this period (can differ from Yahoo's 5-year monthly beta)",
        "correlation_vs_benchmark": round(float(corr), 2),
        "benchmark_sharpe_annualised": _annualised_sharpe(rets["bench"] * 100),
    }


def _build_data_context(cleaned_data: dict, benchmark_df: pd.DataFrame = None) -> dict:
    """
    Extract a rich, structured JSON-serialisable summary of the data.
    Includes calendar-quarter breakdown, MA signals, Bollinger Bands,
    win rate, annualised Sharpe ratio, multi-window momentum and
    (optionally) performance versus a market benchmark such as SPY.
    """
    context = {}

    for ticker, df in cleaned_data.items():
        returns   = df["Daily_Return"].dropna()
        recent_30 = df.tail(30)
        close     = df["Close"]

        # Calendar-quarter breakdown (last 8 quarters) + yearly for long ranges
        quarterly = _calendar_breakdown(df, "Q", max_periods=8)
        span_days = (df.index[-1] - df.index[0]).days
        yearly    = _calendar_breakdown(df, "Y", max_periods=10) if span_days > 550 else []

        full_quarters = [q for q in quarterly if not q["partial_period"]] or quarterly
        best_q  = max(full_quarters, key=lambda q: q["pct_change"]) if full_quarters else None
        worst_q = min(full_quarters, key=lambda q: q["pct_change"]) if full_quarters else None

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

        sharpe   = _annualised_sharpe(returns)
        win_rate = round((returns > 0).sum() / len(returns) * 100, 1)

        # Label the high/low honestly: it is only a "52-week" figure for ~1y of data
        is_one_year  = 330 <= span_days <= 400
        range_label  = "52-week" if is_one_year else f"period ({df.index[0].date()} to {df.index[-1].date()})"
        period_high  = float(df["High"].max())
        period_low   = float(df["Low"].min())

        daily_std    = float(returns.std())
        annual_vol   = round(daily_std * np.sqrt(TRADING_DAYS), 2)
        recent_vol   = round(float(recent_30["Daily_Return"].std()), 3)
        m7, m30, m90 = _pct_over(close, 7), _pct_over(close, 30), _pct_over(close, 90)

        context[ticker] = {
            "period": {
                "start":        str(df.index[0].date()),
                "end":          str(df.index[-1].date()),
                "trading_days": len(df),
                "is_one_year":  is_one_year,
            },
            "price_summary": {
                "start_price":          round(float(close.iloc[0]), 2),
                "end_price":            round(float(close.iloc[-1]), 2),
                "high_low_label":       range_label,
                "period_high":          round(period_high, 2),
                "period_low":           round(period_low, 2),
                "total_return_pct":     round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
                "pct_below_period_high": round((current_price / period_high - 1) * 100, 2),
                "pct_above_period_low":  round((current_price / period_low - 1) * 100, 2),
            },
            "returns": {
                "mean_daily_pct":          round(float(returns.mean()), 3),
                "std_daily_pct":           round(daily_std, 3),
                "annualised_volatility_pct": annual_vol,
                "best_day_pct":            round(float(returns.max()), 2),
                "best_day_date":           str(returns.idxmax().date()),
                "worst_day_pct":           round(float(returns.min()), 2),
                "worst_day_date":          str(returns.idxmin().date()),
                "win_rate_pct":            win_rate,
                "sharpe_ratio_annualised": sharpe,
                "sharpe_interpretation":   _sharpe_label(sharpe),
                "risk_free_rate_used_pct": round(RISK_FREE_RATE * 100, 2),
            },
            "short_term_momentum": {
                "last_7d_change_pct":  m7,
                "last_30d_change_pct": m30,
                "last_90d_change_pct": m90,
                "last_30d_volatility": recent_vol,
                "recent_volatility_vs_full_period": (
                    "higher" if recent_vol > daily_std else "lower"
                ),
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
            "strongest_quarter":   best_q["period"] if best_q else None,
            "weakest_quarter":     worst_q["period"] if worst_q else None,
            "yearly_breakdown":    yearly,
            "outlier_dates": df[df["Is_Outlier"] == True].index.strftime("%Y-%m-%d").tolist(),
        }

        if benchmark_df is not None and not benchmark_df.empty and ticker != BENCHMARK_TICKER:
            bench = _benchmark_stats(df, benchmark_df)
            if bench:
                context[ticker]["vs_benchmark"] = bench

    return context


def _retry_wait_seconds(error_text: str, attempt: int) -> float:
    """
    How long to wait before retrying. Groq's 429 errors say e.g.
    "Please try again in 7.66s" (per-minute token limits), so honour that
    instead of guessing; fall back to exponential backoff.
    """
    import re
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", error_text)
    if m:
        return min(float(m.group(1) or 0) * 60 + float(m.group(2)) + 1, 90)
    return 2 ** (attempt + 1)


def _call_groq(prompt: str, system_prompt: str = "", max_tokens: int = MAX_TOKENS,
               json_mode: bool = False) -> str:
    """
    Send a prompt to Groq and return the response text.
    With json_mode=True the model is constrained to return a JSON object.
    """
    if not GROQ_API_KEY:
        return "⚠️  GROQ_API_KEY not set in .env file. Skipping AI analysis."

    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    messages = [{"role": "system", "content": (system_prompt or "") + NUMERIC_RULES},
                {"role": "user", "content": prompt}]

    # Retry on rate limits / transient errors: the Groq free tier limits
    # tokens per minute, and a full report makes several calls in a row.
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens + (REASONING_TOKEN_BUFFER if GROQ_EXTRA_BODY else 0),
                temperature=0.2,
                extra_body=GROQ_EXTRA_BODY or None,
                response_format={"type": "json_object"} if json_mode else None,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("empty response (token budget used up by reasoning)")
            return content
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            retryable = any(s in msg for s in ("429", "rate limit", "timeout", "503", "502",
                                               "empty response", "json_validate_failed"))
            if not retryable or attempt == 4:
                break
            wait = _retry_wait_seconds(msg, attempt)
            logger.warning(f"Groq rate-limited/unavailable, retrying in {wait:.0f}s...")
            time.sleep(wait)
    logger.error(f"Groq API error: {last_error}")
    return f"⚠️  Groq API call failed: {last_error}"


def _call_groq_json(prompt: str, system_prompt: str = "", max_tokens: int = MAX_TOKENS) -> dict:
    """Call Groq in JSON mode and parse the result. Returns {} on any failure."""
    raw = _call_groq(prompt, system_prompt, max_tokens=max_tokens, json_mode=True)
    if raw.startswith("⚠️"):
        logger.error(raw)
        return {}
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: take the outermost {...} block
        a, b = raw.find("{"), raw.rfind("}")
        try:
            data = json.loads(raw[a:b + 1]) if a != -1 and b > a else {}
        except json.JSONDecodeError:
            logger.error("Could not parse JSON from the language model.")
            data = {}
    return data if isinstance(data, dict) else {}


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
Summarise the return over the period (total_return_pct). State the start and end price.
Compare to the high and low, calling them by high_low_label (only say "52-week" if that is the label).
If vs_benchmark is present, state the benchmark return and excess_return_pct, and whether the stock
outperformed (use outperformed_benchmark).

**2. Quarterly Breakdown**
Walk through each calendar quarter in quarterly_breakdown (use the period labels, e.g. 2025-Q3).
Name strongest_quarter and weakest_quarter with their pct_change. Mention if a quarter is a partial_period.
If yearly_breakdown is non-empty, summarise the year-by-year returns briefly.

**3. Short-Term Momentum**
Discuss the 7-day, 30-day, and 90-day price changes from short_term_momentum.
Is the stock accelerating or decelerating recently?

**4. Technical Signal**
State the moving average signal (ma_signal) and the Bollinger Band signal (bb_signal).
Note that these are short-term indicators: price can stay near a Bollinger band during strong trends,
so present them as context rather than predictions.

**5. Risk & Return Profile**
Use annualised_volatility_pct, win_rate_pct, best_day_pct and worst_day_pct with their dates,
and sharpe_ratio_annualised with its sharpe_interpretation (risk-free rate: risk_free_rate_used_pct).
If vs_benchmark is present, mention beta_vs_benchmark (above 1 = moves more than the market).

**6. Investment Implication**
Based on ALL the sections above, write 2-3 sentences answering: "So what does this mean for an investor?"
State whether the risk profile looks more suitable for a risk-tolerant or risk-averse investor.
Reference at least 2 specific numbers from the data. Do not give a buy/sell recommendation.
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
    total_return = stats["price_summary"]["total_return_pct"]
    best_date    = stats["returns"]["best_day_date"]
    worst_date   = stats["returns"]["worst_day_date"]
    outliers     = stats["outlier_dates"]

    # The LLM's training data ends before the analysis period, so this section
    # must stay at the level of durable background knowledge. It must not
    # claim to know what happened between period_start and period_end.
    system_prompt = """
You are a financial analyst writing background context for a university finance course report.
IMPORTANT: Your training knowledge may end BEFORE the analysis period. You must NOT claim to know
specific events, product launches, earnings results or news that happened during the analysis period.
Only describe durable facts (business model, sector, structural themes) and typical drivers.
Never invent dates, figures, or events.
"""

    prompt = f"""
Write a "Background Context" section for {ticker}. The price data covers {period_start} to {period_end}.

Key data points:
- Total return over the period: {total_return}%
- Best single trading day: {best_date}
- Worst single trading day: {worst_date}
- Statistically extreme outlier dates: {outliers if outliers else "None detected"}

Write 3 short paragraphs with these headers:

**Company & Sector Overview**
Describe what {ticker} does, its sector/industry, and its main revenue drivers (durable facts only).

**Structural Themes to Consider**
Name 2-3 long-running themes (e.g. interest-rate sensitivity, AI investment cycle, regulation,
consumer demand) that typically affect {ticker}. Phrase them as factors to check, not as claims about this period.

**What Typically Drives Large Moves**
For the best/worst/outlier dates above, list the types of events that usually cause such moves for
{ticker} (earnings releases, guidance changes, macro data, analyst actions). State clearly that the
actual cause of each date should be verified against news from that date.
"""
    note = ("_Note: background knowledge from the language model, not live news. "
            "Verify against news sources for the analysis period._\n")
    return note + _call_groq(prompt, system_prompt, max_tokens=800)


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
            "annualised_volatility_pct": data["returns"]["annualised_volatility_pct"],
            "daily_std_pct":             data["returns"]["std_daily_pct"],
            "recent_30d_volatility":     data["short_term_momentum"]["last_30d_volatility"],
            "recent_volatility_vs_full_period": data["short_term_momentum"]["recent_volatility_vs_full_period"],
            "worst_day_pct":             data["returns"]["worst_day_pct"],
            "worst_day_date":            data["returns"]["worst_day_date"],
            "sharpe_ratio_annualised":   data["returns"]["sharpe_ratio_annualised"],
            "sharpe_interpretation":     data["returns"]["sharpe_interpretation"],
            "win_rate_pct":              data["returns"]["win_rate_pct"],
            "total_return_pct":          data["price_summary"]["total_return_pct"],
            "beta_vs_benchmark":         data.get("vs_benchmark", {}).get("beta_vs_benchmark"),
        }
        for ticker, data in data_context.items()
    }

    # Pre-computed rankings so the LLM doesn't have to sort numbers itself
    tickers = list(risk_summary)
    derived = {
        "volatility_ranking_least_to_most_risky":
            sorted(tickers, key=lambda t: risk_summary[t]["annualised_volatility_pct"]),
        "sharpe_ranking_best_to_worst":
            sorted(tickers, key=lambda t: risk_summary[t]["sharpe_ratio_annualised"], reverse=True),
        "single_worst_day": min(
            ({"ticker": t, "pct": risk_summary[t]["worst_day_pct"],
              "date": risk_summary[t]["worst_day_date"]} for t in tickers),
            key=lambda d: d["pct"]),
    }

    system_prompt = """
You are a risk analyst writing a risk assessment for a university finance project.
Use only the provided data. Reference specific numbers in every claim.
Write 2 well-structured paragraphs.
"""

    single = len(tickers) == 1
    prompt = f"""
Write a risk commentary for {"this asset" if single else "these assets"} using the data below:

{json.dumps(risk_summary, indent=2)}

PRE-COMPUTED FACTS (use these exactly):
{json.dumps(derived, indent=2)}

Paragraph 1 — Volatility:
{"State the annualised volatility" if single else "Give the volatility_ranking_least_to_most_risky order, with each annualised_volatility_pct"}.
State the worst day and its date (single_worst_day).
Say whether recent 30-day volatility is higher or lower than the full period (recent_volatility_vs_full_period).

Paragraph 2 — Risk-Adjusted Performance:
Use sharpe_ratio_annualised with its sharpe_interpretation{"" if single else " and the sharpe_ranking_best_to_worst"}.
Discuss whether the return justified the volatility. If beta_vs_benchmark is present, explain it.
Conclude with a practical implication for a risk-averse vs risk-tolerant investor (no buy/sell advice).
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

        bench = data.get("vs_benchmark")

        system_prompt = """
You are a senior equity analyst writing a market benchmarking section for a university finance report.
Be specific and analytical. Reference only the exact numbers provided.
Do NOT state peer or sector returns from memory: only the benchmark data given here is real.
Write exactly 3 paragraphs with the headers shown.
"""
        prompt = f"""
Only one asset ({ticker}) was analysed. Write a benchmarking analysis against the market benchmark.

Data for {ticker}:
- Total return: {data['price_summary']['total_return_pct']}%
- Annualised volatility: {data['returns']['annualised_volatility_pct']}%
- Annualised Sharpe ratio: {data['returns']['sharpe_ratio_annualised']} ({data['returns']['sharpe_interpretation']})
- Win rate: {data['returns']['win_rate_pct']}%
- 30-day momentum: {data['short_term_momentum']['last_30d_change_pct']}%
- MA signal: {data['technical_signals']['ma_signal']}
- BB signal: {data['technical_signals']['bb_signal']}

BENCHMARK COMPARISON (real data, may be empty):
{json.dumps(bench or {}, indent=2)}

Write using EXACTLY these section headers:

**Benchmark Comparison**
{"State the benchmark return, the stock's return, excess_return_pct, and whether it outperformed (outperformed_benchmark)." if bench else "Benchmark data is unavailable; say so and describe the absolute return only."}

**Risk Profile vs Market**
{"Compare the stock's Sharpe ratio to benchmark_sharpe_annualised, and explain beta_vs_benchmark and correlation_vs_benchmark." if bench else "Describe the volatility and Sharpe ratio on an absolute basis."}

**Relative Momentum & Outlook**
Given the 30-day momentum and the technical signals, describe the near-term picture.
State what an investor should monitor next. Do not give a buy/sell recommendation.
"""
        return _call_groq(prompt, system_prompt, max_tokens=700)

    # ── Multi-stock: full comparison ──────────────────────────────────────────
    comparison_data = {
        ticker: {
            "total_return_pct":       data["price_summary"]["total_return_pct"],
            "current_price":          data["price_summary"]["end_price"],
            "period_high":            data["price_summary"]["period_high"],
            "pct_below_period_high":  data["price_summary"]["pct_below_period_high"],
            "annualised_volatility_pct": data["returns"]["annualised_volatility_pct"],
            "sharpe_ratio_annualised": data["returns"]["sharpe_ratio_annualised"],
            "win_rate_pct":           data["returns"]["win_rate_pct"],
            "last_30d_change_pct":    data["short_term_momentum"]["last_30d_change_pct"],
            "excess_return_vs_benchmark_pct": data.get("vs_benchmark", {}).get("excess_return_pct"),
            "ma_signal":              data["technical_signals"]["ma_signal"],
            "bb_signal":              data["technical_signals"]["bb_signal"],
        }
        for ticker, data in data_context.items()
    }

    def rank(field, reverse=True):
        vals = {t: v[field] for t, v in comparison_data.items() if v[field] is not None}
        return [f"{t} ({vals[t]})" for t in sorted(vals, key=vals.get, reverse=reverse)]

    returns_rank = rank("total_return_pct")
    derived = {
        "return_ranking_best_to_worst":     returns_rank,
        "return_spread_pct_points": round(
            max(v["total_return_pct"] for v in comparison_data.values())
            - min(v["total_return_pct"] for v in comparison_data.values()), 2),
        "furthest_below_period_high":       rank("pct_below_period_high", reverse=False)[0],
        "volatility_ranking_lowest_first":  rank("annualised_volatility_pct", reverse=False),
        "sharpe_ranking_best_to_worst":     rank("sharpe_ratio_annualised"),
        "win_rate_ranking_best_to_worst":   rank("win_rate_pct"),
        "momentum_30d_ranking_strongest_first": rank("last_30d_change_pct"),
    }

    system_prompt = """
You are a senior equity analyst writing a comparative stock analysis for a finance course.
Use only the provided numbers. Be specific, analytical, and objective.
Write exactly 3 paragraphs with the headers shown.
"""

    prompt = f"""
Write a comparative analysis of the following assets:

{json.dumps(comparison_data, indent=2)}

PRE-COMPUTED RANKINGS (use these exactly, do not re-sort):
{json.dumps(derived, indent=2)}

Use EXACTLY these section headers:

**Return Comparison**
Give return_ranking_best_to_worst. Mention excess_return_vs_benchmark_pct where available.
Name furthest_below_period_high and what that implies.
End with 1 sentence on what return_spread_pct_points tells us.

**Risk & Consistency Comparison**
Use volatility_ranking_lowest_first and sharpe_ranking_best_to_worst (Sharpe is annualised).
Use win_rate_ranking_best_to_worst for consistency.
The best risk-adjusted return is the first entry of sharpe_ranking_best_to_worst; explain why.

**Momentum & Technical Outlook**
Use momentum_30d_ranking_strongest_first.
Summarise the ma_signal and bb_signal for each asset in one sentence each.
Conclude which asset looks most technically constructive right now, framed as an observation, not a recommendation.
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

def generate_fundamental_analysis(ticker: str, overview: dict, income: dict, balance: dict,
                                  current_price: float = None) -> str:
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

    # Pre-compute percentages and directions instead of asking the LLM to do arithmetic.
    def pct(v):
        return round(v * 100, 2) if v is not None else None
    facts = {
        "roe_pct": pct(overview.get("roe")),
        "roa_pct": pct(overview.get("roa")),
        "profit_margin_pct": pct(overview.get("profit_margin")),
        "operating_margin_pct": pct(overview.get("operating_margin")),
        "current_price": current_price,
    }
    target = overview.get("analyst_target_price")
    if target and current_price:
        up = round((target / current_price - 1) * 100, 2)
        facts["analyst_target_implies"] = (f"{up}% upside" if up > 0 else f"{abs(up)}% downside")
    if len(income_summary) >= 2:
        q0, q1 = income_summary[0], income_summary[1]
        facts["revenue_latest_vs_prior_quarter"] = _direction(q0.get("total_revenue"), q1.get("total_revenue"))
        facts["net_income_latest_vs_prior_quarter"] = _direction(q0.get("net_income"), q1.get("net_income"))

    prompt = f"""
Write a fundamental analysis for {ticker} using the data below.

COMPANY OVERVIEW:
{json.dumps(overview, indent=2)}

RECENT INCOME STATEMENT (last 2 quarters):
{json.dumps(income_summary, indent=2)}

LATEST BALANCE SHEET:
{json.dumps(balance_summary, indent=2)}

PRE-COMPUTED FACTS (use these exactly):
{json.dumps(facts, indent=2)}

Use EXACTLY these section headers:

**Company Profile**
State the company name, sector, industry, and market cap.
Write 1 sentence describing the business using the description field.

**Valuation Assessment**
Discuss the P/E ratio — is the stock cheap or expensive relative to typical market averages (S&P 500 avg ~20-25x)?
Comment on P/B ratio and PEG ratio if available. What do these suggest about market expectations?

**Profitability**
State roe_pct, roa_pct, profit_margin_pct and operating_margin_pct from PRE-COMPUTED FACTS.
Is the company efficiently generating profit?

**Financial Health**
Using the latest balance sheet, compare cash with long_term_debt and short_term_debt.
If debt_to_equity or current_ratio are null, say they are not provided by this source.

**Revenue & Earnings Trend**
Use revenue_latest_vs_prior_quarter and net_income_latest_vs_prior_quarter exactly as given.
Note that sequential quarters can be affected by seasonality.

**Analyst Consensus**
State the analyst target price and analyst_target_implies (if available).
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
    # Only keep commodities that actually returned a price. If none did,
    # don't ask the LLM to write paragraphs about missing data.
    commodity_data = {k: v for k, v in (commodity_data or {}).items()
                      if isinstance(v, dict) and v.get("latest_price") is not None}
    if not commodity_data:
        return ("⚠️  No commodity price data could be retrieved (Yahoo Finance returned nothing). "
                "Check your connection/VPN and try again.")

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
            # 1 week = 5 trading days back, 1 month = 21 trading days back
            ret_1w  = _pct_over(close, 5)
            ret_1m  = _pct_over(close, 21)
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

    def _commodity_line(name, d):
        # Format safely: any field may be missing when a download fails
        price, chg = d.get("latest_price"), d.get("1m_change_pct")
        if price is None:
            return f"- {name}: not available"
        chg_txt = f"{chg:+.1f}% 1M" if isinstance(chg, (int, float)) else "1M change n/a"
        return f"- {name}: ${price} ({d.get('trend', 'N/A')}, {chg_txt})"

    commodity_context = ""
    if commodity_data:
        commodity_context = (
            "\nCOMMODITY CONTEXT (for macro backdrop):\n"
            + _commodity_line("WTI Oil", commodity_data.get("WTI_Oil") or {}) + "\n"
            + _commodity_line("Gold", commodity_data.get("Gold") or {}) + "\n"
        )

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


# ─────────────────────────────────────────────────────────────────────────────
#  yfinance Financial Statement Analysis (Free, No Quota)
# ─────────────────────────────────────────────────────────────────────────────

def _direction(new, old, flat_tolerance_pct: float = 0.5) -> str:
    """'increased' / 'decreased' / 'roughly flat' between two values (None-safe)."""
    if new is None or old is None or old == 0:
        return "n/a"
    change = (new - old) / abs(old) * 100
    if abs(change) < flat_tolerance_pct:
        return "roughly flat"
    return "increased" if change > 0 else "decreased"


def _chg_pct(new, old):
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100, 2)


def _fmt_big(v):
    """Format a raw USD amount as $X.XXB / $X.XM."""
    if v is None:
        return "N/A"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e9:
        return f"{sign}${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{sign}${v / 1e6:.1f}M"
    return f"{sign}${v:,.0f}"


def _series_facts(rows: list, field: str, label: str) -> dict:
    """
    Year-by-year changes for one field. `rows` are most-recent-first.
    Returns the chronological changes plus whether growth is accelerating.
    """
    chron = [r for r in reversed(rows) if r.get(field) is not None]
    steps = []
    for prev, curr in zip(chron, chron[1:]):
        steps.append({
            "from_period": prev["period"][:10], "to_period": curr["period"][:10],
            "from": _fmt_big(prev[field]) if abs(prev[field]) > 1e5 else prev[field],
            "to":   _fmt_big(curr[field]) if abs(curr[field]) > 1e5 else curr[field],
            "change_pct": _chg_pct(curr[field], prev[field]),
            "direction":  _direction(curr[field], prev[field]),
        })
    facts = {f"{label}_changes": steps}
    if len(steps) >= 2 and steps[-1]["change_pct"] is not None and steps[-2]["change_pct"] is not None:
        last, prev = steps[-1]["change_pct"], steps[-2]["change_pct"]
        if last >= 0 and prev >= 0:
            trend = "growth accelerating" if last > prev else "growth decelerating"
        elif last < 0 and prev < 0:
            trend = "declining faster" if last < prev else "declining more slowly"
        else:
            trend = "reversed to growth" if last >= 0 else "reversed to decline"
        facts[f"{label}_latest_trend"] = trend
    return facts


def _income_facts(annual: list, quarterly: list) -> dict:
    facts = {}
    facts.update(_series_facts(annual, "total_revenue", "annual_revenue"))
    facts.update(_series_facts(annual, "net_income", "annual_net_income"))
    if len(annual) >= 2:
        c, p = annual[0], annual[1]
        for m in ("gross_margin_pct", "operating_margin_pct", "net_margin_pct"):
            if c.get(m) is not None and p.get(m) is not None:
                diff = round(c[m] - p[m], 2)
                facts[f"{m}_latest_vs_prior_year"] = {
                    "latest": c[m], "prior": p[m], "change_pct_points": diff,
                    "direction": "expanded" if diff > 0.1 else ("compressed" if diff < -0.1 else "stable"),
                }
    if len(quarterly) >= 2:
        q0, q1 = quarterly[0], quarterly[1]
        facts["latest_quarter_vs_prior_quarter"] = {
            "latest_quarter": q0["period"], "prior_quarter": q1["period"],
            "revenue_latest": _fmt_big(q0.get("total_revenue")),
            "revenue_prior":  _fmt_big(q1.get("total_revenue")),
            "revenue_change_pct": _chg_pct(q0.get("total_revenue"), q1.get("total_revenue")),
            "revenue_direction":  _direction(q0.get("total_revenue"), q1.get("total_revenue")),
            "net_income_latest": _fmt_big(q0.get("net_income")),
            "net_income_prior":  _fmt_big(q1.get("net_income")),
            "net_income_direction": _direction(q0.get("net_income"), q1.get("net_income")),
            "note": "Sequential quarters can differ because of seasonality; the YoY comparison is more reliable.",
        }
    if len(quarterly) >= 5:
        q0, q4 = quarterly[0], quarterly[4]
        facts["latest_quarter_vs_same_quarter_last_year"] = {
            "latest_quarter": q0["period"], "year_ago_quarter": q4["period"],
            "revenue_change_pct": _chg_pct(q0.get("total_revenue"), q4.get("total_revenue")),
            "revenue_direction":  _direction(q0.get("total_revenue"), q4.get("total_revenue")),
            "net_income_change_pct": _chg_pct(q0.get("net_income"), q4.get("net_income")),
            "net_income_direction":  _direction(q0.get("net_income"), q4.get("net_income")),
        }
    if quarterly and quarterly[0].get("net_margin_pct") is not None and quarterly[0]["net_margin_pct"] > 40:
        facts["warning"] = ("Latest quarter net margin above 40%: this may include one-off items "
                            "(e.g. investment gains or tax effects). Mention this possibility.")
    return facts


def generate_yfinance_income_analysis(ticker: str, income: dict) -> str:
    """Analyse income statement data fetched from yfinance."""
    annual = income.get("annual", [])
    quarterly = income.get("quarterly", [])
    yoy = income.get("yoy_growth", {})

    if not annual:
        return f"⚠️  No income statement data available for {ticker}."

    facts = _income_facts(annual, quarterly)

    system_prompt = """
You are a financial analyst writing an income statement review for a university finance course.
Use ONLY the exact numbers provided. Format large numbers clearly ($2.3B, $450M).
Never invent figures. Write professionally and specifically.
"""
    prompt = f"""
Analyse the income statement for {ticker} using the data below.

ANNUAL DATA (most recent first):
{json.dumps(annual[:4], indent=2, default=str)}

QUARTERLY DATA (most recent first):
{json.dumps(quarterly[:5], indent=2, default=str)}

YEAR-OVER-YEAR GROWTH:
{json.dumps(yoy, indent=2)}

PRE-COMPUTED FACTS (directions and changes are already correct; use them as-is):
{json.dumps(facts, indent=2, default=str)}

Write using EXACTLY these section headers:

**Revenue Trend**
State annual revenue for each year (revenue_fmt) and each change from annual_revenue_changes.
State annual_revenue_latest_trend if present.

**Profitability Analysis**
Use the *_margin_pct_latest_vs_prior_year facts: state each margin and whether it expanded, compressed or was stable.
Explain what this means for the business.

**Quarterly Momentum**
Use latest_quarter_vs_same_quarter_last_year as the main comparison (if present), then
latest_quarter_vs_prior_quarter. Use the direction words exactly as given.
If a warning is present, mention it.

**R&D Investment**
If rd_pct_of_revenue is available, state it and what it suggests about investment in future growth.
If not available, say so in one sentence.

**Key Takeaway**
2-3 sentences summarising the income statement story. Reference at least 3 specific numbers.
"""
    return _call_groq(prompt, system_prompt, max_tokens=1000)


def _balance_facts(annual: list) -> dict:
    facts = {}
    facts.update(_series_facts(annual, "total_assets", "total_assets"))
    facts.update(_series_facts(annual, "total_debt", "total_debt"))
    facts.update(_series_facts(annual, "shareholder_equity", "equity"))
    if annual:
        r = annual[0]
        if r.get("cash") is not None and r.get("total_debt") is not None:
            facts["latest_cash_vs_total_debt"] = {
                "cash": _fmt_big(r["cash"]), "total_debt": _fmt_big(r["total_debt"]),
                "net_cash": _fmt_big(r["net_cash"]),
                "position": "net cash (cash exceeds debt)" if r["net_cash"] >= 0
                            else "net debt (debt exceeds cash)",
                "cash_covers_debt_pct": round(r["cash"] / r["total_debt"] * 100, 1)
                                        if r["total_debt"] else None,
            }
        de = r.get("debt_to_equity_x")
        if de is not None:
            facts["leverage_assessment"] = (
                "low (below 0.5x)" if de < 0.5 else
                "moderate (0.5x-1.5x)" if de < 1.5 else
                "high (above 1.5x); check whether buybacks have shrunk equity")
        elif r.get("shareholder_equity") is not None and r["shareholder_equity"] <= 0:
            facts["leverage_assessment"] = "debt-to-equity not meaningful: shareholder equity is zero or negative"
        if r.get("retained_earnings") is not None and r["retained_earnings"] < 0:
            facts["retained_earnings_note"] = (
                "Retained earnings are negative. For mature companies this is often caused by "
                "large share buybacks and dividends exceeding cumulative profits, not by losses. Check the cash flow statement.")
    return facts


def generate_yfinance_balance_analysis(ticker: str, balance: dict) -> str:
    """Analyse balance sheet data fetched from yfinance."""
    annual = balance.get("annual", [])

    if not annual:
        return f"⚠️  No balance sheet data available for {ticker}."

    facts = _balance_facts(annual)

    system_prompt = """
You are a financial analyst writing a balance sheet review for a university finance course.
Use ONLY the exact numbers provided. Format large numbers clearly ($2.3B, $450M).
Never invent figures. Write professionally and specifically.
"""
    prompt = f"""
Analyse the balance sheet for {ticker} using the data below.

ANNUAL BALANCE SHEET DATA (most recent first):
{json.dumps(annual[:4], indent=2, default=str)}

PRE-COMPUTED FACTS (directions and assessments are already correct; use them as-is):
{json.dumps(facts, indent=2, default=str)}

Write using EXACTLY these section headers:

**Asset Base & Growth**
State total assets per year (assets_fmt) and the changes from total_assets_changes.

**Debt & Leverage**
State total_debt_fmt and debt_to_equity_x (a multiple, e.g. 1.34x) for the most recent year.
Use total_debt_changes for the direction and leverage_assessment for the verdict.

**Equity & Retained Earnings**
State shareholder equity per year (equity_fmt) using equity_changes.
If retained_earnings_note is present, explain it.

**Cash Position & Liquidity**
Use latest_cash_vs_total_debt (net cash vs net debt) and current_ratio_x (above 1.0 = current assets cover current liabilities).

**Key Takeaway**
2-3 sentences on financial strength and stability. Reference at least 3 specific numbers.
"""
    return _call_groq(prompt, system_prompt, max_tokens=900)


def _cashflow_facts(annual: list) -> dict:
    facts = {}
    facts.update(_series_facts(annual, "operating_cf", "operating_cf"))
    facts.update(_series_facts(annual, "free_cash_flow", "fcf"))
    facts.update(_series_facts(annual, "capex", "capex"))
    if annual:
        r = annual[0]
        if r.get("capex_pct_of_operating_cf") is not None:
            facts["latest_capex_pct_of_operating_cf"] = r["capex_pct_of_operating_cf"]
        returned = sum(abs(v) for v in (r.get("share_buybacks"), r.get("dividends_paid")) if v)
        if returned and r.get("free_cash_flow"):
            facts["shareholder_returns_latest_year"] = {
                "buybacks": _fmt_big(abs(r["share_buybacks"])) if r.get("share_buybacks") else "N/A",
                "dividends": _fmt_big(abs(r["dividends_paid"])) if r.get("dividends_paid") else "N/A",
                "total_returned": _fmt_big(returned),
                "pct_of_fcf": round(returned / r["free_cash_flow"] * 100, 1) if r["free_cash_flow"] > 0 else None,
            }
        if r.get("financing_cf") is not None:
            facts["financing_cf_meaning"] = (
                "negative: net cash returned to shareholders/lenders (buybacks, dividends, debt repayment)"
                if r["financing_cf"] < 0 else
                "positive: net cash raised (new debt or equity issuance)")
        if r.get("free_cash_flow") is not None:
            facts["is_cash_generative"] = r["free_cash_flow"] > 0
    return facts


def generate_yfinance_cashflow_analysis(ticker: str, cashflow: dict) -> str:
    """Analyse cash flow statement fetched from yfinance."""
    annual = cashflow.get("annual", [])
    fcf_growth = cashflow.get("fcf_growth_pct")

    if not annual:
        return f"⚠️  No cash flow data available for {ticker}."

    facts = _cashflow_facts(annual)

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

PRE-COMPUTED FACTS (directions are already correct; use them as-is):
{json.dumps(facts, indent=2, default=str)}

Write using EXACTLY these section headers:

**Operating Cash Flow**
State operating cash flow per year (operating_cf_fmt) and the changes from operating_cf_changes.

**Free Cash Flow (FCF)**
State FCF per year (fcf_fmt) and fcf_changes. Explain briefly why FCF (cash left after capex)
matters more than accounting profit. Use is_cash_generative.

**Capital Expenditure**
State capex per year (capex_fmt, shown negative because it is cash spent) and
latest_capex_pct_of_operating_cf. Is the company investing more heavily relative to its cash generation?

**Financing Activities**
Use financing_cf_meaning and shareholder_returns_latest_year (buybacks, dividends, % of FCF).
What does this say about capital allocation priorities?

**Key Takeaway**
2-3 sentences on cash flow quality and sustainability. Reference at least 3 specific numbers.
"""
    return _call_groq(prompt, system_prompt, max_tokens=900)


def _metrics_facts(metrics: dict) -> dict:
    val  = metrics.get("valuation", {}) or {}
    earn = metrics.get("earnings", {}) or {}
    grow = metrics.get("growth", {}) or {}
    prof = metrics.get("profitability", {}) or {}
    fin  = metrics.get("financial_health", {}) or {}
    ana  = metrics.get("analyst", {}) or {}
    facts = {}

    pe = val.get("pe_trailing")
    if pe is not None:
        facts["pe_vs_sp500_typical_20_to_25x"] = (
            "below typical market range" if pe < 20 else
            "within typical market range" if pe <= 25 else "above typical market range")
    pf = val.get("pe_forward")
    if pe is not None and pf is not None:
        facts["forward_vs_trailing_pe"] = (
            "forward P/E is lower: analysts expect earnings to grow" if pf < pe else
            "forward P/E is higher: analysts expect earnings to decline")
    eg, rg = earn.get("eps_growth_pct"), grow.get("revenue_yoy_pct")
    if eg is not None and rg is not None:
        facts["expected_eps_growth_vs_recent_revenue_growth"] = (
            f"expected EPS growth ({eg}%) is HIGHER than recent revenue growth ({rg}%)" if eg > rg else
            f"expected EPS growth ({eg}%) is LOWER than recent revenue growth ({rg}%)" if eg < rg else
            f"expected EPS growth equals recent revenue growth ({eg}%)")
    fy = prof.get("fcf_yield_pct")
    if fy is not None:
        facts["fcf_yield_vs_3pct"] = "above 3% (strong)" if fy > 3 else "below 3% (modest relative to market value)"
    de = fin.get("debt_to_equity_x")
    if de is not None:
        facts["leverage_assessment"] = (
            "low (below 0.5x)" if de < 0.5 else "moderate (0.5x-1.5x)" if de < 1.5 else "high (above 1.5x)")
    cr = fin.get("current_ratio_x")
    if cr is not None:
        facts["liquidity_assessment"] = (
            "current ratio below 1.0x: current liabilities exceed current assets" if cr < 1 else
            "current ratio at/above 1.0x: current assets cover current liabilities")
    beta = val.get("beta")
    if beta is not None:
        facts["beta_interpretation"] = (
            "lower volatility than the market (beta < 0.8)" if beta < 0.8 else
            "roughly market-like volatility (beta 0.8-1.2)" if beta <= 1.2 else
            "higher volatility than the market (beta > 1.2)")
    up = ana.get("upside_pct")
    if up is not None:
        facts["analyst_target_implies"] = (
            f"{up}% upside to the mean target" if up > 0 else f"{abs(up)}% downside to the mean target")
    return facts


def generate_key_metrics_dashboard(ticker: str, metrics: dict) -> str:
    """
    Generate a key metrics dashboard analysis including EPS growth,
    revenue growth, FCF yield, valuation ratios, and analyst consensus.
    """
    if not metrics:
        return f"⚠️  No key metrics data available for {ticker}."

    facts = _metrics_facts(metrics)

    system_prompt = """
You are an equity analyst writing a metrics dashboard summary for a university finance report.
Use ONLY the exact numbers provided. Be specific and analytical.
"""
    prompt = f"""
Write a key metrics dashboard for {ticker} using the data below.

METRICS DATA:
{json.dumps(metrics, indent=2, default=str)}

PRE-COMPUTED ASSESSMENTS (already correct; use them as-is and do not contradict them):
{json.dumps(facts, indent=2)}

Write using EXACTLY these section headers:

**Valuation Snapshot**
State P/E (trailing and forward), P/B, P/S, and PEG ratio.
Use pe_vs_sp500_typical_20_to_25x and forward_vs_trailing_pe. Note that a high P/B can reflect
buybacks shrinking book equity rather than overvaluation.

**Growth Profile**
State EPS TTM and forward EPS. Describe eps_growth_pct as analyst-EXPECTED growth (see eps_growth_basis).
State revenue_yoy_pct (see revenue_growth_basis). Use expected_eps_growth_vs_recent_revenue_growth exactly.

**Profitability & Cash Generation**
State gross, operating and net margin %, ROE % and ROA %.
State FCF yield % with fcf_yield_vs_3pct. State dividend_yield_pct if available.

**Financial Health**
State current_ratio_x and debt_to_equity_x (multiples, e.g. 0.84x) with liquidity_assessment and leverage_assessment.
State beta (Yahoo 5-year monthly beta) with beta_interpretation.

**Analyst Consensus**
State the recommendation, target price and analyst_target_implies, plus the number of analysts.
Note that analyst targets are opinions, not forecasts of what will happen.
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

SENTIMENT SUMMARY (simple headline keyword tags, not a model score): {pos} Positive | {neg} Negative | {neu} Neutral or Mixed

HEADLINES:
{headlines}

Write using EXACTLY these section headers:

**News Overview**
Summarise the overall news tone ({pos} positive, {neg} negative, {neu} neutral/mixed). Note these tags come from simple keyword matching.
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