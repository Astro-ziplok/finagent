"""
main.py — FinAgent Fixed Pipeline Entry Point
=============================================
Runs a focused batch analysis on the default tickers in config.py.
Fast (~2 minutes), covering the 3 core rubric requirements:
  - Trend summary per ticker
  - Risk commentary (all tickers)
  - Comparative analysis (all tickers)

For deeper analysis (income statements, balance sheets, cash flow,
news sentiment, sector rotation), use the interactive agent instead:
    python agent.py

Usage:
    python main.py
"""

import logging
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import TICKERS, REPORTS_ANALYSIS_DIR, REPORTS_CHARTS_DIR, REPORTS_DIR
from src.collection.fetch_data import collect_all_data
from src.cleaning.clean_data import clean_all_stocks, get_summary_stats
from src.visualization.charts import generate_all_charts
from src.analysis.ai_analysis import (
    _build_data_context,
    generate_trend_summary,
    generate_anomaly_commentary,
    generate_risk_commentary,
    generate_comparative_analysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _wrap(text: str, width: int = 100) -> str:
    """Wrap long paragraph lines at `width` characters."""
    lines = []
    for line in text.split("\n"):
        if len(line) <= width or line.startswith(("──", "==", "**")):
            lines.append(line)
        else:
            lines.append(textwrap.fill(line.strip(), width=width,
                                       subsequent_indent="  "))
    return "\n".join(lines)


def main():
    print("\n" + "=" * 65)
    print("  🚀 FINAGENT — Batch Pipeline")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print("  Analysis: Trend | Anomaly | Risk | Comparison")
    print("=" * 65 + "\n")

    # ── Step 1: Data Collection ───────────────────────────────────────────────
    print("STEP 1: DATA COLLECTION")
    print("-" * 40)
    stock_data, _ = collect_all_data()

    if not stock_data:
        logger.error("No stock data collected. Check internet connection or VPN.")
        sys.exit(1)

    # ── Step 2: Clean & Process ───────────────────────────────────────────────
    print("\nSTEP 2: DATA CLEANING & PROCESSING")
    print("-" * 40)
    cleaned_data = clean_all_stocks(stock_data)

    summary = get_summary_stats(cleaned_data)
    print("\n📊 Summary Statistics:")
    print(summary.to_string())

    # Save summary stats
    snapshot_path = REPORTS_DIR / "data_snapshots" / "summary_stats.csv"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(snapshot_path)
    print(f"\n💾 Summary stats → data_snapshots/summary_stats.csv")

    # ── Step 3: Visualizations ────────────────────────────────────────────────
    print("\nSTEP 3: VISUALIZATION")
    print("-" * 40)
    chart_paths = generate_all_charts(cleaned_data)
    print(f"\n✅ {len(chart_paths)} chart(s) saved to {REPORTS_CHARTS_DIR}/")

    # ── Step 4: AI Analysis ───────────────────────────────────────────────────
    print("\nSTEP 4: AI ANALYSIS")
    print("-" * 40)

    context  = _build_data_context(cleaned_data)
    sections = []

    # Report header
    sections.append("=" * 70)
    sections.append("FINAGENT — AI ANALYSIS REPORT")
    sections.append(f"Tickers analysed: {', '.join(cleaned_data.keys())}")
    sections.append("=" * 70)

    # Trend summary + anomaly per ticker
    sections.append("\n\n── TREND SUMMARIES ──────────────────────────────────────────────────\n")
    for ticker in cleaned_data:
        print(f"  Generating trend summary for {ticker}...")
        sections.append(f"[ {ticker} ]")
        sections.append(generate_trend_summary(ticker, context))
        sections.append("")

    sections.append("\n── ANOMALY COMMENTARY ───────────────────────────────────────────────\n")
    for ticker in cleaned_data:
        print(f"  Generating anomaly commentary for {ticker}...")
        sections.append(f"[ {ticker} ]")
        sections.append(generate_anomaly_commentary(ticker, context))
        sections.append("")

    # Risk commentary (all tickers together)
    sections.append("\n── RISK COMMENTARY ──────────────────────────────────────────────────\n")
    print("  Generating risk commentary...")
    sections.append(generate_risk_commentary(context))

    # Comparative analysis (all tickers together)
    sections.append("\n\n── COMPARATIVE ANALYSIS ─────────────────────────────────────────────\n")
    print("  Generating comparative analysis...")
    sections.append(generate_comparative_analysis(context))

    # ── Step 5: Save Report ───────────────────────────────────────────────────
    print("\nSTEP 5: SAVING REPORT")
    print("-" * 40)
    full_report = _wrap("\n".join(sections))
    report_path = REPORTS_ANALYSIS_DIR / "ai_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"💾 Report saved → {report_path}")

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("🤖 AI ANALYSIS PREVIEW (first ticker)")
    print("=" * 65)
    first = list(cleaned_data.keys())[0]
    trend = generate_trend_summary(first, context)
    # Print first 20 lines of the trend summary as preview
    preview_lines = trend.split("\n")[:20]
    print("\n".join(preview_lines))
    if len(trend.split("\n")) > 20:
        print("  ... (see full report in reports/analysis/)")

    print("\n" + "=" * 65)
    print("✅ Pipeline complete!")
    print(f"   Charts:     {REPORTS_CHARTS_DIR}/")
    print(f"   AI Report:  {report_path}")
    print(f"   Stats:      data_snapshots/summary_stats.csv")
    print("\n💡 For deeper analysis, run: python agent.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()