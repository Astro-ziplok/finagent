"""
src/visualization/charts.py — Visualization Module
====================================================
Generates all required charts and saves them to organised subfolders.

Folder structure:
  reports/charts/AAPL/chart1_price_volume.png
  reports/charts/AAPL/chart4_bollinger_bands.png
  reports/charts/chart2_correlation_heatmap.png   ← cross-ticker charts at root
  reports/charts/chart3_return_distributions.png
  reports/charts/chart5_comparative_returns.png

The four required chart types:
  Chart 1 — Price Trend Line Chart with Volume Overlay   ← REQUIRED
  Chart 2 — Correlation Heatmap across assets            ← REQUIRED
  Chart 3 — Distribution Plot of daily returns           ← REQUIRED
  Chart 4 — Rolling Statistics (Bollinger Bands)         ← REQUIRED
  Chart 5 — Normalised Comparative Returns               ← BONUS
"""

import logging
import matplotlib
matplotlib.use("Agg")
# Suppress matplotlib's verbose categorical unit warnings
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (REPORTS_CHARTS_DIR, CHART_DPI, CHART_STYLE,
                    CHART_FIGSIZE, ROLLING_SHORT, ROLLING_LONG)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    plt.style.use(CHART_STYLE)
except:
    plt.style.use("ggplot")


def _compare_dir(tickers: list) -> Path:
    """Return (and create) a named comparison subfolder e.g. reports/charts/compare_AAPL_MSFT/"""
    name = "compare_" + "_".join(sorted(tickers))
    d = REPORTS_CHARTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ticker_dir(ticker: str) -> Path:
    """Return (and create) the per-ticker chart subfolder."""
    d = REPORTS_CHARTS_DIR / ticker
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 1: Price Trend + Volume
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_and_volume(df: pd.DataFrame, ticker: str) -> str:
    # Extract year range from data
    start_year = df.index.min().year
    end_year   = df.index.max().year
    year_label = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax1.plot(df.index, df["Close"], label="Close Price", color="#1f77b4", linewidth=1.5)
    ax1.plot(df.index, df[f"MA_{ROLLING_SHORT}"],  label=f"{ROLLING_SHORT}-Day MA", color="#ff7f0e", linewidth=1.2, linestyle="--")
    ax1.plot(df.index, df[f"MA_{ROLLING_LONG}"], label=f"{ROLLING_LONG}-Day MA",  color="#2ca02c", linewidth=1.2, linestyle="--")

    outliers = df[df["Is_Outlier"] == True]
    if not outliers.empty:
        ax1.scatter(outliers.index, outliers["Close"], color="red", s=40, zorder=5, label="Flagged Outlier")

    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.set_title(f"{ticker} — Price Trend & Volume ({year_label})", fontsize=14, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Fix 2: Convert volume to millions for readable axis labels
    volume_millions = df["Volume"] / 1_000_000
    ax2.bar(df.index, volume_millions, color="#1f77b4", alpha=0.5, width=1)
    ax2.set_ylabel("Volume (Millions)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(True, alpha=0.3)
    # Format y-axis ticks as e.g. "100M", "200M"
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}M")
    )

    plt.tight_layout()
    path = _ticker_dir(ticker) / "chart1_price_volume.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Chart 1 saved: {ticker}/chart1_price_volume.png")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 2: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(cleaned_data: dict) -> str:
    close_prices = pd.DataFrame({t: df["Close"] for t, df in cleaned_data.items()})
    corr_matrix  = close_prices.corr()

    # Extract year range from the data
    all_indices = [df.index for df in cleaned_data.values()]
    start_year  = min(idx.min().year for idx in all_indices)
    end_year    = max(idx.max().year for idx in all_indices)
    year_label  = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, ax=ax, linewidths=0.5, square=True,
        annot_kws={"size": 11}
    )
    ax.set_title(
        f"Stock Price Correlation Heatmap ({year_label})\n"
        "(1.0 = perfect positive correlation,  "
        "0 = no correlation,  "
        "-1 = perfect negative correlation)",
        fontsize=12, fontweight="bold", pad=15
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = REPORTS_CHARTS_DIR / "chart2_correlation_heatmap.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Chart 2 saved: chart2_correlation_heatmap.png")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 3: Return Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_return_distributions(cleaned_data: dict) -> str:
    tickers = list(cleaned_data.keys())
    cols = 2
    rows = (len(tickers) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()
    colors = plt.cm.tab10.colors

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        df = cleaned_data[ticker]
        returns = df["Daily_Return"].dropna()

        # Fix 4: Year range for each subplot
        start_year = df.index.min().year
        end_year   = df.index.max().year
        year_label = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"

        ax.hist(returns, bins=50, density=True, alpha=0.4, color=colors[i])
        # Fix 5: Proper legend label instead of raw variable name
        returns.plot.kde(ax=ax, color=colors[i], linewidth=2, label="Daily Return (%)")
        mean_r = returns.mean()
        ax.axvline(mean_r, color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean Return: {mean_r:.2f}%")
        ax.set_title(f"{ticker} — Daily Return Distribution ({year_label})", fontweight="bold")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Overall title with year range
    all_indices = [df.index for df in cleaned_data.values()]
    overall_start = min(idx.min().year for idx in all_indices)
    overall_end   = max(idx.max().year for idx in all_indices)
    overall_label = f"{overall_start}" if overall_start == overall_end else f"{overall_start}–{overall_end}"
    plt.suptitle(f"Distribution of Daily Returns ({overall_label})",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = REPORTS_CHARTS_DIR / "chart3_return_distributions.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Chart 3 saved: chart3_return_distributions.png")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 4: Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

def plot_bollinger_bands(df: pd.DataFrame, ticker: str) -> str:
    start_year = df.index.min().year
    end_year   = df.index.max().year
    year_label = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)

    ax.plot(df.index, df["Close"],    label="Close Price",    color="#1f77b4", linewidth=1.5)
    ax.plot(df.index, df["BB_Mid"],   label="20-Day MA (Mid)",color="orange",  linewidth=1.2, linestyle="--")
    ax.plot(df.index, df["BB_Upper"], label="Upper Band (+2σ)",color="#d62728", linewidth=1.0, linestyle=":")
    ax.plot(df.index, df["BB_Lower"], label="Lower Band (-2σ)",color="#2ca02c", linewidth=1.0, linestyle=":")
    ax.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], alpha=0.08, color="grey")

    ax.set_title(f"{ticker} — Bollinger Bands (20-Day, {year_label})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Start x-axis where BB data begins (after 20-day warmup period)
    first_valid = df["BB_Upper"].first_valid_index()
    if first_valid:
        ax.set_xlim(first_valid, df.index[-1])

    plt.tight_layout()

    path = _ticker_dir(ticker) / "chart4_bollinger_bands.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Chart 4 saved: {ticker}/chart4_bollinger_bands.png")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 5 (Bonus): Normalised Comparative Returns
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparative_returns(cleaned_data: dict) -> str:
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    colors = plt.cm.tab10.colors

    all_indices = [df.index for df in cleaned_data.values()]
    start_year  = min(idx.min().year for idx in all_indices)
    end_year    = max(idx.max().year for idx in all_indices)
    year_label  = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"

    for i, (ticker, df) in enumerate(cleaned_data.items()):
        normalised = (df["Close"] / df["Close"].iloc[0]) * 100
        ax.plot(df.index, normalised, label=ticker, color=colors[i], linewidth=1.5)

    ax.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Baseline (100)")
    ax.set_title(f"Normalised Comparative Returns ({year_label}, Base = 100)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Indexed Return (Start = 100)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = REPORTS_CHARTS_DIR / "chart5_comparative_returns.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Chart 5 saved: chart5_comparative_returns.png")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 6 (Bonus): Fundamental Ratios Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_fundamental_ratios(fundamental_data: dict) -> str:
    """
    Bar chart comparing key ratios (P/E, ROE, ROA) across tickers.
    Only runs if Alpha Vantage fundamental data is available.
    """
    rows = []
    for ticker, data in fundamental_data.items():
        if ticker == "commodities" or "overview" not in data:
            continue
        ov = data["overview"]
        if not ov:
            continue
        rows.append({
            "ticker": ticker,
            "P/E Ratio": ov.get("pe_ratio") or 0,
            "ROE (%)":   round((ov.get("roe") or 0) * 100, 2),
            "ROA (%)":   round((ov.get("roa") or 0) * 100, 2),
            "D/E Ratio": ov.get("debt_to_equity") or 0,
        })

    if not rows:
        return ""

    df = pd.DataFrame(rows).set_index("ticker")
    metrics = [c for c in df.columns if df[c].sum() != 0]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(df.index, df[metric], color=colors[:len(df)], alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title(metric, fontweight="bold", fontsize=12)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on top of each bar
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * abs(h),
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Fundamental Ratios Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = REPORTS_CHARTS_DIR / "chart6_fundamental_ratios.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Chart 6 saved: chart6_fundamental_ratios.png")
    return str(path)




# ─────────────────────────────────────────────────────────────────────────────
#  CHART 8 (Bonus): Candlestick Chart — Last 90 Days
# ─────────────────────────────────────────────────────────────────────────────

def plot_candlestick(df: pd.DataFrame, ticker: str, days: int = 90) -> str:
    """
    Generate a professional candlestick chart for the last `days` trading days.

    Uses mplfinance — the standard Python library for OHLC/candlestick charts.
    Green candles = price closed higher than open (bullish day)
    Red candles   = price closed lower than open (bearish day)
    Wicks show the intraday high and low range.

    Overlaid with a 20-day moving average for trend context.
    Volume bars shown in the lower panel.

    Args:
        df:     Cleaned DataFrame with OHLCV columns and DatetimeIndex
        ticker: Stock symbol for title and filename
        days:   Number of most recent trading days to display (default: 90)

    Returns:
        Path to the saved chart file
    """
    try:
        import mplfinance as mpf
    except ImportError:
        logger.warning("  mplfinance not installed. Run: pip install mplfinance")
        logger.warning("  Skipping candlestick chart.")
        return ""

    # Use only the last `days` rows for readability
    plot_df = df.tail(days).copy()

    # Ensure required OHLCV columns exist
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in plot_df.columns]
    if missing:
        logger.warning(f"  Missing columns for candlestick: {missing}")
        return ""

    # Keep only OHLCV — mplfinance is strict about extra columns
    plot_df = plot_df[required].copy()

    # Ensure index is DatetimeIndex with no timezone
    plot_df.index = pd.to_datetime(plot_df.index)
    if hasattr(plot_df.index, "tz") and plot_df.index.tz is not None:
        plot_df.index = plot_df.index.tz_localize(None)

    # Year range for title
    start_year = plot_df.index.min().year
    end_year   = plot_df.index.max().year
    year_label = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"
    date_range = f"{plot_df.index.min().strftime('%b %Y')} – {plot_df.index.max().strftime('%b %Y')}"

    # 20-day MA overlay (calculated on the full dataset, then trimmed)
    ma20 = df["Close"].rolling(20).mean().reindex(plot_df.index)

    # Build mplfinance additional plot for the MA line
    ap = [mpf.make_addplot(ma20, color="orange", width=1.5,
                           linestyle="--", label="20-Day MA")]

    # Professional colour style
    mc = mpf.make_marketcolors(
        up="green", down="red",
        edge="inherit",
        wick={"up": "green", "down": "red"},
        volume={"up": "green", "down": "red"},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle="--",
        gridcolor="lightgrey",
        facecolor="#f0f0f8",
        figcolor="#f0f0f8",
        y_on_right=False,
    )

    # Output path
    path = _ticker_dir(ticker) / "chart8_candlestick.png"

    # Plot and save
    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        title=f"\n{ticker} — Candlestick Chart ({date_range})\n"
              f"Last {days} Trading Days  |  Green = Up Day  |  Red = Down Day",
        ylabel="Price (USD)",
        ylabel_lower="Volume (M)",
        volume=True,
        addplot=ap,
        figsize=(16, 9),
        tight_layout=True,
        returnfig=True,
        volume_panel=1,
        panel_ratios=(3, 1),
        datetime_format="%b %Y",
    )

    # Format volume axis in millions
    vol_ax = axes[2] if len(axes) > 2 else axes[1]
    vol_ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight",
                facecolor="#f0f0f8")
    plt.close(fig)

    logger.info(f"  ✓ Chart 8 saved: {ticker}/chart8_candlestick.png")
    return str(path)




def plot_comparison_normalised_price(cleaned_data: dict, cdir: "Path" = None) -> str:
    """
    Chart C1: Normalised price comparison — all stocks indexed to 100 at start.
    Shows true relative performance regardless of absolute price differences.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10.colors

    for i, (ticker, df) in enumerate(cleaned_data.items()):
        norm = (df["Close"] / df["Close"].iloc[0]) * 100
        ax.plot(df.index, norm, label=ticker, color=colors[i],
                linewidth=2.0, alpha=0.9)
        # Annotate final value
        final_val = norm.iloc[-1]
        ax.annotate(f"{ticker}: {final_val:.1f}",
                    xy=(df.index[-1], final_val),
                    xytext=(8, 0), textcoords="offset points",
                    color=colors[i], fontsize=9, fontweight="bold")

    ax.axhline(100, color="black", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Baseline (100)")
    ax.fill_between(ax.get_xlim(), 100, ax.get_ylim()[0],
                    alpha=0.03, color="red")
    ax.fill_between(ax.get_xlim(), 100, ax.get_ylim()[1],
                    alpha=0.03, color="green")

    ax.set_title("Normalised Price Comparison (Base = 100 at Start)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Indexed Price (Start = 100)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    folder = cdir or REPORTS_CHARTS_DIR
    path = folder / "C1_normalised_price.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ C1 saved: {path.parent.name}/C1_normalised_price.png")
    return str(path)


def plot_comparison_financial_bars(yf_data: dict, cdir: "Path" = None) -> str:
    """
    Chart C2: Side-by-side bar chart comparing key financials:
    Revenue, Net Income, and Free Cash Flow across tickers.
    Uses yfinance data (free, no quota).
    """
    rows = []
    for ticker, data in yf_data.items():
        if not isinstance(data, dict):
            continue
        income   = data.get("income", {})
        cashflow = data.get("cashflow", {})
        metrics  = data.get("metrics", {})

        # Most recent annual figures
        annual_income = income.get("annual", [{}])[0] if income.get("annual") else {}
        annual_cf     = cashflow.get("annual", [{}])[0] if cashflow.get("annual") else {}

        rev = annual_income.get("total_revenue")
        ni  = annual_income.get("net_income")
        fcf = annual_cf.get("free_cash_flow")
        pe  = (metrics.get("valuation") or {}).get("pe_trailing")

        if rev or ni or fcf:
            rows.append({
                "ticker":          ticker,
                "Revenue ($B)":    round(rev / 1e9, 2) if rev else 0,
                "Net Income ($B)": round(ni  / 1e9, 2) if ni  else 0,
                "FCF ($B)":        round(fcf / 1e9, 2) if fcf else 0,
            })

    if not rows:
        logger.warning("  No financial data available for comparison bar chart.")
        return ""

    df = pd.DataFrame(rows).set_index("ticker")
    metrics_to_plot = [c for c in df.columns if df[c].abs().sum() > 0]

    if not metrics_to_plot:
        return ""

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    tickers = df.index.tolist()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        bars = ax.bar(tickers, df[metric],
                      color=colors[:len(tickers)], alpha=0.85,
                      edgecolor="black", linewidth=0.5)

        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            sign = "+" if h > 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + abs(h) * 0.02,
                    f"{sign}{h:.2f}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    plt.suptitle("Financial Comparison (Most Recent Annual)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    folder = cdir or REPORTS_CHARTS_DIR
    path = folder / "C2_financial_bars.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ C2 saved: {path.parent.name}/C2_financial_bars.png")
    return str(path)


def plot_comparison_valuation_radar(yf_data: dict, cdir: "Path" = None) -> str:
    """
    Chart C3: Side-by-side P/E, EPS growth, revenue growth, FCF yield, margin bars.
    Gives a quick snapshot of which company is cheaper / growing faster.
    """
    rows = []
    for ticker, data in yf_data.items():
        if not isinstance(data, dict):
            continue
        m  = data.get("metrics", {}) or {}
        val  = m.get("valuation", {}) or {}
        earn = m.get("earnings", {}) or {}
        grow = m.get("growth", {}) or {}
        prof = m.get("profitability", {}) or {}

        rows.append({
            "ticker":            ticker,
            "P/E (trailing)":    val.get("pe_trailing") or 0,
            "EPS Growth (%)":    earn.get("eps_growth_pct") or 0,
            "Revenue Growth (%)":grow.get("revenue_yoy_pct") or 0,
            "Net Margin (%)":    prof.get("profit_margin_pct") or 0,
            "FCF Yield (%)":     prof.get("fcf_yield_pct") or 0,
        })

    if not rows or len(rows) < 2:
        logger.warning("  Need 2+ tickers with metrics for valuation comparison chart.")
        return ""

    df = pd.DataFrame(rows).set_index("ticker")
    metrics = list(df.columns)
    colors  = plt.cm.tab10.colors
    tickers = df.index.tolist()
    n       = len(metrics)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = df[metric].tolist()
        bars = ax.bar(tickers, vals,
                      color=colors[:len(tickers)], alpha=0.85,
                      edgecolor="black", linewidth=0.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + abs(v) * 0.03 if v >= 0 else v - abs(v) * 0.08,
                    f"{v:.1f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9, fontweight="bold")

        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    plt.suptitle("Valuation & Growth Metrics Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    folder = cdir or REPORTS_CHARTS_DIR
    path = folder / "C3_valuation_metrics.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ C3 saved: {path.parent.name}/C3_valuation_metrics.png")
    return str(path)


def plot_comparison_correlation_heatmap(cleaned_data: dict, cdir: "Path" = None) -> str:
    """
    Chart C4: Correlation heatmap of daily returns (not prices) between tickers.
    Return correlation is more meaningful than price correlation for comparisons.
    """
    returns = pd.DataFrame({
        ticker: df["Daily_Return"].dropna()
        for ticker, df in cleaned_data.items()
    })
    corr = returns.corr()

    # Extract year range from the data
    all_indices = [df.index for df in cleaned_data.values()]
    start_year  = min(idx.min().year for idx in all_indices)
    end_year    = max(idx.max().year for idx in all_indices)
    year_label  = f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"

    fig, ax = plt.subplots(figsize=(max(6, len(cleaned_data) * 2),
                                    max(5, len(cleaned_data) * 1.8)))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, square=True,
        annot_kws={"size": 12, "weight": "bold"},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(
        f"Daily Return Correlation Heatmap ({year_label})\n"
        "(1.0 = perfect positive correlation,  "
        "0 = no correlation,  "
        "-1 = perfect negative correlation)",
        fontsize=12, fontweight="bold", pad=15
    )
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    folder = cdir or REPORTS_CHARTS_DIR
    path = folder / "C4_return_correlation.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ C4 saved: {path.parent.name}/C4_return_correlation.png")
    return str(path)


def plot_financial_dashboard(ticker: str, yf_ticker_data: dict) -> str:
    """
    Fix 1: Year-by-year financial dashboard for a single company.
    A 2×2 grid combining:
      - Top left:  Revenue & Net Income bar chart (annual)
      - Top right: Margin trends line chart (gross, operating, net %)
      - Bottom left: Free Cash Flow bar chart (annual)
      - Bottom right: EPS trend line chart (annual)
    Uses yfinance data — completely free.
    """
    income   = yf_ticker_data.get("income", {})
    cashflow = yf_ticker_data.get("cashflow", {})
    metrics  = yf_ticker_data.get("metrics", {})

    annual_income = income.get("annual", [])
    annual_cf     = cashflow.get("annual", [])

    if not annual_income:
        logger.warning(f"  No annual income data for {ticker} financial dashboard.")
        return ""

    # Reverse so oldest year is first (left on chart)
    annual_income = list(reversed(annual_income))
    annual_cf     = list(reversed(annual_cf)) if annual_cf else []

    # Extract data — years as plain strings for labels only
    years   = [str(r["period"][:4]) for r in annual_income]
    revenue = [round((r.get("total_revenue") or 0) / 1e9, 2) for r in annual_income]
    net_inc = [round((r.get("net_income") or 0) / 1e9, 2) for r in annual_income]
    g_margin= [r.get("gross_margin_pct") or 0 for r in annual_income]
    o_margin= [r.get("operating_margin_pct") or 0 for r in annual_income]
    n_margin= [r.get("net_margin_pct") or 0 for r in annual_income]
    fcf     = [round((r.get("free_cash_flow") or 0) / 1e9, 2) for r in annual_cf] if annual_cf else []
    fcf_yrs = [str(r["period"][:4]) for r in annual_cf] if annual_cf else []

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{ticker} — Financial Dashboard (Year-by-Year)",
                 fontsize=16, fontweight="bold", y=1.01)

    colors = plt.cm.tab10.colors
    # Use integer positions to avoid matplotlib categorical unit warnings
    x     = list(range(len(years)))
    x_fcf = list(range(len(fcf_yrs)))

    # ── Top Left: Revenue & Net Income ───────────────────────────────────────
    ax = axes[0, 0]
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], revenue, width,
                   label="Revenue", color=colors[0], alpha=0.85, edgecolor="black", lw=0.5)
    bars2 = ax.bar([i + width/2 for i in x], net_inc, width,
                   label="Net Income", color=colors[1], alpha=0.85, edgecolor="black", lw=0.5)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h != 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + abs(h)*0.02,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_title("Revenue & Net Income ($B)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel("USD Billions")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.8)

    # ── Top Right: Margin Trends ──────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, g_margin, "o-", label="Gross Margin %", color=colors[2], lw=2, ms=6)
    ax.plot(x, o_margin, "s-", label="Operating Margin %", color=colors[3], lw=2, ms=6)
    ax.plot(x, n_margin, "^-", label="Net Margin %", color=colors[4], lw=2, ms=6)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    for i, (g, o, n) in enumerate(zip(g_margin, o_margin, n_margin)):
        ax.annotate(f"{g:.1f}%", (i, g), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color=colors[2])
        ax.annotate(f"{n:.1f}%", (i, n), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color=colors[4])
    ax.set_title("Margin Trends (%)", fontweight="bold")
    ax.set_ylabel("Margin (%)")
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Bottom Left: Free Cash Flow ───────────────────────────────────────────
    ax = axes[1, 0]
    if fcf and fcf_yrs:
        bar_colors = [colors[0] if v >= 0 else "#d62728" for v in fcf]
        bars = ax.bar(x_fcf, fcf, color=bar_colors, alpha=0.85,
                      edgecolor="black", lw=0.5)
        for bar in bars:
            h = bar.get_height()
            if h != 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + abs(h)*0.02 if h >= 0 else h - abs(h)*0.06,
                        f"{h:.1f}", ha="center",
                        va="bottom" if h >= 0 else "top",
                        fontsize=8, fontweight="bold")
        ax.set_xticks(x_fcf); ax.set_xticklabels(fcf_yrs)
        ax.axhline(0, color="black", lw=0.8)
    else:
        ax.text(0.5, 0.5, "No cash flow data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
    ax.set_title("Free Cash Flow ($B)", fontweight="bold")
    ax.set_ylabel("USD Billions")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Bottom Right: EPS Trend ───────────────────────────────────────────────
    ax = axes[1, 1]
    eps_ttm     = (metrics.get("earnings") or {}).get("eps_ttm")
    eps_forward = (metrics.get("earnings") or {}).get("eps_forward")

    if eps_ttm is not None:
        eps_vals  = [eps_ttm, eps_forward] if eps_forward else [eps_ttm]
        eps_lbls  = ["TTM", "Forward"] if eps_forward else ["TTM"]
        eps_x     = list(range(len(eps_vals)))
        bar_cols  = [colors[0] if v >= 0 else "#d62728" for v in eps_vals]
        bars = ax.bar(eps_x, eps_vals, color=bar_cols,
                      alpha=0.85, edgecolor="black", lw=0.5)
        for bar in bars:
            h = bar.get_height()
            if h is not None and h != 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + abs(h)*0.02 if h >= 0 else h - abs(h)*0.06,
                        f"${h:.2f}", ha="center",
                        va="bottom" if h >= 0 else "top",
                        fontsize=11, fontweight="bold")
        ax.set_xticks(eps_x); ax.set_xticklabels(eps_lbls)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("EPS (TTM vs Forward)", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No EPS data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title("EPS", fontweight="bold")
    ax.set_ylabel("USD per Share")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = _ticker_dir(ticker) / "chart7_financial_dashboard.png"
    plt.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Financial dashboard saved: {ticker}/chart7_financial_dashboard.png")
    return str(path)


def generate_comparison_charts(cleaned_data: dict,
                                yf_data: dict = None) -> list:
    """
    Generate all 4 comparison charts for multi-ticker compare queries.
    Charts are saved to reports/charts/compare_AAPL_MSFT/ (named after tickers).

    Charts generated:
      C1 — Normalised price comparison (indexed to 100)
      C2 — Side-by-side financial bars (Revenue, Net Income, FCF)
      C3 — Valuation & growth metrics comparison (P/E, EPS growth, margin)
      C4 — Daily return correlation heatmap
    """
    if len(cleaned_data) < 2:
        logger.warning("  Comparison charts require at least 2 tickers.")
        return []

    tickers = list(cleaned_data.keys())
    cdir    = _compare_dir(tickers)
    logger.info(f"  Generating comparison charts → {cdir.name}/")
    saved = []

    saved.append(plot_comparison_normalised_price(cleaned_data, cdir))
    saved.append(plot_comparison_correlation_heatmap(cleaned_data, cdir))

    if yf_data:
        path = plot_comparison_financial_bars(yf_data, cdir)
        if path:
            saved.append(path)
        path = plot_comparison_valuation_radar(yf_data, cdir)
        if path:
            saved.append(path)

    saved = [p for p in saved if p]
    logger.info(f"  ✓ {len(saved)} comparison chart(s) saved to {cdir.name}/")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
#  Master function
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_charts(cleaned_data: dict, fundamental_data: dict = None) -> list:
    """Generate all charts and save to organised subfolders."""
    logger.info("=" * 60)
    logger.info("STEP 3: VISUALIZATION")
    logger.info("=" * 60)

    saved = []

    for ticker, df in cleaned_data.items():
        saved.append(plot_price_and_volume(df, ticker))
        saved.append(plot_bollinger_bands(df, ticker))
        # Bonus: candlestick chart (last 90 days)
        path = plot_candlestick(df, ticker)
        if path:
            saved.append(path)

    saved.append(plot_correlation_heatmap(cleaned_data))
    saved.append(plot_return_distributions(cleaned_data))
    saved.append(plot_comparative_returns(cleaned_data))

    if fundamental_data:
        av_data = fundamental_data.get("av", {})
        if av_data and isinstance(av_data, dict):
            path = plot_fundamental_ratios(av_data)
            if path:
                saved.append(path)

    logger.info(f"\nVisualization complete. {len(saved)} chart(s) saved under {REPORTS_CHARTS_DIR}")
    return saved


if __name__ == "__main__":
    from config import DATA_PROC_DIR, TICKERS
    cleaned_data = {}
    for ticker in TICKERS:
        path = DATA_PROC_DIR / f"{ticker}_processed.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            cleaned_data[ticker] = df
    if cleaned_data:
        generate_all_charts(cleaned_data)