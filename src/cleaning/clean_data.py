"""
src/cleaning/clean_data.py — Data Cleaning & Processing Module
==============================================================
Responsibility: Take raw stock DataFrames and return clean, feature-rich ones.

This module handles all the data quality issues listed in the rubric:
  ✓ Missing values       → forward-fill with logging
  ✓ Duplicate records    → detect and remove
  ✓ Data type normalisation → enforce float types, proper DatetimeIndex
  ✓ Outlier detection    → flag extreme daily return values (e.g. data errors)
  ✓ Feature engineering  → daily returns, 7-day MA, 30-day MA, Bollinger Bands, volatility
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (DATA_PROC_DIR, ROLLING_SHORT, ROLLING_LONG,
                    BOLLINGER_WINDOW, BOLLINGER_STD)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_stock_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean and enrich a single stock DataFrame.

    Steps performed:
      1. Sort by date (ascending)
      2. Remove duplicate dates
      3. Handle missing values (forward-fill, then drop any remaining)
      4. Normalise numeric column types to float64
      5. Detect and flag outliers in daily returns
      6. Engineer new features

    Args:
        df:     Raw OHLCV DataFrame with DatetimeIndex
        ticker: Stock symbol (used for logging messages)

    Returns:
        Cleaned DataFrame with extra feature columns
    """
    logger.info(f"Cleaning {ticker}...")
    original_len = len(df)

    # ── Step 1: Sort by date ──────────────────────────────────────────────────
    df = df.sort_index()

    # ── Step 2: Remove duplicate rows ────────────────────────────────────────
    duplicates = df.index.duplicated(keep="first")
    n_dupes = duplicates.sum()
    if n_dupes > 0:
        logger.warning(f"  ⚠ {ticker}: Removed {n_dupes} duplicate date(s)")
        df = df[~duplicates]

    # ── Step 3: Handle missing values ────────────────────────────────────────
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logger.warning(f"  ⚠ {ticker}: Found {missing_before} missing value(s) — forward-filling")
        # Forward fill: each missing value takes the previous row's value
        # This is standard practice for financial time series (no trading on weekends)
        df = df.ffill()

    # Drop any rows that are STILL missing (e.g. missing at the very start)
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"  ⚠ {ticker}: Dropping {remaining_missing} rows still missing after ffill")
        df = df.dropna()

    # ── Step 4: Normalise data types ──────────────────────────────────────────
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure Volume is integer (it's a count of shares)
    df["Volume"] = df["Volume"].fillna(0).astype(int)

    # ── Step 5: Feature Engineering ───────────────────────────────────────────
    # 5a. Daily Return: percentage change in closing price day-over-day
    #     Formula: (today - yesterday) / yesterday * 100
    df["Daily_Return"] = df["Close"].pct_change() * 100

    # 5b. Rolling Moving Averages — smooth out short-term noise
    df[f"MA_{ROLLING_SHORT}"]  = df["Close"].rolling(window=ROLLING_SHORT).mean()
    df[f"MA_{ROLLING_LONG}"]   = df["Close"].rolling(window=ROLLING_LONG).mean()

    # 5c. Volatility — rolling standard deviation of daily returns (risk measure)
    #     Higher value = more volatile / riskier asset
    df["Volatility_30d"] = df["Daily_Return"].rolling(window=ROLLING_LONG).std()

    # 5d. Bollinger Bands — used to identify potential breakouts
    #     Upper band = MA + 2 standard deviations
    #     Lower band = MA - 2 standard deviations
    bb_mid = df["Close"].rolling(window=BOLLINGER_WINDOW).mean()
    bb_std = df["Close"].rolling(window=BOLLINGER_WINDOW).std()
    df["BB_Mid"]   = bb_mid
    df["BB_Upper"] = bb_mid + (BOLLINGER_STD * bb_std)
    df["BB_Lower"] = bb_mid - (BOLLINGER_STD * bb_std)

    # ── Step 6: Outlier Detection ─────────────────────────────────────────────
    # Flag daily returns more than 3 standard deviations from the mean.
    # These might be: earnings surprises, stock splits, data errors.
    returns = df["Daily_Return"].dropna()
    mean_r  = returns.mean()
    std_r   = returns.std()
    threshold = 3 * std_r

    df["Is_Outlier"] = (df["Daily_Return"] - mean_r).abs() > threshold
    n_outliers = df["Is_Outlier"].sum()
    if n_outliers > 0:
        logger.warning(f"  ⚠ {ticker}: Flagged {n_outliers} outlier day(s) (|return| > 3σ)")
        outlier_dates = df[df["Is_Outlier"]].index.strftime("%Y-%m-%d").tolist()
        logger.warning(f"    Dates: {outlier_dates}")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_len = len(df)
    logger.info(f"  ✓ {ticker}: {original_len} → {final_len} rows after cleaning")

    return df


def clean_all_stocks(stock_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Clean all stocks and save the processed DataFrames to CSV files.

    Args:
        stock_data: Dict of raw DataFrames from the collection module

    Returns:
        Dict of cleaned DataFrames
    """
    logger.info("=" * 60)
    logger.info("STEP 2: DATA CLEANING & PROCESSING")
    logger.info("=" * 60)

    cleaned = {}

    for ticker, df in stock_data.items():
        try:
            clean_df = clean_stock_dataframe(df.copy(), ticker)

            # Save processed data
            output_path = DATA_PROC_DIR / f"{ticker}_processed.csv"
            clean_df.to_csv(output_path)
            logger.info(f"  💾 Saved processed data to {output_path}")

            cleaned[ticker] = clean_df

        except Exception as e:
            logger.error(f"Failed to clean {ticker}: {e}")

    logger.info(f"\nCleaning complete. Processed {len(cleaned)} tickers.")
    return cleaned


def get_summary_stats(cleaned_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute a summary statistics table for the report.
    Useful for copy-pasting into your technical report.

    Returns a DataFrame with one row per ticker showing:
    mean return, volatility, best day, worst day, etc.
    """
    rows = []
    for ticker, df in cleaned_data.items():
        returns = df["Daily_Return"].dropna()
        rows.append({
            "Ticker":         ticker,
            "Start Date":     df.index[0].date(),
            "End Date":       df.index[-1].date(),
            "Trading Days":   len(df),
            "Avg Close ($)":  round(df["Close"].mean(), 2),
            "Mean Return (%)":round(returns.mean(), 3),
            "Volatility (%)": round(returns.std(), 3),
            "Best Day (%)":   round(returns.max(), 2),
            "Worst Day (%)":  round(returns.min(), 2),
            "Outliers":       int(df["Is_Outlier"].sum()),
        })

    summary = pd.DataFrame(rows)
    summary.set_index("Ticker", inplace=True)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Run directly to test: python src/cleaning/clean_data.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load raw data from CSV files for testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import DATA_RAW_DIR, TICKERS

    raw_data = {}
    for ticker in TICKERS:
        path = DATA_RAW_DIR / f"{ticker}_raw.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            raw_data[ticker] = df

    if raw_data:
        cleaned = clean_all_stocks(raw_data)
        summary = get_summary_stats(cleaned)
        print("\n📊 Summary Statistics:")
        print(summary.to_string())
    else:
        print("No raw data found. Run fetch_data.py first.")