"""
tests/test_cleaning.py — Unit Tests for Data Cleaning Module
=============================================================
Run with:  python -m pytest tests/ -v
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cleaning.clean_data import clean_stock_dataframe, get_summary_stats


def _make_sample_df(n_days=100, seed=42):
    """Create a realistic sample OHLCV DataFrame for testing."""
    np.random.seed(seed)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="B")
    close = 100 + np.cumsum(np.random.randn(n_days))
    close = np.maximum(close, 1)
    return pd.DataFrame({
        "Open":   close * np.random.uniform(0.99, 1.01, n_days),
        "High":   close * np.random.uniform(1.00, 1.02, n_days),
        "Low":    close * np.random.uniform(0.98, 1.00, n_days),
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 50_000_000, n_days),
    }, index=dates)


class TestCleanStockDataframe(unittest.TestCase):

    def setUp(self):
        self.df = _make_sample_df()
        self.ticker = "TEST"

    # ── Output structure ──────────────────────────────────────────────────────

    def test_returns_dataframe(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        self.assertIsInstance(result, pd.DataFrame)

    def test_original_columns_preserved(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, result.columns)

    def test_feature_columns_added(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        for col in ["Daily_Return", "MA_7", "MA_30",
                    "Volatility_30d", "BB_Mid", "BB_Upper", "BB_Lower", "Is_Outlier"]:
            self.assertIn(col, result.columns, f"Feature '{col}' missing")

    def test_index_is_datetime(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_sorted_ascending(self):
        shuffled = self.df.sample(frac=1)
        result = clean_stock_dataframe(shuffled, self.ticker)
        self.assertTrue(result.index.is_monotonic_increasing)

    # ── Missing values ────────────────────────────────────────────────────────

    def test_missing_values_filled(self):
        df = self.df.copy()
        df.loc[df.index[5], "Close"] = np.nan
        df.loc[df.index[10], "Close"] = np.nan
        result = clean_stock_dataframe(df, self.ticker)
        self.assertEqual(result["Close"].isna().sum(), 0)

    def test_consecutive_missing_values_filled(self):
        df = self.df.copy()
        df.loc[df.index[20:25], "Close"] = np.nan
        result = clean_stock_dataframe(df, self.ticker)
        self.assertEqual(result["Close"].isna().sum(), 0)

    # ── Duplicate removal ─────────────────────────────────────────────────────

    def test_duplicate_dates_removed(self):
        df = self.df.copy()
        df = pd.concat([df, df.iloc[[0]]])
        result = clean_stock_dataframe(df, self.ticker)
        self.assertEqual(result.index.duplicated().sum(), 0)

    # ── Feature engineering ───────────────────────────────────────────────────

    def test_daily_return_calculation(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        expected = (self.df["Close"].iloc[1] / self.df["Close"].iloc[0] - 1) * 100
        self.assertAlmostEqual(result["Daily_Return"].iloc[1], expected, places=4)

    def test_ma7_nan_for_first_six_rows(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        self.assertTrue(result["MA_7"].iloc[:6].isna().all())
        self.assertFalse(pd.isna(result["MA_7"].iloc[6]))

    def test_bollinger_upper_always_above_lower(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        valid = result[["BB_Upper", "BB_Lower"]].dropna()
        self.assertTrue((valid["BB_Upper"] >= valid["BB_Lower"]).all())

    def test_bollinger_mid_equals_20day_ma(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        expected = result["Close"].rolling(window=20).mean()
        pd.testing.assert_series_equal(
            result["BB_Mid"].dropna(),
            expected.dropna(),
            check_names=False, rtol=1e-5)

    # ── Outlier detection ─────────────────────────────────────────────────────

    def test_outlier_column_is_boolean(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        self.assertEqual(result["Is_Outlier"].dtype, bool)

    def test_extreme_return_flagged(self):
        df = self.df.copy()
        df.loc[df.index[50], "Close"] = df["Close"].iloc[49] * 1.5  # +50% spike
        result = clean_stock_dataframe(df, self.ticker)
        self.assertTrue(result["Is_Outlier"].iloc[50])

    def test_normal_data_low_outlier_rate(self):
        result = clean_stock_dataframe(self.df.copy(), self.ticker)
        self.assertLess(result["Is_Outlier"].mean(), 0.05)


class TestGetSummaryStats(unittest.TestCase):

    def setUp(self):
        df = _make_sample_df()
        self.cleaned = {
            "AAPL": clean_stock_dataframe(df.copy(), "AAPL"),
            "MSFT": clean_stock_dataframe(df.copy(), "MSFT"),
        }

    def test_returns_dataframe(self):
        self.assertIsInstance(get_summary_stats(self.cleaned), pd.DataFrame)

    def test_one_row_per_ticker(self):
        self.assertEqual(len(get_summary_stats(self.cleaned)), 2)

    def test_required_columns_present(self):
        result = get_summary_stats(self.cleaned)
        for col in ["Mean Return (%)", "Volatility (%)", "Best Day (%)", "Worst Day (%)"]:
            self.assertIn(col, result.columns)

    def test_volatility_positive(self):
        result = get_summary_stats(self.cleaned)
        self.assertTrue((result["Volatility (%)"] > 0).all())

    def test_best_day_gte_worst_day(self):
        result = get_summary_stats(self.cleaned)
        self.assertTrue((result["Best Day (%)"] >= result["Worst Day (%)"]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)