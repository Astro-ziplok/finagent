"""
tests/test_visualization.py — Unit Tests for Visualization Module
=================================================================
Tests that charts are generated without errors and saved to disk.
Uses a small synthetic dataset so tests run fast.

Run with:  python -m pytest tests/ -v
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_cleaned_df(ticker="TEST", n=60, seed=42):
    """Create a minimal cleaned DataFrame with all feature columns."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n))
    close = np.maximum(close, 1)

    df = pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)

    # Add feature columns (normally added by clean_data.py)
    df["Daily_Return"]  = df["Close"].pct_change() * 100
    df["MA_7"]          = df["Close"].rolling(7).mean()
    df["MA_30"]         = df["Close"].rolling(30).mean()
    df["Volatility_30d"]= df["Daily_Return"].rolling(30).std()
    bb_mid              = df["Close"].rolling(20).mean()
    bb_std              = df["Close"].rolling(20).std()
    df["BB_Mid"]        = bb_mid
    df["BB_Upper"]      = bb_mid + 2 * bb_std
    df["BB_Lower"]      = bb_mid - 2 * bb_std
    df["Is_Outlier"]    = df["Daily_Return"].abs() > (df["Daily_Return"].std() * 3)
    return df


class TestChartGeneration(unittest.TestCase):
    """Tests that each chart function runs without error and saves a file."""

    def setUp(self):
        self.df      = _make_cleaned_df("AAPL")
        self.df2     = _make_cleaned_df("MSFT", seed=99)
        self.cleaned = {"AAPL": self.df, "MSFT": self.df2}

        # Ensure output dirs exist
        from config import REPORTS_CHARTS_DIR
        self.charts_dir = REPORTS_CHARTS_DIR
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def test_plot_price_and_volume_saves_file(self):
        from src.visualization.charts import plot_price_and_volume
        path = plot_price_and_volume(self.df, "AAPL")
        self.assertTrue(Path(path).exists(),
                        "chart1_price_volume.png was not saved")

    def test_plot_bollinger_bands_saves_file(self):
        from src.visualization.charts import plot_bollinger_bands
        path = plot_bollinger_bands(self.df, "AAPL")
        self.assertTrue(Path(path).exists(),
                        "chart4_bollinger_bands.png was not saved")

    def test_plot_return_distributions_saves_file(self):
        from src.visualization.charts import plot_return_distributions
        path = plot_return_distributions(self.cleaned)
        self.assertTrue(Path(path).exists(),
                        "chart3_return_distributions.png was not saved")

    def test_plot_correlation_heatmap_saves_file(self):
        from src.visualization.charts import plot_correlation_heatmap
        path = plot_correlation_heatmap(self.cleaned)
        self.assertTrue(Path(path).exists(),
                        "chart2_correlation_heatmap.png was not saved")

    def test_plot_comparative_returns_saves_file(self):
        from src.visualization.charts import plot_comparative_returns
        path = plot_comparative_returns(self.cleaned)
        self.assertTrue(Path(path).exists(),
                        "chart5_comparative_returns.png was not saved")

    def test_comparison_charts_saved_to_named_folder(self):
        """Comparison charts must go into compare_AAPL_MSFT/ subfolder."""
        from src.visualization.charts import generate_comparison_charts
        paths = generate_comparison_charts(self.cleaned)
        self.assertGreater(len(paths), 0, "No comparison charts were generated")
        for p in paths:
            folder = Path(p).parent.name
            self.assertTrue(
                folder.startswith("compare_"),
                f"Chart saved to wrong folder: {folder}"
            )

    def test_single_ticker_skips_comparison(self):
        """Comparison charts should not be generated for a single ticker."""
        from src.visualization.charts import generate_comparison_charts
        paths = generate_comparison_charts({"AAPL": self.df})
        self.assertEqual(len(paths), 0,
                         "Comparison charts generated for single ticker (should be empty)")

    def test_generate_all_charts_returns_list(self):
        """generate_all_charts should return a non-empty list of file paths."""
        from src.visualization.charts import generate_all_charts
        paths = generate_all_charts(self.cleaned)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)
        for p in paths:
            self.assertTrue(Path(p).exists(), f"Chart file not found: {p}")


if __name__ == "__main__":
    unittest.main(verbosity=2)