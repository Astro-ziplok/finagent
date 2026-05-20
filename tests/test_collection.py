"""
tests/test_collection.py — Unit Tests for Data Collection Module
================================================================
Tests fetch_data.py functions using mocking so no real API calls
are made during testing (avoids rate limits and network dependency).

Run with:  python -m pytest tests/ -v
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTickerSanitisation(unittest.TestCase):
    """
    Tests for ticker cleaning logic.
    The cleaning is applied inline in fetch_stock_prices — we test
    the output by checking what ticker key appears in the result.
    """

    def _make_mock_df(self, n=50):
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n))
        return pd.DataFrame({
            "Open": close * 0.99, "High": close * 1.01,
            "Low":  close * 0.98, "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        }, index=dates)

    @patch("src.collection.fetch_data.yf.download")
    def test_dollar_prefix_stripped(self, mock_dl):
        """$AAPL should be cleaned to AAPL before download."""
        mock_dl.return_value = self._make_mock_df()
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["$AAPL"])
        self.assertIn("AAPL", result)
        self.assertNotIn("$AAPL", result)

    @patch("src.collection.fetch_data.yf.download")
    def test_lowercase_converted_to_uppercase(self, mock_dl):
        mock_dl.return_value = self._make_mock_df()
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["aapl"])
        self.assertIn("AAPL", result)

    @patch("src.collection.fetch_data.yf.download")
    def test_combined_dollar_and_lowercase(self, mock_dl):
        mock_dl.return_value = self._make_mock_df()
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["$googl"])
        self.assertIn("GOOGL", result)


class TestFetchStockPricesMocked(unittest.TestCase):
    """
    Tests for fetch_stock_prices() using mocked yfinance calls.
    No real network requests are made.
    """

    def _make_mock_df(self, n=252):
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n))
        return pd.DataFrame({
            "Open": close * 0.99, "High": close * 1.01,
            "Low":  close * 0.98, "Close": close,
            "Volume": np.random.randint(1_000_000, 50_000_000, n),
        }, index=dates)

    @patch("src.collection.fetch_data.yf.download")
    def test_returns_dict_with_ticker_key(self, mock_dl):
        mock_dl.return_value = self._make_mock_df()
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["AAPL"])
        self.assertIn("AAPL", result)

    @patch("src.collection.fetch_data.yf.download")
    def test_empty_download_skips_ticker(self, mock_dl):
        """If yfinance returns empty DataFrame, ticker should be skipped."""
        mock_dl.return_value = pd.DataFrame()
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["FAKE"])
        self.assertEqual(len(result), 0)

    @patch("src.collection.fetch_data.yf.download")
    def test_result_has_ohlcv_columns(self, mock_dl):
        mock_dl.return_value = self._make_mock_df()
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["AAPL"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, result["AAPL"].columns)

    @patch("src.collection.fetch_data.yf.download")
    def test_result_has_datetime_index(self, mock_dl):
        mock_dl.return_value = self._make_mock_df()
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["AAPL"])
        self.assertIsInstance(result["AAPL"].index, pd.DatetimeIndex)

    @patch("src.collection.fetch_data.yf.download")
    def test_multiple_tickers_returned(self, mock_dl):
        mock_dl.return_value = self._make_mock_df()
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["AAPL", "MSFT"])
        self.assertEqual(len(result), 2)

    @patch("src.collection.fetch_data.yf.download")
    def test_multiindex_columns_flattened(self, mock_dl):
        """yfinance sometimes returns MultiIndex columns — these must be flattened."""
        df = self._make_mock_df()
        df.columns = pd.MultiIndex.from_tuples([(c, "AAPL") for c in df.columns])
        mock_dl.return_value = df
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertNotIsInstance(result["AAPL"].columns, pd.MultiIndex)

    @patch("src.collection.fetch_data.yf.download")
    def test_no_timezone_in_index(self, mock_dl):
        """Timezone info must be stripped from the DatetimeIndex."""
        df = self._make_mock_df()
        df.index = df.index.tz_localize("UTC")
        mock_dl.return_value = df
        from src.collection.fetch_data import fetch_stock_prices
        result = fetch_stock_prices(["AAPL"])
        self.assertIsNone(result["AAPL"].index.tz)


class TestNewsAPIHandling(unittest.TestCase):
    """Tests for news fetching — checks graceful handling of missing keys."""

    def test_missing_newsapi_key_returns_empty(self):
        """If NEWS_API_KEY is not set, fetch_financial_news should return empty DataFrame."""
        import src.collection.fetch_data as fd
        original = fd.NEWS_API_KEY
        fd.NEWS_API_KEY = ""
        try:
            result = fd.fetch_financial_news(["AAPL"])
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)
        finally:
            fd.NEWS_API_KEY = original


if __name__ == "__main__":
    unittest.main(verbosity=2)