"""
tests/test_analysis.py — Unit Tests for AI Analysis Module
===========================================================
Tests the data context builder and analysis functions.
LLM calls (Groq) are mocked so tests run offline with no API cost.

Run with:  python -m pytest tests/ -v
"""

import sys
import json
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_cleaned_df(n=252, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n))
    close = np.maximum(close, 1)
    df = pd.DataFrame({
        "Open":   close * 0.99, "High": close * 1.01,
        "Low":    close * 0.98, "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)
    df["Daily_Return"]   = df["Close"].pct_change() * 100
    df["MA_7"]           = df["Close"].rolling(7).mean()
    df["MA_30"]          = df["Close"].rolling(30).mean()
    df["Volatility_30d"] = df["Daily_Return"].rolling(30).std()
    bb_mid               = df["Close"].rolling(20).mean()
    bb_std               = df["Close"].rolling(20).std()
    df["BB_Mid"]         = bb_mid
    df["BB_Upper"]       = bb_mid + 2 * bb_std
    df["BB_Lower"]       = bb_mid - 2 * bb_std
    df["Is_Outlier"]     = (df["Daily_Return"].abs() > df["Daily_Return"].std() * 3)
    return df


class TestBuildDataContext(unittest.TestCase):
    """Tests for _build_data_context() — the data preparation layer."""

    def setUp(self):
        df = _make_cleaned_df()
        self.cleaned = {"AAPL": df, "MSFT": _make_cleaned_df(seed=99)}
        from src.analysis.ai_analysis import _build_data_context
        self.context = _build_data_context(self.cleaned)

    def test_all_tickers_in_context(self):
        self.assertIn("AAPL", self.context)
        self.assertIn("MSFT", self.context)

    def test_context_has_required_keys(self):
        for key in ["period", "full_year_price", "returns",
                    "short_term_momentum", "technical_signals",
                    "quarterly_breakdown", "outlier_dates"]:
            self.assertIn(key, self.context["AAPL"],
                          f"Context missing key: {key}")

    def test_total_return_is_float(self):
        ret = self.context["AAPL"]["full_year_price"]["total_return_pct"]
        self.assertIsInstance(ret, float)

    def test_win_rate_between_0_and_100(self):
        wr = self.context["AAPL"]["returns"]["win_rate_pct"]
        self.assertGreaterEqual(wr, 0)
        self.assertLessEqual(wr, 100)

    def test_ma_signal_is_string(self):
        signal = self.context["AAPL"]["technical_signals"]["ma_signal"]
        self.assertIsInstance(signal, str)
        self.assertGreater(len(signal), 0)

    def test_bb_signal_is_string(self):
        signal = self.context["AAPL"]["technical_signals"]["bb_signal"]
        self.assertIsInstance(signal, str)

    def test_quarterly_breakdown_has_entries(self):
        qb = self.context["AAPL"]["quarterly_breakdown"]
        self.assertIsInstance(qb, list)
        self.assertGreater(len(qb), 0)

    def test_outlier_dates_is_list(self):
        od = self.context["AAPL"]["outlier_dates"]
        self.assertIsInstance(od, list)

    def test_sharpe_ratio_is_numeric(self):
        sharpe = self.context["AAPL"]["returns"]["sharpe_ratio_approx"]
        self.assertIsInstance(sharpe, (int, float))

    def test_52w_high_gte_52w_low(self):
        prices = self.context["AAPL"]["full_year_price"]
        self.assertGreaterEqual(prices["52w_high"], prices["52w_low"])


class TestAnalysisFunctionsWithMockedLLM(unittest.TestCase):
    """
    Tests for analysis generator functions.
    Groq API is mocked — we test that functions call the LLM correctly
    and handle both successful and failed responses gracefully.
    """

    def setUp(self):
        df = _make_cleaned_df()
        self.cleaned = {"AAPL": df}
        from src.analysis.ai_analysis import _build_data_context
        self.context = _build_data_context(self.cleaned)

    @patch("src.analysis.ai_analysis._call_groq")
    def test_trend_summary_returns_string(self, mock_groq):
        mock_groq.return_value = "Mocked trend summary for AAPL."
        from src.analysis.ai_analysis import generate_trend_summary
        result = generate_trend_summary("AAPL", self.context)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @patch("src.analysis.ai_analysis._call_groq")
    def test_trend_summary_calls_groq(self, mock_groq):
        mock_groq.return_value = "Trend summary."
        from src.analysis.ai_analysis import generate_trend_summary
        generate_trend_summary("AAPL", self.context)
        mock_groq.assert_called_once()

    @patch("src.analysis.ai_analysis._call_groq")
    def test_anomaly_commentary_no_outliers(self, mock_groq):
        """With no outlier dates, should return a static string without calling LLM."""
        from src.analysis.ai_analysis import _build_data_context, generate_anomaly_commentary
        # Force no outliers
        df = _make_cleaned_df()
        df["Is_Outlier"] = False
        context = _build_data_context({"AAPL": df})
        result = generate_anomaly_commentary("AAPL", context)
        mock_groq.assert_not_called()
        self.assertIn("No statistical outliers", result)

    @patch("src.analysis.ai_analysis._call_groq")
    def test_risk_commentary_returns_string(self, mock_groq):
        mock_groq.return_value = "Mocked risk commentary."
        from src.analysis.ai_analysis import generate_risk_commentary
        result = generate_risk_commentary(self.context)
        self.assertIsInstance(result, str)

    @patch("src.analysis.ai_analysis._call_groq")
    def test_comparative_single_stock_uses_sector_benchmarking(self, mock_groq):
        """Single stock should trigger sector benchmarking, not multi-stock comparison."""
        mock_groq.return_value = "Sector benchmarking output."
        from src.analysis.ai_analysis import generate_comparative_analysis
        result = generate_comparative_analysis(self.context)
        # The prompt for single stock should include "Sector Benchmarking"
        call_args = mock_groq.call_args[0][0]  # First positional arg = prompt
        self.assertIn("Sector Benchmarking", call_args)

    @patch("src.analysis.ai_analysis._call_groq")
    def test_market_context_returns_string(self, mock_groq):
        mock_groq.return_value = "Market context output."
        from src.analysis.ai_analysis import generate_market_context
        result = generate_market_context("AAPL", self.context)
        self.assertIsInstance(result, str)

    @patch("src.analysis.ai_analysis._call_groq")
    def test_groq_failure_returns_warning_string(self, mock_groq):
        """If Groq API fails, functions should return a warning string, not crash."""
        mock_groq.return_value = "⚠️  Groq API call failed: Error code: 403"
        from src.analysis.ai_analysis import generate_trend_summary
        result = generate_trend_summary("AAPL", self.context)
        self.assertIsInstance(result, str)  # Should not raise an exception

    def test_call_groq_missing_key_returns_warning(self):
        """With no GROQ_API_KEY set, _call_groq should return a warning, not crash."""
        import src.analysis.ai_analysis as ai
        original = ai.GROQ_API_KEY
        ai.GROQ_API_KEY = ""
        try:
            result = ai._call_groq("test prompt")
            self.assertIn("⚠️", result)
        finally:
            ai.GROQ_API_KEY = original


if __name__ == "__main__":
    unittest.main(verbosity=2)