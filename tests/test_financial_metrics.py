"""
tests/test_financial_metrics.py — Tests for financial calculation correctness
==============================================================================
Checks the metric definitions that the AI analysis relies on: annualised
Sharpe ratio, calendar quarters, benchmark statistics, unit conversions,
pre-computed directions, and keyword sentiment. No network or LLM calls.

Run with:  python -m pytest tests/ -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_analysis import _make_cleaned_df  # noqa: E402


class TestSharpeRatio(unittest.TestCase):

    def test_sharpe_is_annualised_with_risk_free_rate(self):
        from src.analysis.ai_analysis import _annualised_sharpe
        from config import RISK_FREE_RATE, TRADING_DAYS
        np.random.seed(0)
        r_pct = pd.Series(np.random.normal(0.1, 1.5, 500))  # daily % returns
        r = r_pct / 100
        rf_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
        expected = (r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS)
        self.assertAlmostEqual(_annualised_sharpe(r_pct), round(expected, 2), places=2)

    def test_sharpe_far_larger_than_daily_ratio(self):
        """Regression: the old code reported the raw daily mean/std (~16x too small)."""
        from src.analysis.ai_analysis import _annualised_sharpe
        r_pct = pd.Series([0.35, -1.0, 2.1, 0.4, -0.6, 1.2, 0.3, -0.2] * 40)
        daily_ratio = r_pct.mean() / r_pct.std()
        self.assertGreater(_annualised_sharpe(r_pct), daily_ratio * 10)

    def test_sharpe_labels(self):
        from src.analysis.ai_analysis import _sharpe_label
        self.assertIn("negative", _sharpe_label(-0.3))
        self.assertIn("weak", _sharpe_label(0.3))
        self.assertIn("good", _sharpe_label(1.5))
        self.assertIn("exceptional", _sharpe_label(3.1))


class TestContextPeriods(unittest.TestCase):

    def test_quarters_are_calendar_quarters(self):
        from src.analysis.ai_analysis import _build_data_context
        df = _make_cleaned_df(n=252)  # business days from 2024-01-01
        ctx = _build_data_context({"X": df})["X"]
        self.assertEqual(ctx["quarterly_breakdown"][0]["period"], "2024-Q1")
        self.assertTrue(all("-Q" in q["period"] for q in ctx["quarterly_breakdown"]))

    def test_quarter_returns_chain_to_total_return(self):
        """Quarter returns are measured from the previous quarter's close, so they compound to the total."""
        from src.analysis.ai_analysis import _build_data_context
        df = _make_cleaned_df(n=252)
        ctx = _build_data_context({"X": df})["X"]
        compounded = np.prod([1 + q["pct_change"] / 100 for q in ctx["quarterly_breakdown"]]) - 1
        self.assertAlmostEqual(compounded * 100, ctx["price_summary"]["total_return_pct"], delta=0.1)

    def test_multi_year_range_not_called_52_week(self):
        from src.analysis.ai_analysis import _build_data_context
        df = _make_cleaned_df(n=1000)  # ~4 years
        ctx = _build_data_context({"X": df})["X"]
        self.assertNotIn("52-week", ctx["price_summary"]["high_low_label"])
        self.assertFalse(ctx["period"]["is_one_year"])
        self.assertGreater(len(ctx["yearly_breakdown"]), 1)

    def test_one_year_range_is_52_week(self):
        from src.analysis.ai_analysis import _build_data_context
        df = _make_cleaned_df(n=252)
        ctx = _build_data_context({"X": df})["X"]
        self.assertEqual(ctx["price_summary"]["high_low_label"], "52-week")

    def test_momentum_uses_n_trading_days(self):
        from src.analysis.ai_analysis import _pct_over
        close = pd.Series([100.0] * 10 + [110.0])
        self.assertEqual(_pct_over(close, 7), 10.0)
        self.assertIsNone(_pct_over(close, 50))


class TestBenchmark(unittest.TestCase):

    def test_benchmark_stats_present_and_beta_correct(self):
        from src.analysis.ai_analysis import _build_data_context
        bench = _make_cleaned_df(seed=1)
        stock = bench.copy()
        # Stock moves exactly 2x the benchmark each day -> beta == 2
        bench_ret = bench["Close"].pct_change().fillna(0)
        stock["Close"] = 100 * np.cumprod(1 + 2 * bench_ret)
        ctx = _build_data_context({"X": stock}, benchmark_df=bench)["X"]
        vb = ctx["vs_benchmark"]
        self.assertAlmostEqual(vb["beta_vs_benchmark"], 2.0, places=1)
        self.assertAlmostEqual(vb["correlation_vs_benchmark"], 1.0, places=2)
        self.assertEqual(vb["outperformed_benchmark"], vb["excess_return_pct"] > 0)

    def test_no_benchmark_key_without_data(self):
        from src.analysis.ai_analysis import _build_data_context
        ctx = _build_data_context({"X": _make_cleaned_df()})["X"]
        self.assertNotIn("vs_benchmark", ctx)


class TestPrecomputedDirections(unittest.TestCase):

    def test_direction_words(self):
        from src.analysis.ai_analysis import _direction
        self.assertEqual(_direction(109.9, 113.83), "decreased")
        self.assertEqual(_direction(120, 100), "increased")
        self.assertEqual(_direction(100.1, 100), "roughly flat")
        self.assertEqual(_direction(None, 100), "n/a")

    def test_income_facts_quarter_direction(self):
        """Regression: report said revenue 'increased from $113.83B to $109.90B'."""
        from src.analysis.ai_analysis import _income_facts
        quarterly = [
            {"period": "2026-03-31", "total_revenue": 109.90e9, "net_income": 62.58e9, "net_margin_pct": 56.9},
            {"period": "2025-12-31", "total_revenue": 113.83e9, "net_income": 34.5e9, "net_margin_pct": 30.3},
        ]
        facts = _income_facts([], quarterly)
        q = facts["latest_quarter_vs_prior_quarter"]
        self.assertEqual(q["revenue_direction"], "decreased")
        self.assertEqual(q["net_income_direction"], "increased")
        self.assertIn("warning", facts)  # >40% net margin flagged as possible one-off

    def test_metrics_facts_eps_vs_revenue(self):
        """Regression: report said EPS growth 16.36% was 'higher' than revenue growth 17.0%."""
        from src.analysis.ai_analysis import _metrics_facts
        facts = _metrics_facts({
            "earnings": {"eps_growth_pct": 16.36},
            "growth": {"revenue_yoy_pct": 17.0},
            "valuation": {"beta": 1.06, "pe_trailing": 36.6, "pe_forward": 31.5},
            "financial_health": {"debt_to_equity_x": 0.8},
        })
        self.assertIn("LOWER", facts["expected_eps_growth_vs_recent_revenue_growth"])
        self.assertIn("market-like", facts["beta_interpretation"])
        self.assertIn("moderate", facts["leverage_assessment"])
        self.assertIn("above", facts["pe_vs_sp500_typical_20_to_25x"])

    def test_comparative_rankings_are_sorted(self):
        from src.analysis.ai_analysis import _build_data_context, generate_comparative_analysis
        ctx = _build_data_context({"A": _make_cleaned_df(seed=1), "B": _make_cleaned_df(seed=2),
                                   "C": _make_cleaned_df(seed=3)})
        with patch("src.analysis.ai_analysis._call_groq", return_value="ok") as m:
            generate_comparative_analysis(ctx)
        prompt = m.call_args[0][0]
        best = max(ctx, key=lambda t: ctx[t]["price_summary"]["total_return_pct"])
        self.assertIn(f'"return_ranking_best_to_worst": [\n    "{best} (', prompt)


class TestFundamentalUnits(unittest.TestCase):

    def test_dividend_yield_from_rate_and_price(self):
        from src.fundamentals.yfinance_fundamentals import _dividend_yield_pct
        # yfinance >=0.2.5x reports dividendYield=0.31 meaning 0.31%; must NOT become 31%
        info = {"dividendYield": 0.31, "dividendRate": 1.04, "currentPrice": 335.0}
        self.assertAlmostEqual(_dividend_yield_pct(info), 0.31, places=2)

    def test_dividend_yield_fallback_trailing(self):
        from src.fundamentals.yfinance_fundamentals import _dividend_yield_pct
        self.assertAlmostEqual(_dividend_yield_pct({"trailingAnnualDividendYield": 0.0030354}), 0.3, places=2)
        self.assertIsNone(_dividend_yield_pct({}))

    def test_key_metrics_units(self):
        """Margins keep precision; D/E converted from Yahoo's percent to a multiple."""
        import src.fundamentals.yfinance_fundamentals as yff

        class FakeTicker:
            info = {"grossMargins": 0.48786, "profitMargins": 0.2692, "operatingMargins": 0.3197,
                    "debtToEquity": 84.286, "revenueGrowth": 0.117, "trailingEps": 8.25,
                    "forwardEps": 9.6, "currentPrice": 300.0, "marketCap": 4.5e12,
                    "freeCashflow": 1.0e11, "dividendRate": 1.04}

        with patch("yfinance.Ticker", return_value=FakeTicker()):
            m = yff.fetch_key_metrics("AAPL")
        self.assertEqual(m["profitability"]["gross_margin_pct"], 48.79)
        self.assertEqual(m["financial_health"]["debt_to_equity_x"], 0.84)
        self.assertEqual(m["growth"]["revenue_yoy_pct"], 11.7)
        self.assertAlmostEqual(m["dividend"]["dividend_yield_pct"], 0.35, places=2)
        self.assertIn("expected", m["earnings"]["eps_growth_basis"])


class TestKeywordSentiment(unittest.TestCase):

    def test_no_substring_false_positives(self):
        from src.collection.fetch_data import keyword_sentiment
        self.assertEqual(keyword_sentiment("Apple issues software update for enterprise customers again"), "Neutral")
        self.assertEqual(keyword_sentiment("Executive team meets investors"), "Neutral")

    def test_real_keywords(self):
        from src.collection.fetch_data import keyword_sentiment
        self.assertEqual(keyword_sentiment("Nvidia shares surge after earnings beat"), "Positive")
        self.assertEqual(keyword_sentiment("Tesla stock falls on delivery miss"), "Negative")
        self.assertEqual(keyword_sentiment("Stock rises despite analyst downgrade"), "Mixed")


class TestCommodityHandling(unittest.TestCase):

    def test_all_missing_commodities_skip_llm(self):
        from src.analysis.ai_analysis import generate_commodity_context
        data = {"WTI_Oil": {"latest_price": None, "trend": "N/A"},
                "Gold": {"latest_price": None, "trend": "N/A"}}
        with patch("src.analysis.ai_analysis._call_groq") as m:
            out = generate_commodity_context(data)
        m.assert_not_called()
        self.assertIn("No commodity price data", out)

    def test_sector_analysis_handles_missing_commodity_change(self):
        """Regression: formatting 'N/A' with :+.1f raised ValueError."""
        from src.analysis.ai_analysis import generate_sector_analysis
        sectors = {"Technology": {"etf": "XLK", "1w_return_pct": 1.0, "1m_return_pct": 2.0,
                                  "3m_return_pct": 5.0, "volatility_30d": 1.1, "trend": "Rising"}}
        commodities = {"WTI_Oil": {"latest_price": None}, "Gold": {"latest_price": 2400.0, "trend": "Rising"}}
        with patch("src.analysis.ai_analysis._call_groq", return_value="ok"):
            self.assertEqual(generate_sector_analysis(sectors, commodities), "ok")


if __name__ == "__main__":
    unittest.main()
