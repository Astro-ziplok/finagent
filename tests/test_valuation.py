"""
tests/test_valuation.py — Tests for the valuation models and engine
====================================================================
The maths is checked against textbook closed forms (Gordon growth, a
residual-income firm earning exactly its cost of equity, etc.). The engine
is run on mocked market data, so no network or API calls are made.

Run with:  python -m pytest tests/ -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.valuation import models as M  # noqa: E402


class TestModels(unittest.TestCase):

    def test_capm_and_wacc(self):
        self.assertAlmostEqual(M.capm(0.05, 1.2, 0.05), 0.11)
        w = M.wacc(0.10, 0.06, 0.25, equity_value=800, debt_value=200)
        self.assertAlmostEqual(w["weight_equity"], 0.8)
        self.assertAlmostEqual(w["wacc"], 0.8 * 0.10 + 0.2 * 0.06 * 0.75)

    def test_growth_path_fades_to_terminal(self):
        p = M.growth_path(0.20, 0.15, 0.025, years=10)
        self.assertEqual(len(p), 10)
        self.assertEqual(p[:2], [0.20, 0.15])
        self.assertTrue(all(a >= b for a, b in zip(p[1:], p[2:])))     # monotonic fade
        self.assertGreater(p[-1], 0.025)                                # reaches terminal after year 10

    def test_dcf_equals_gordon_when_growth_is_constant(self):
        """With constant growth g, the DCF must equal FCF1 / (WACC - g)."""
        r0, m, w, g = 1000.0, 0.2, 0.09, 0.025
        out = M.dcf_fcff(r0, [g] * 5, m, w, g, net_debt=0, shares=1)
        self.assertAlmostEqual(out["enterprise_value"], r0 * m * (1 + g) / (w - g), places=6)

    def test_dcf_subtracts_net_debt_and_divides_by_shares(self):
        a = M.dcf_fcff(1000, [0.05] * 5, 0.2, 0.09, 0.025, net_debt=0, shares=10)
        b = M.dcf_fcff(1000, [0.05] * 5, 0.2, 0.09, 0.025, net_debt=500, shares=10)
        self.assertAlmostEqual(a["per_share"] - b["per_share"], 50)

    def test_dcf_rejects_wacc_below_terminal_growth(self):
        with self.assertRaises(ValueError):
            M.dcf_fcff(1000, [0.05] * 5, 0.2, 0.025, 0.025, 0, 1)

    def test_reverse_dcf_recovers_growth(self):
        args = dict(base_revenue=1000, fcf_margin=0.2, wacc_rate=0.09, g_terminal=0.025, net_debt=100, shares=10)
        price = M.dcf_fcff(growth=[0.07] * 10, **{k: v for k, v in args.items() if k != "wacc_rate"},
                           wacc_rate=0.09)["per_share"]
        self.assertAlmostEqual(M.reverse_dcf(price, years=10, **args), 0.07, places=4)

    def test_residual_income_equals_book_when_roe_equals_cost_of_equity(self):
        """A firm earning exactly its cost of equity is worth its book value."""
        ke, b0 = 0.10, 50.0
        eps, b = [], b0
        for _ in range(5):
            eps.append(ke * b)
            b += eps[-1]                     # payout 0: book grows by earnings
        self.assertAlmostEqual(M.residual_income(b0, eps, 0.0, ke, 0.025)["per_share"], b0, places=6)

    def test_ddm_equals_gordon_when_growth_is_constant(self):
        d0, ke, g = 2.0, 0.08, 0.03
        self.assertAlmostEqual(M.ddm(d0, [g] * 5, ke, g)["per_share"], d0 * (1 + g) / (ke - g), places=6)

    def test_peer_stats_filters_and_requires_minimum(self):
        s = M.peer_stats([10, 12, 14, None, -5, 900], lo=1, hi=150)
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["median"], 12)
        self.assertIsNone(M.peer_stats([10, 12], min_n=3))

    def test_regression_perfect_fit(self):
        r = M.regression([1, 2, 3, 4], [3, 5, 7, 9])
        self.assertAlmostEqual(r["slope"], 2)
        self.assertAlmostEqual(r["intercept"], 1)
        self.assertAlmostEqual(r["r2"], 1)


def _info(**kw):
    base = {"shortName": "Example", "sector": "Technology", "industry": "Software - Infrastructure",
            "currentPrice": 100.0, "marketCap": 100e9, "sharesOutstanding": 1e9, "beta": 1.2,
            "forwardEps": 5.0, "trailingEps": 4.5, "bookValue": 20.0, "priceToBook": 5.0,
            "ebitda": 8e9, "totalRevenue": 40e9, "forwardPE": 20.0, "enterpriseToEbitda": 13.0,
            "enterpriseToRevenue": 2.6, "dividendRate": 0.5, "payoutRatio": 0.1, "returnOnEquity": 0.25,
            "financialCurrency": "USD", "currency": "USD", "fiftyTwoWeekLow": 80, "fiftyTwoWeekHigh": 120}
    base.update(kw)
    return base


def _yf_entry():
    years = ["2025-12-31", "2024-12-31", "2023-12-31"]
    return {
        "income": {"annual": [{"period": y, "total_revenue": (40 - 3 * i) * 1e9, "interest_expense": 0.3e9,
                               "pretax_income": 9e9, "tax_provision": 1.8e9, "diluted_eps": 4.5 - 0.4 * i}
                              for i, y in enumerate(years)]},
        "balance": {"annual": [{"period": years[0], "total_debt": 6e9, "cash_and_st_investments": 4e9}]},
        "cashflow": {"annual": [{"period": y, "free_cash_flow": 6e9, "stock_based_comp": 1e9} for y in years]},
    }


def _inputs(info):
    return {"ticker": "EXM", "info": info, "as_of": "2026-01-01",
            "revenue_est": {"0y": {"avg": 44e9, "growth": 0.10, "numberOfAnalysts": 20, "yearAgoRevenue": 40e9},
                            "+1y": {"avg": 48e9, "growth": 0.09}},
            "eps_est": {"0y": {"avg": 5.0, "numberOfAnalysts": 20}, "+1y": {"avg": 5.5}},
            "monthly_close": pd.Series([90 + i for i in range(36)],
                                       index=pd.date_range("2024-01-31", periods=36, freq="ME"))}


PEERS = {f"P{i}": {"name": f"Peer {i}", "forward_pe": 15 + i, "ev_ebitda": 10 + i, "ev_sales": 2 + i * 0.2,
                   "pb": 3 + i * 0.3, "roe": 0.15 + i * 0.01, "exp_eps_growth": 0.05 + i * 0.02}
         for i in range(6)}
RF = {"value": 0.045, "as_of": "2026-01-01", "source": "Test Treasury source"}


class TestEngine(unittest.TestCase):

    def _value(self, extra_inputs=None, **info_kw):
        from src.valuation import engine
        inputs = {**_inputs(_info(**info_kw)), **(extra_inputs or {})}
        with patch.object(engine, "company_inputs", return_value=inputs), \
             patch.object(engine, "risk_free_rate", return_value=RF), \
             patch.object(engine, "peer_multiples", return_value=PEERS), \
             patch.object(engine, "select_peers", return_value={"peers": list(PEERS), "basis": "industry",
                                                                 "rationale": "Same industry (test)."}):
            return engine.value_company("EXM", _yf_entry())

    def test_cash_generative_uses_dcf_and_comps(self):
        v = self._value()
        self.assertEqual(v["company_type"], "cash_generative")
        weights = {k: m["weight"] for k, m in v["methods"].items() if m["role"] == "weighted"}
        self.assertEqual(set(weights), {"dcf", "comps"})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertIn("skipped", v["methods"]["ri"])
        self.assertIn("skipped", v["methods"]["ddm"])
        self.assertIsNotNone(v["reverse_dcf"]["implied_growth"])
        self.assertAlmostEqual(v["fair_value"],
                               0.5 * v["methods"]["dcf"]["value"] + 0.5 * v["methods"]["comps"]["value"])

    def test_every_assumption_has_a_source(self):
        v = self._value()
        names = {a["name"] for a in v["assumptions"]}
        for needed in ("Risk-free rate", "Equity risk premium", "Beta (adjusted)", "WACC", "Terminal growth",
                       "Forecast horizon", "Free cash flow margin", "Peer group", "Method weights"):
            self.assertIn(needed, names)
        for a in v["assumptions"]:
            self.assertTrue(a["source"], f"{a['name']} has no source")
        self.assertIn("Test Treasury source", next(a["source"] for a in v["assumptions"] if a["name"] == "Risk-free rate"))

    def test_beta_is_blume_adjusted(self):
        v = self._value(beta=1.5)
        self.assertAlmostEqual(v["cost_of_capital"]["beta"], 0.67 * 1.5 + 0.33)

    def test_bank_uses_residual_income_not_dcf(self):
        v = self._value(industry="Banks - Diversified", sector="Financial Services")
        self.assertEqual(v["company_type"], "financial")
        self.assertIn("skipped", v["methods"]["dcf"])
        self.assertEqual(v["methods"]["ri"]["role"], "weighted")

    def test_dividend_payer_uses_ddm(self):
        v = self._value(dividendRate=4.0, payoutRatio=0.7)
        self.assertEqual(v["company_type"], "dividend")
        self.assertEqual(v["methods"]["ddm"]["role"], "weighted")

    def test_recent_stock_split_skips_pe_history(self):
        split = pd.Series([10.0], index=[pd.Timestamp("2025-06-10")])
        v = self._value(extra_inputs={"splits": split})
        self.assertIn("split", v["methods"]["history"]["skipped"].lower())

    def test_valuation_gap_note_direction(self):
        below = self._value(currentPrice=10.0)            # price far below every method
        self.assertIn("below the range", below["valuation_gap_note"])
        above = self._value(currentPrice=5000.0)          # price far above every method
        self.assertIn("above the range", above["valuation_gap_note"])

    def test_currency_mismatch_skips_per_share_models(self):
        v = self._value(financialCurrency="JPY")
        self.assertIn("JPY", v["methods"]["dcf"]["skipped"])


class TestPeerSelection(unittest.TestCase):

    def test_override_industry_sector_and_none(self):
        from src.valuation.inputs import select_peers
        self.assertEqual(select_peers("MA", "Financial Services", "Credit Services")["basis"], "curated")
        nv = select_peers("NVDA", "Technology", "Semiconductors")
        self.assertEqual(nv["basis"], "industry")
        self.assertNotIn("NVDA", nv["peers"])
        self.assertEqual(select_peers("SONY", "Technology", "Consumer Electronics")["basis"], "sector")
        self.assertEqual(select_peers("XYZ", "Unknown", "Unknown")["basis"], "none")


class TestReportValuation(unittest.TestCase):

    def test_report_includes_valuation_and_sources(self):
        from src.valuation import engine
        from src.report.research_report import build_research_report
        from tests.test_analysis import _make_cleaned_df
        from tests.test_research_report import _fake_yf, THESIS, SECTIONS
        tmp = Path(tempfile.mkdtemp())
        try:
            with patch.object(engine, "company_inputs", return_value=_inputs(_info())), \
                 patch.object(engine, "risk_free_rate", return_value=RF), \
                 patch.object(engine, "peer_multiples", return_value=PEERS), \
                 patch.object(engine, "select_peers", return_value={"peers": list(PEERS), "basis": "industry",
                                                                     "rationale": "Same industry (test)."}), \
                 patch("src.report.research_report._call_groq_json",
                       side_effect=lambda p, *a, **k: THESIS if "Write the investment thesis" in p else SECTIONS):
                yf = _fake_yf()
                yf["income"]["annual"][0]["tax_provision"] = 1e9
                path = build_research_report(["AAA"], {"AAA": _make_cleaned_df(seed=1)}, {"AAA": yf},
                                             _make_cleaned_df(seed=3), out_root=tmp)
            page = path.read_text(encoding="utf-8")
            for s in ("Valuation Summary", "Football Field", "Key Valuation Assumptions", "DCF Sensitivity",
                      "Appendix F – Valuation Detail", "Appendix G – Valuation Assumptions",
                      "Test Treasury source", "Kroll", "model fair value", "Not a recommendation"):
                self.assertIn(s, page)
            self.assertNotIn("BUY", page.upper().replace("BUYBACK", ""))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
