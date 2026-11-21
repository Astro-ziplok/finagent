"""
tests/test_research_report.py — Tests for the HTML research report
===================================================================
Uses synthetic prices and statements; the language model is mocked,
so no network or API calls are made.

Run with:  python -m pytest tests/ -v
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_analysis import _make_cleaned_df  # noqa: E402


def _fake_yf(scale=1.0):
    """Minimal yfinance-shaped fundamentals for one company (most recent first)."""
    years = ["2025-09-30", "2024-09-30", "2023-09-30"]
    income = [{"period": y, "total_revenue": (400 - 30 * i) * 1e9 * scale,
               "gross_profit": (190 - 15 * i) * 1e9 * scale, "operating_income": (130 - 10 * i) * 1e9 * scale,
               "net_income": (110 - 8 * i) * 1e9 * scale, "diluted_eps": 7 - i * 0.5,
               "gross_margin_pct": 47.5 - i, "operating_margin_pct": 32.5 - i, "net_margin_pct": 27.5 - i}
              for i, y in enumerate(years)]
    balance = [{"period": y, "cash": 35e9, "total_debt": (90 + 5 * i) * 1e9, "net_cash": -(55 + 5 * i) * 1e9,
                "shareholder_equity": 70e9, "total_assets": 360e9, "total_liabilities": 290e9,
                "debt_to_equity_x": round((90 + 5 * i) / 70, 2), "current_ratio_x": 1.0}
               for i, y in enumerate(years)]
    cashflow = [{"period": y, "operating_cf": (120 - 5 * i) * 1e9, "capex": -10e9,
                 "free_cash_flow": (110 - 5 * i) * 1e9, "share_buybacks": -90e9, "dividends_paid": -15e9,
                 "financing_cf": -100e9, "capex_pct_of_operating_cf": 8.5}
                for i, y in enumerate(years)]
    metrics = {"name": "Example Corp", "sector": "Technology", "industry": "Hardware",
               "profile": {"summary": "Example Corp designs and sells devices and services.", "employees": 1000},
               "valuation": {"current_price": 200.0, "pe_trailing": 35.0, "pe_forward": 30.0,
                             "ps_ratio": 9.0, "pb_ratio": 40.0, "beta": 1.1, "market_cap_fmt": "$3.00T"},
               "earnings": {"eps_ttm": 7.0, "eps_forward": 8.0, "eps_growth_pct": 14.29},
               "growth": {"revenue_yoy_pct": 8.0},
               "profitability": {"gross_margin_pct": 47.5, "profit_margin_pct": 27.5, "fcf_yield_pct": 3.4},
               "financial_health": {"debt_to_equity_x": 1.29, "current_ratio_x": 1.0},
               "analyst": {"target_price": 220.0, "target_low": 170.0, "target_high": 260.0, "upside_pct": 10.0},
               "dividend": {"dividend_yield_pct": 0.5}}
    return {"income": {"annual": income, "quarterly": [], "yoy_growth": {"revenue_growth_pct": 8.1}},
            "balance": {"annual": balance}, "cashflow": {"annual": cashflow}, "metrics": metrics}


THESIS = {"title": "AAA: STEADY COMPOUNDER AT A PREMIUM", "intro": "Intro text (Fig.01).",
          "overall_view": "Balanced", "view_rationale": "Because numbers.",
          "theses": [{"heading": "Thesis one", "paragraphs": [{"lead": "Lead", "text": "Body (Fig.02) and (Fig.99)."}]}]}
SECTIONS = {
    "business": {"intro": "Biz intro.", "blocks": [{"heading": "Products", "paragraphs": [{"lead": "L", "text": "T"}]}]},
    "performance": {"intro": "", "blocks": []},
    "financial": {"intro": "", "blocks": [{"heading": "Margins", "paragraphs": [{"lead": "M", "text": "x (vs_benchmark)."}]}]},
    "valuation": {"intro": "", "blocks": []},
    "risks": [{"code": "V1", "category": "Valuation risk", "title": "Premium multiple", "text": "t",
               "impact": "i", "mitigant": "m", "likelihood": "Medium", "impact_level": "High"}],
}


class TestResearchReport(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cleaned = {"AAA": _make_cleaned_df(seed=1), "BBB": _make_cleaned_df(seed=2)}
        self.bench = _make_cleaned_df(seed=3)
        self.yf = {"AAA": _fake_yf(), "BBB": _fake_yf(0.5)}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, tickers, llm_returns):
        """llm_returns = (thesis_reply, sections_reply); sections may be requested in 1 or 2 calls."""
        from src.report.research_report import build_research_report
        thesis_reply, sections_reply = llm_returns

        def fake_llm(prompt, *args, **kwargs):
            return thesis_reply if "Write the investment thesis" in prompt else sections_reply

        with patch("src.report.research_report._call_groq_json", side_effect=fake_llm):
            return build_research_report(tickers, self.cleaned, self.yf, self.bench, out_root=self.tmp,
                                         use_valuation=False)

    def test_single_company_report_structure(self):
        path = self._build(["AAA"], [THESIS, SECTIONS])
        page = path.read_text(encoding="utf-8")
        for name in ("Investment Summary", "Business Description", "Financial Analysis",
                     "Valuation", "Investment Risks", "Appendix B", "Appendix E"):
            self.assertIn(name, page)
        self.assertIn("AAA: STEADY COMPOUNDER AT A PREMIUM", page)
        figs = sorted(path.parent.glob("fig*.png"))
        self.assertGreaterEqual(len(figs), 8)          # market + fundamentals + risk matrix
        for f in figs:                                  # every figure file is referenced
            self.assertIn(f'src="{f.name}"', page)
        self.assertIn('href="#fig01"', page)            # valid citation becomes a link
        self.assertNotIn("Fig.99", page)                # invalid citation removed
        self.assertNotIn("vs_benchmark", page)          # leaked field name removed
        self.assertIn("not investment advice", page)
        data = json.loads((path.parent / "data.json").read_text(encoding="utf-8"))
        self.assertIn("risk_flags", data["facts"]["AAA"])

    def test_html_is_compact_for_reading(self):
        """Figures are separate PNG files, so the HTML itself stays small."""
        path = self._build(["AAA"], [THESIS, SECTIONS])
        self.assertLess(path.stat().st_size, 60_000)
        self.assertNotIn("base64", path.read_text(encoding="utf-8"))

    def test_comparison_report(self):
        path = self._build(["AAA", "BBB"], [THESIS, SECTIONS])
        page = path.read_text(encoding="utf-8")
        self.assertIn("Risk vs Return", page)
        self.assertIn("<th>AAA</th><th>BBB</th>", page)  # key metrics side by side

    def test_llm_failure_falls_back_to_data_report(self):
        path = self._build(["AAA"], [{}, {}])
        page = path.read_text(encoding="utf-8")
        self.assertIn("AI narrative unavailable", page)
        self.assertIn("Appendix B", page)                # tables still present

    def test_data_only_mode_makes_no_llm_calls(self):
        from src.report.research_report import build_research_report
        with patch("src.report.research_report._call_groq_json") as m:
            build_research_report(["AAA"], self.cleaned, self.yf, self.bench, out_root=self.tmp, use_llm=False,
                                  use_valuation=False)
        m.assert_not_called()


class TestReportHelpers(unittest.TestCase):

    def test_render_text_escapes_and_links(self):
        from src.report.research_report import _render_text
        out = _render_text("**Up** <script> (Fig.1, Fig.7) & (balance sheet facts).", {"Fig.01"})
        self.assertIn("<strong>Up</strong>", out)
        self.assertIn("&lt;script&gt;", out)
        self.assertIn('href="#fig01"', out)
        self.assertNotIn("Fig.07", out)
        self.assertNotIn("facts", out)

    def test_render_text_strips_dotted_field_names_and_links_figure_word(self):
        from src.report.research_report import _render_text
        out = _render_text("up 44.7% (growth.revenue_yoy_pct). Figure 04 shows it; Sharpe 1.5 (good).", {"Fig.04"})
        self.assertNotIn("revenue_yoy_pct", out)
        self.assertIn('href="#fig04"', out)
        self.assertIn("(good)", out)                    # ordinary parentheses are kept

    def test_quote_vs_history_mismatch_creates_data_note(self):
        from src.analysis.ai_analysis import _build_data_context
        from src.report.research_report import _collect_facts
        cleaned = {"AAA": _make_cleaned_df(seed=1)}
        ctx = _build_data_context(cleaned)
        yf = _fake_yf()
        yf["metrics"]["valuation"]["current_price"] = ctx["AAA"]["price_summary"]["end_price"] * 0.85
        notes = _collect_facts(["AAA"], ctx, cleaned, {"AAA": yf}, None)["AAA"]["data_notes"]
        self.assertEqual(len(notes), 1)
        self.assertIn("-15.0%", notes[0])

    def test_risk_flags_are_data_driven(self):
        from src.analysis.ai_analysis import _build_data_context
        from src.report.research_report import _collect_facts
        cleaned = {"AAA": _make_cleaned_df(seed=1)}
        ctx = _build_data_context(cleaned)
        flags = _collect_facts(["AAA"], ctx, cleaned, {"AAA": _fake_yf()}, None)["AAA"]["risk_flags"]
        self.assertTrue(any("Valuation premium" in f for f in flags))   # P/E 35x > 30x


if __name__ == "__main__":
    unittest.main()
