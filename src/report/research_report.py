"""
src/report/research_report.py — Equity research report (HTML)
==============================================================
Builds one coherent research note, laid out like a CFA Institute Research
Challenge report: numbered figures in a left sidebar, the argument on the
right, red section bars, and financial tables in the appendix.

How the report keeps a clear direction:
  1. All numbers, comparisons and risk flags are computed in Python (facts).
  2. The language model first writes ONE investment thesis: a headline,
     an overall view and three supporting arguments.
  3. Every other section is then written to support or test that thesis,
     citing the numbered figures, instead of being an independent dump.

Output (one folder per report, easy for people and for Claude to read):
    reports/research/<TICKERS>_<date>/report.html   ← open in a browser, Ctrl+P to save PDF
    reports/research/<TICKERS>_<date>/figNN.png     ← sidebar figures
    reports/research/<TICKERS>_<date>/data.json     ← every fact the report used

Usage:
    from src.report.research_report import build_research_report
    path = build_research_report(["AAPL"], cleaned_data, yf_data, benchmark_df)
"""

import html
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (REPORTS_DIR, BENCHMARK_TICKER, RISK_FREE_RATE, GROQ_TPM_LIMIT,
                    REASONING_TOKEN_BUFFER, GROQ_EXTRA_BODY)
from src.analysis.ai_analysis import (_build_data_context, _calendar_breakdown, _call_groq_json,
                                      _metrics_facts, _income_facts, _balance_facts,
                                      _cashflow_facts, _fmt_big, DISCLAIMER, NUMERIC_RULES)
from src.report import figures as F
from src.report import valuation_section as VS
from src.valuation.engine import value_company

logger = logging.getLogger(__name__)

RESEARCH_DIR = REPORTS_DIR / "research"
SOURCE_MARKET = f"Source: Yahoo Finance ({BENCHMARK_TICKER} as benchmark), FinAgent analysis"
SOURCE_FIN = "Source: Company filings via Yahoo Finance, FinAgent analysis"

# Section order mirrors the reference report
SECTIONS = [
    ("summary",     "Investment Summary"),
    ("business",    "Business Description"),
    ("performance", "Market Performance & Risk Profile"),
    ("financial",   "Financial Analysis"),
    ("valuation",   "Valuation"),
    ("risks",       "Investment Risks"),
]


@dataclass
class Figure:
    fid: str          # "Fig.01"
    section: str      # which section's sidebar it sits in
    title: str        # shown in the figure's title bar
    file: str         # "fig01.png"
    shows: str        # one-line description given to the language model
    source: str
    html: str = ""    # sidebar table (used instead of an image when set)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Facts (everything numeric is computed here, never by the LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _round(obj, nd=2):
    """Round floats recursively so the facts JSON stays compact."""
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, (np.floating,)):
        return round(float(obj), nd)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _max_drawdown(close: pd.Series) -> dict:
    dd = close / close.cummax() - 1
    trough = dd.idxmin()
    peak = close[:trough].idxmax()
    return {"max_drawdown_pct": round(float(dd.min()) * 100, 2),
            "peak_date": str(peak.date()), "trough_date": str(trough.date()),
            "current_drawdown_pct": round(float(dd.iloc[-1]) * 100, 2)}


def _risk_flags(c: dict, m: dict, dd: dict, inc_f: dict, bal_f: dict, cf_annual: list) -> list:
    """Data-driven risk candidates. The narrative may only discuss risks from this list."""
    flags = []
    r, v = c["returns"], (m.get("valuation") or {})
    fh, an = (m.get("financial_health") or {}), (m.get("analyst") or {})
    if r["annualised_volatility_pct"] > 35:
        flags.append(f"High volatility: annualised volatility {r['annualised_volatility_pct']}%")
    if (v.get("beta") or 0) > 1.2:
        flags.append(f"High market sensitivity: Yahoo beta {v['beta']}")
    if dd["max_drawdown_pct"] < -25:
        flags.append(f"Deep drawdown: {dd['max_drawdown_pct']}% from {dd['peak_date']} to {dd['trough_date']}")
    if (v.get("pe_trailing") or 0) > 30:
        flags.append(f"Valuation premium: trailing P/E {v['pe_trailing']}x vs typical market 20-25x")
    if v.get("pe_trailing") is None and v.get("pe_forward") is None:
        flags.append("No meaningful P/E (earnings may be negative)")
    if (fh.get("debt_to_equity_x") or 0) > 1.5:
        flags.append(f"High leverage: debt/equity {fh['debt_to_equity_x']}x")
    if fh.get("current_ratio_x") is not None and fh["current_ratio_x"] < 1:
        flags.append(f"Tight liquidity: current ratio {fh['current_ratio_x']}x")
    if cf_annual and (cf_annual[0].get("free_cash_flow") or 0) < 0:
        flags.append(f"Negative free cash flow in the latest year: {_fmt_big(cf_annual[0]['free_cash_flow'])}")
    for k in ("gross_margin_pct", "operating_margin_pct", "net_margin_pct"):
        f = inc_f.get(f"{k}_latest_vs_prior_year")
        if f and f["direction"] == "compressed":
            flags.append(f"Margin compression: {k.replace('_pct', '').replace('_', ' ')} "
                         f"{f['prior']}% -> {f['latest']}%")
    rev = inc_f.get("annual_revenue_changes") or []
    if rev and rev[-1]["direction"] == "decreased":
        flags.append(f"Revenue declined {rev[-1]['change_pct']}% in the latest fiscal year")
    if (an.get("upside_pct") or 0) < 0:
        flags.append(f"Price above analyst mean target ({an['upside_pct']}% to target)")
    vb = c.get("vs_benchmark") or {}
    if vb and not vb.get("outperformed_benchmark"):
        flags.append(f"Underperformed {vb['benchmark']} by {abs(vb['excess_return_pct'])} percentage points")
    if c["short_term_momentum"]["recent_volatility_vs_full_period"] == "higher":
        flags.append("Recent 30-day volatility is above the full-period average")
    if r["sharpe_ratio_annualised"] < 0.5:
        flags.append(f"Weak risk-adjusted return: Sharpe {r['sharpe_ratio_annualised']}")
    if bal_f.get("leverage_assessment", "").startswith("debt-to-equity not meaningful"):
        flags.append("Shareholder equity is zero or negative")
    return flags


def _collect_facts(tickers, ctx, cleaned, yf_data, rss_news, valuations=None) -> dict:
    facts = {}
    for t in tickers:
        c, d = ctx[t], yf_data.get(t) or {}
        m = d.get("metrics") or {}
        inc, bal, cf = d.get("income") or {}, d.get("balance") or {}, d.get("cashflow") or {}
        inc_f = _income_facts(inc.get("annual") or [], inc.get("quarterly") or [])
        bal_f = _balance_facts(bal.get("annual") or [])
        cf_f = _cashflow_facts(cf.get("annual") or [])
        dd = _max_drawdown(cleaned[t]["Close"])
        prof = m.get("profile") or {}
        notes = []
        quote = (m.get("valuation") or {}).get("current_price")
        last = c["price_summary"]["end_price"]
        if quote and last and abs(quote / last - 1) > 0.03:
            notes.append(f"Yahoo's live quote (${quote:,.2f}) differs from the last close in the price "
                         f"history (${last:,.2f} on {c['period']['end']}) by {(quote / last - 1) * 100:+.1f}%. "
                         "Valuation multiples and analyst upside use the live quote; price performance, "
                         "returns and drawdown use the price history.")
        headlines = []
        if rss_news is not None and not getattr(rss_news, "empty", True) and "ticker" in rss_news:
            for _, row in rss_news[rss_news["ticker"] == t].head(5).iterrows():
                headlines.append(f"[{str(row.get('published_at', ''))[:10]}] {row.get('title', '')}")
        facts[t] = _round({
            "profile": {"name": m.get("name", t), "sector": m.get("sector"),
                        "industry": m.get("industry"), "employees": prof.get("employees"),
                        "country": prof.get("country"), "business_summary": prof.get("summary")},
            "period": c["period"],
            "price": c["price_summary"],
            "returns": c["returns"],
            "momentum": c["short_term_momentum"],
            "technical": {k: c["technical_signals"][k] for k in ("ma_signal", "bb_signal")},
            "quarterly_returns": [(q["period"], q["pct_change"], "partial" if q["partial_period"] else "")
                                  for q in c["quarterly_breakdown"]],
            "strongest_quarter": c.get("strongest_quarter"),
            "weakest_quarter": c.get("weakest_quarter"),
            "vs_benchmark": c.get("vs_benchmark"),
            "drawdown": dd,
            "valuation": m.get("valuation"), "earnings": m.get("earnings"),
            "growth": m.get("growth"), "profitability": m.get("profitability"),
            "financial_health": m.get("financial_health"), "analyst": m.get("analyst"),
            "dividend": m.get("dividend"),
            "metric_assessments": _metrics_facts(m) if m else {},
            "income_statement_facts": inc_f,
            "balance_sheet_facts": bal_f,
            "cash_flow_facts": cf_f,
            "risk_flags": _risk_flags(c, m, dd, inc_f, bal_f, cf.get("annual") or []),
            "recent_headlines": headlines,
            "data_notes": notes,
            "valuation": VS.valuation_facts((valuations or {}).get(t)),
        })
        v = (valuations or {}).get(t) or {}
        if v.get("upside") is not None and v["upside"] < -0.15:
            facts[t]["risk_flags"].append(
                f"Price is {abs(v['upside']) * 100:.0f}% above the model fair value (${v['fair_value']:,.2f})")
        if v.get("dcf") and v["dcf"]["terminal_share_of_ev"] > 0.7:
            facts[t]["risk_flags"].append(
                f"DCF depends heavily on the long run: terminal value is "
                f"{v['dcf']['terminal_share_of_ev'] * 100:.0f}% of enterprise value")
    return facts


# ─────────────────────────────────────────────────────────────────────────────
#  2. Figures
# ─────────────────────────────────────────────────────────────────────────────

def _make_figures(tickers, ctx, cleaned, yf_data, benchmark_df, out_dir: Path, valuations=None) -> list:
    """Draw every figure the data supports and number them in reading order."""
    single = len(tickers) == 1
    t0 = tickers[0]
    bl = BENCHMARK_TICKER
    bench = None
    if benchmark_df is not None and not benchmark_df.empty:
        start = min(cleaned[t].index[0] for t in tickers)
        bench = benchmark_df[benchmark_df.index >= start]
    yf = {t: yf_data.get(t) or {} for t in tickers}
    bench_q = _calendar_breakdown(bench, "Q", 8) if (bench is not None and len(bench) > 20) else []

    specs = [  # (section, title, shows, source, draw function)
        ("summary", f"Share Price vs {bl} [Rebased, 100 = Start]",
         "share price(s) rebased to 100 against the benchmark over the period", SOURCE_MARKET,
         lambda p: F.price_vs_benchmark({t: cleaned[t] for t in tickers}, bench, bl, p)),
        ("summary", (f"{t0} vs {bl} Quarterly Returns [%]" if single else "Risk vs Return Over the Period [%]"),
         ("calendar-quarter returns of the stock vs the benchmark" if single
          else "annualised volatility vs total return for each ticker"), SOURCE_MARKET,
         (lambda p: F.quarterly_returns(ctx[t0]["quarterly_breakdown"], bench_q, t0, bl, p)) if single
         else (lambda p: F.risk_return(ctx, p))),
        ("summary", "Analyst Target Range vs Current Price [%]",
         "analyst low / mean / high target as % from the current price", "Source: Yahoo Finance analyst consensus",
         lambda p: F.analyst_targets(yf, p)),
        ("business", ("Revenue, Net Income [USD bn] and Net Margin [%]" if single
                      else "Revenue Growth and Net Margin, Latest FY [%]"),
         ("annual revenue and net income with net margin" if single
          else "latest fiscal-year revenue growth and net margin per ticker"), SOURCE_FIN,
         lambda p: F.revenue_net_income(yf, p)),
        ("performance", "Drawdown From Peak [%]",
         "percentage fall from the running price peak, vs the benchmark", SOURCE_MARKET,
         lambda p: F.drawdown({t: cleaned[t] for t in tickers}, bench, bl, p)),
        ("financial", "Margin Trends [%]" if single else "Margins, Latest FY [%]",
         "gross, operating and net margins" + (" by fiscal year" if single else " per ticker"), SOURCE_FIN,
         lambda p: F.margins(yf, p)),
        ("financial", "Operating Cash Flow, Capex and FCF [USD bn]" if single else "Free Cash Flow Yield [%]",
         ("operating cash flow, capex and free cash flow by fiscal year" if single
          else "free cash flow yield per ticker vs a 3% reference"), SOURCE_FIN,
         lambda p: F.cash_flow(yf, p)),
        ("financial", "Cash vs Total Debt [USD bn] and Debt/Equity [x]" if single
         else "Debt/Equity and Current Ratio [x]",
         ("cash, total debt and debt-to-equity by fiscal year" if single
          else "debt-to-equity and current ratio per ticker"), SOURCE_FIN,
         lambda p: F.balance_sheet(yf, p)),
        ("financial", "Shareholder Returns vs Free Cash Flow [USD bn]",
         "buybacks and dividends stacked against free cash flow by fiscal year", SOURCE_FIN,
         lambda p: F.shareholder_returns(yf, p)),
    ]
    vals = {t: v for t, v in (valuations or {}).items() if v}
    if vals:
        v0 = vals.get(t0)
        specs += [
            ("valuation", "Valuation Summary", "each method's value per share, its weight, the weighted fair "
             "value and implied upside/downside", "Source: FinAgent valuation (Appendix F)", VS.summary_table(vals)),
            ("valuation", "Valuation Football Field" + (" [% vs price]" if not single else " [USD/share]"),
             "value range of every method against today's price and the 52-week range",
             "Source: FinAgent valuation; Yahoo Finance prices", lambda p: F.football_field(vals, p)),
            ("valuation", "Key Valuation Assumptions", "risk-free rate, ERP, beta, cost of equity, WACC, "
             "terminal growth, horizon", "Sources: see Appendix G", VS.key_assumptions(vals)),
        ]
        if single and v0:
            specs += [
                ("valuation", "DCF Sensitivity: WACC vs Terminal Growth", "DCF value per share across WACC "
                 "and terminal-growth assumptions", "Source: FinAgent valuation", VS.sensitivity_table(v0)),
                ("valuation", f"Peer Regression: {(v0.get('regression') or {}).get('y_label', '')} vs "
                 f"{(v0.get('regression') or {}).get('x_label', '')}",
                 "peers' multiple against their fundamentals with the fitted line; the company highlighted",
                 "Source: Yahoo Finance, FinAgent analysis",
                 lambda p: F.peer_regression(v0.get("regression"), t0, p)),
                ("valuation", "Trailing P/E History [x]", "the company's own trailing P/E with mean and "
                 "+/-1 standard deviation", "Source: Yahoo Finance prices and reported EPS, FinAgent analysis",
                 lambda p: F.pe_history(v0.get("history"), p)),
            ]
    out, n = [], 0
    for section, title, shows, source, draw in specs:
        if isinstance(draw, str):          # ready-made sidebar table
            if draw:
                n += 1
                out.append(Figure(f"Fig.{n:02d}", section, title, "", shows, source, html=draw))
            continue
        fname = f"fig{n + 1:02d}.png"
        try:
            ok = draw(out_dir / fname)
        except Exception as e:  # one bad chart must not break the report
            logger.warning(f"Figure '{title}' skipped: {e}")
            ok = False
        if ok:
            n += 1
            out.append(Figure(f"Fig.{n:02d}", section, title, fname, shows, source))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  3. Narrative (two LLM calls: thesis first, then sections built on it)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_REPORT = """
You are a senior equity research analyst writing a CFA Institute Research Challenge-style
report for a university finance course. Write like a sell-side research note: confident,
specific and argued. Each paragraph opens with a short lead-in that states the point,
followed by evidence with exact numbers.
Hard rules:
- Use ONLY facts from DATA. Never invent figures, events, products, market shares, peers or news.
- The business description must come from profile.business_summary only.
- Cite figures as (Fig.NN) using only the IDs in FIGURES, where they support the point.
- Do not issue buy/sell ratings or price targets of your own. You may cite the analyst consensus in DATA.
- valuation.fair_value is a model estimate, not a target: describe it as "model fair value" with its range,
  say which assumptions drive it (cost of capital, terminal value share, peer group), and use the reverse
  DCF to explain what growth today's price requires. If valuation_gap_note is present, explain it.
- Never mention DATA field names or cite DATA in parentheses (no "(vs_benchmark)", "(analyst)",
  "(balance sheet facts)"). Write for a human reader; parentheses are only for figure citations.
- If data_notes are present, follow them and do not compare numbers based on different prices
  as if they were the same price.
- Two different betas exist. Always say which one you mean:
  "1-year daily beta vs the benchmark" (vs_benchmark.beta_vs_benchmark) or
  "Yahoo 5-year monthly beta" (valuation.beta, interpreted by metric_assessments.beta_interpretation).
- Return valid JSON only, exactly in the requested shape.
"""

PARAGRAPHS_SHAPE = '[{"lead": "short bold lead-in, 3-8 words", "text": "60-110 words"}]'


def _compact_for_llm(facts: dict, level: int = 1) -> dict:
    """
    Shorter version of the facts for the language model (data.json keeps the full
    version). Groq's free tier caps prompt + output at 8,000 tokens per minute, so
    year-by-year change records become one line each, e.g.
    "FY24: $26.97B -> $60.92B (+125.9%)", and long text fields are trimmed.
    """
    def compact(v):
        if isinstance(v, list) and v and all(isinstance(x, dict) and "change_pct" in x and "to_period" in x for x in v):
            out = []
            for x in v:
                chg = x.get("change_pct")
                chg_txt = "n/a" if chg is None else f"{chg:+.1f}%"
                flat = ", roughly flat" if x.get("direction") == "roughly flat" else ""
                out.append(f"FY{str(x['to_period'])[:4]}: {x.get('from')} -> {x.get('to')} ({chg_txt}{flat})")
            return out
        if isinstance(v, dict):
            return {k: compact(val) for k, val in v.items()}
        return v

    slim = {}
    for t, f in facts.items():
        g = compact(f)
        prof = dict(g.get("profile") or {})
        if prof.get("business_summary"):
            prof["business_summary"] = prof["business_summary"][:550]
        g["profile"] = prof
        g["recent_headlines"] = [h[:110] for h in (g.get("recent_headlines") or [])[:3]]
        g["returns"] = {k: v for k, v in (g.get("returns") or {}).items()
                        if k not in ("mean_daily_pct", "std_daily_pct", "risk_free_rate_used_pct")}
        g["momentum"] = {k: v for k, v in (g.get("momentum") or {}).items() if k != "last_30d_avg_volume"}
        if level >= 2:
            g = _tighter(g)
        slim[t] = g
    return slim


def _tighter(g: dict) -> dict:
    """
    Level-2 compaction for comparison reports: one line per valuation method,
    only the latest two years of statement changes, no low-value context.
    """
    for k in ("recent_headlines", "technical", "quarterly_returns", "momentum", "period", "growth"):
        g.pop(k, None)
    g["profile"] = {**(g.get("profile") or {}),
                    "business_summary": ((g.get("profile") or {}).get("business_summary") or "")[:300]}
    inc = g.get("income_statement_facts") or {}
    g["income_statement_facts"] = {k: (v[-2:] if isinstance(v, list) else v) for k, v in inc.items()
                                   if not k.startswith("latest_quarter_vs_prior")}
    cf = g.get("cash_flow_facts") or {}
    g["cash_flow_facts"] = {k: (v[-2:] if isinstance(v, list) else v) for k, v in cf.items()
                            if not k.startswith("capex_")}
    bal = g.get("balance_sheet_facts") or {}
    g["balance_sheet_facts"] = {k: v for k, v in bal.items()
                                if k in ("latest_cash_vs_total_debt", "leverage_assessment", "retained_earnings_note")}
    v = dict(g.get("valuation") or {})
    if v.get("methods"):
        v["methods"] = {k: (f"{m['value']} ({m['role']}, weight {m['weight_pct']}%, range {m['range'][0]}-{m['range'][1]})"
                            if "value" in m else f"not used: {m['not_used_because'][:90]}")
                        for k, m in v["methods"].items()}
    v.pop("peer_multiples", None)
    g["valuation"] = v
    return g


def _thesis_prompt(label, facts, figs, benchmark):
    return f"""
Write the investment thesis for {label}. Benchmark: {benchmark}. Risk-free rate used: {RISK_FREE_RATE * 100:.1f}%.

DATA:
{json.dumps(facts, separators=(",", ":"), default=str)}

FIGURES:
{json.dumps([{"id": f.fid, "section": f.section, "shows": f.shows} for f in figs], separators=(",", ":"))}

Return JSON with this shape:
{{
  "title": "'{label}: ' followed by a 4-8 word theme in UPPER CASE",
  "intro": "2-3 sentences: what the company is and the core argument of this report",
  "overall_view": "Constructive | Balanced | Cautious",
  "view_rationale": "one sentence citing two numbers",
  "theses": [{{"heading": "thesis headline, 5-10 words", "paragraphs": {PARAGRAPHS_SHAPE}}}]
}}
Exactly 3 theses with 1-2 paragraphs each. The three must form ONE argument that supports
overall_view; one thesis must address valuation (model fair value vs price, and what the reverse DCF
implies); one must address the strongest counter-argument from risk_flags.
"""


SECTION_KEYS = ("business", "performance", "financial", "valuation", "risks")


def _sections_prompt(label, facts, figs, thesis, keys=SECTION_KEYS):
    block = f'{{"intro": "1-2 sentences", "blocks": [{{"heading": "sub-heading, 3-8 words", "paragraphs": {PARAGRAPHS_SHAPE}}}]}}'
    risk_shape = ('[{"code": "M1", "category": "Market risk | Valuation risk | Financial risk | Business risk", '
                  '"title": "short risk title", "text": "40-80 words", '
                  '"impact": "what it would do to returns, 1-2 sentences", '
                  '"mitigant": "what offsets it, from DATA only, 1-2 sentences", '
                  '"likelihood": "Low | Medium | High", "impact_level": "Low | Medium | High"}]')
    shape = ",\n".join(f'  "{k}": {risk_shape if k == "risks" else block}' for k in keys)
    guidance = {
        "business": "- business: 1-2 blocks, based on profile.business_summary (what it sells, segments, customers).",
        "performance": "- performance: 2 blocks: returns vs benchmark and quarters; volatility, drawdown, beta, Sharpe.",
        "financial": "- financial: 3-4 blocks: revenue & earnings, margins, cash flow & shareholder returns, balance sheet.",
        "valuation": ("- valuation: 3-4 blocks: (1) why these methods (valuation.why_these_methods, methods not used),\n"
                      "  (2) DCF result, terminal value share and the reverse DCF versus consensus growth,\n"
                      "  (3) peer multiples / regression / own P/E history, naming the peer group and why,\n"
                      "  (4) weighted model fair value and range vs price, and the assumptions it is most sensitive to\n"
                      "  (use dcf_sensitivity numbers exactly). Valuation blocks may have 1-2 paragraphs each."),
        "risks": ("- risks: 4-6 risks taken from risk_flags. Codes: M = market, V = valuation, F = financial, B = business,\n"
                  "  numbered M1, M2, V1... Order from most to least important."),
    }
    return f"""
Write these sections of the research report on {label}: {", ".join(keys)}.

THESIS (already written; every section must support or test it, no generic filler).
The Investment Summary already states these points with headline numbers. Do NOT restate them:
each section goes one level deeper with additional data (other years, quarters, ratios, drivers).
{json.dumps(thesis, separators=(",", ":"))}

DATA:
{json.dumps(facts, separators=(",", ":"), default=str)}

FIGURES:
{json.dumps([{"id": f.fid, "section": f.section, "shows": f.shows} for f in figs], separators=(",", ":"))}

Return JSON with exactly this shape:
{{
{shape}
}}
Guidance:
{chr(10).join(guidance[k] for k in keys)}
"""


def _output_budget(prompt: str, desired: int) -> int:
    """
    Largest output budget that keeps prompt + output under Groq's tokens-per-minute
    cap (the free tier counts both, and rejects a single request above the cap).
    Token count is estimated from characters (about 4 characters per token, slightly conservative).
    """
    est_prompt = int(len(SYSTEM_REPORT + NUMERIC_RULES + prompt) / 4.0) + 150  # Groq measured ~4.6 chars/token
    reserve = REASONING_TOKEN_BUFFER if GROQ_EXTRA_BODY else 0
    return max(0, min(desired, GROQ_TPM_LIMIT - est_prompt - reserve - 200))


def _thesis_brief(thesis: dict) -> dict:
    """The thesis without its paragraphs: enough to keep sections on-message, far fewer tokens."""
    return {"title": thesis.get("title"), "overall_view": thesis.get("overall_view"),
            "view_rationale": thesis.get("view_rationale"),
            "theses": [{"heading": t.get("heading"),
                        "points": [p.get("lead") for p in (t.get("paragraphs") or []) if isinstance(p, dict)]}
                       for t in thesis.get("theses") or []]}


def _write_sections(label, facts, figs, thesis) -> dict:
    """All body sections in one call when it fits; otherwise split into two calls."""
    thesis = _thesis_brief(thesis)
    prompt = _sections_prompt(label, facts, figs, thesis)
    budget = _output_budget(prompt, 4200)
    if budget >= 3200:
        return _call_groq_json(prompt, SYSTEM_REPORT, max_tokens=budget)
    sections = {}
    for keys in (("business", "performance", "valuation"), ("financial", "risks")):
        p = _sections_prompt(label, facts, figs, thesis, keys)
        b = _output_budget(p, 2600)
        if b < 1200:
            logger.warning("Report data too large for the Groq token limit; try fewer tickers "
                           "or raise GROQ_TPM_LIMIT if your Groq tier allows it.")
            return {}
        sections.update(_call_groq_json(p, SYSTEM_REPORT, max_tokens=b))
    return sections if all(k in sections for k in SECTION_KEYS) else {}


def _fallback_narrative(label, facts) -> tuple:
    """Plain data-only narrative used when the language model is unavailable."""
    msg = ("AI narrative unavailable (language model call failed). "
           "Figures and appendix tables below are complete and data-driven.")
    thesis = {"title": f"{label}: DATA SUMMARY", "intro": msg, "overall_view": "", "view_rationale": "",
              "theses": []}
    risks = []
    for t, f in facts.items():
        for i, flag in enumerate(f.get("risk_flags", [])[:6], 1):
            risks.append({"code": f"R{i}", "category": "Risk flag", "title": f"{t}: {flag}", "text": "",
                          "impact": "", "mitigant": "", "likelihood": "", "impact_level": ""})
    sections = {k: {"intro": "", "blocks": []} for k in ("business", "performance", "financial", "valuation")}
    sections["risks"] = risks
    return thesis, sections


# ─────────────────────────────────────────────────────────────────────────────
#  4. HTML rendering
# ─────────────────────────────────────────────────────────────────────────────

# Every key in the facts given to the language model; filled by build_research_report.
# Any "(key)" the model echoes back is a data-field citation and is removed.
_FIELD_NAMES: set = set()


def _collect_keys(obj, out: set):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k).lower())
            _collect_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_keys(v, out)
    return out


def _render_text(text: str, valid_ids: set) -> str:
    """Escape LLM text, keep **bold**, and turn valid Fig.NN citations into links."""
    s = html.escape(str(text or ""))
    if _FIELD_NAMES:
        s = re.sub(r"\s*\(([A-Za-z0-9_. ]+)\)",
                   lambda m: "" if m.group(1).strip().lower() in _FIELD_NAMES else m.group(0), s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

    def fig(m):
        fid = f"Fig.{int(m.group(1)):02d}"
        return f'<a href="#{fid.replace(".", "").lower()}">{fid}</a>' if fid in valid_ids else ""

    # remove leaked data-field citations such as "(vs_benchmark)" or "(balance sheet facts)"
    s = re.sub(r"\s*\((?:[a-z ]*\bfacts?\b[a-z ]*|analyst|data)\)", "", s)
    # any parenthetical (without a figure) containing snake_case or dotted field names,
    # e.g. "(vs_benchmark.excess_return_pct)" or "(profile.business_summary)"
    s = re.sub(r"\s*\((?![^)]*Fig)[^)]*?(?:\b[a-z0-9]+_[a-z0-9_]+\b|\b[a-z_]+\.[a-z_]+\b)[^)]*\)", "", s)
    s = re.sub(r"\bFig(?:ure)?\.?\s?(\d{1,2})\b", fig, s)
    # tidy citations emptied by removed (invalid) figure ids
    s = re.sub(r"\(\s*(?:,|&amp;|and|\s)*\)", "", s)
    s = re.sub(r"\(\s*(?:,|&amp;|and)\s*", "(", s)
    s = re.sub(r"\s*(?:,|&amp;|and)\s*\)", ")", s)
    return re.sub(r"\s+([.,;])", r"\1", s).strip()


def _paragraphs(paras, valid) -> str:
    out = []
    for p in paras or []:
        if not isinstance(p, dict):
            p = {"lead": "", "text": str(p)}
        lead = _render_text(p.get("lead", ""), valid).rstrip(":")
        body = _render_text(p.get("text", ""), valid)
        out.append(f'<p>{f"<span class=lead>{lead}:</span> " if lead else ""}{body}</p>')
    return "\n".join(out)


def _section_body(sec: dict, valid) -> str:
    parts = []
    if sec.get("intro"):
        parts.append(f'<p class="sec-intro">{_render_text(sec["intro"], valid)}</p>')
    for b in sec.get("blocks") or []:
        if b.get("heading"):
            parts.append(f'<h3>{_render_text(b["heading"], valid)}</h3>')
        parts.append(_paragraphs(b.get("paragraphs"), valid))
    return "\n".join(parts)


def _risks_body(risks, valid) -> str:
    parts = []
    for r in risks or []:
        cat = html.escape(str(r.get("category", "")).upper())
        head = f'[{html.escape(str(r.get("code", "")))}] {cat} | {_render_text(r.get("title", ""), valid).upper()}'
        parts.append(f"<h3>{head}</h3>")
        if r.get("text"):
            parts.append(f"<p>{_render_text(r['text'], valid)}</p>")
        if r.get("impact"):
            parts.append(f'<p><span class=lead>Impact:</span> {_render_text(r["impact"], valid)}</p>')
        if r.get("mitigant"):
            parts.append(f'<p><span class=lead>Mitigant:</span> {_render_text(r["mitigant"], valid)}</p>')
    return "\n".join(parts)


def _figure_html(f: Figure) -> str:
    anchor = f.fid.replace(".", "").lower()
    body = f.html if f.html else f'<img src="{f.file}" alt="{html.escape(f.title)}">'
    return (f'<figure id="{anchor}"><div class="fig-title">{html.escape(f.fid)}: {html.escape(f.title)}</div>'
            f'{body}<figcaption>{html.escape(f.source)}</figcaption></figure>')


def _money(v):
    return "–" if v is None else f"{v / 1e9:,.2f}"


def _pct(v):
    return "–" if v is None else f"{v:.1f}%"


def _num(v, suffix=""):
    return "–" if v is None else f"{v:,.2f}{suffix}"


def _table(title: str, rows: list, lines: list) -> str:
    """rows: statement rows most-recent-first; lines: (label, key, formatter, bold)."""
    rows = list(reversed(rows or []))
    if not rows:
        return ""
    head = "".join(f"<th>FY{str(r['period'])[:4]}</th>" for r in rows)
    body = []
    for label, key, fmt, bold in lines:
        if all(r.get(key) is None for r in rows):
            continue
        cells = "".join(f"<td>{fmt(r.get(key))}</td>" for r in rows)
        body.append(f'<tr{" class=total" if bold else ""}><td>{label}</td>{cells}</tr>')
    return (f'<table class="fin"><caption>{html.escape(title)}</caption>'
            f"<thead><tr><th>Unit: USD bn</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>")


INCOME_LINES = [("Revenue", "total_revenue", _money, True), ("Gross profit", "gross_profit", _money, False),
                ("Operating income", "operating_income", _money, False), ("EBITDA", "ebitda", _money, False),
                ("R&amp;D expense", "rd_expense", _money, False), ("Net income", "net_income", _money, True),
                ("Diluted EPS (USD)", "diluted_eps", _num, False),
                ("Gross margin", "gross_margin_pct", _pct, False),
                ("Operating margin", "operating_margin_pct", _pct, False),
                ("Net margin", "net_margin_pct", _pct, False)]
BALANCE_LINES = [("Cash &amp; equivalents", "cash", _money, False), ("Total assets", "total_assets", _money, True),
                 ("Long-term debt", "long_term_debt", _money, False), ("Total debt", "total_debt", _money, False),
                 ("Total liabilities", "total_liabilities", _money, True),
                 ("Shareholders' equity", "shareholder_equity", _money, True),
                 ("Retained earnings", "retained_earnings", _money, False),
                 ("Net cash / (debt)", "net_cash", _money, False),
                 ("Debt / equity (x)", "debt_to_equity_x", _num, False),
                 ("Current ratio (x)", "current_ratio_x", _num, False)]
CASHFLOW_LINES = [("Operating cash flow", "operating_cf", _money, True), ("Capital expenditure", "capex", _money, False),
                  ("Free cash flow", "free_cash_flow", _money, True), ("Investing cash flow", "investing_cf", _money, False),
                  ("Financing cash flow", "financing_cf", _money, False), ("Share buybacks", "share_buybacks", _money, False),
                  ("Dividends paid", "dividends_paid", _money, False),
                  ("Capex / operating CF", "capex_pct_of_operating_cf", _pct, False)]


def _metrics_table(tickers, yf_data, ctx) -> str:
    lines = [
        ("Last close in price history (USD)", lambda t: _num(ctx[t]["price_summary"]["end_price"])),
        ("Live quote used for multiples (USD)",
         lambda t: _num((((yf_data.get(t) or {}).get("metrics") or {}).get("valuation") or {}).get("current_price"))),
        ("Market cap", lambda t: (((yf_data.get(t) or {}).get("metrics") or {}).get("valuation") or {}).get("market_cap_fmt", "–")),
        ("Total return over period", lambda t: _pct(ctx[t]["price_summary"]["total_return_pct"])),
        (f"Excess return vs {BENCHMARK_TICKER}", lambda t: _pct((ctx[t].get("vs_benchmark") or {}).get("excess_return_pct"))),
        ("Annualised volatility", lambda t: _pct(ctx[t]["returns"]["annualised_volatility_pct"])),
        ("Sharpe ratio (annualised)", lambda t: _num(ctx[t]["returns"]["sharpe_ratio_annualised"])),
        (f"Beta vs {BENCHMARK_TICKER} (period, daily)", lambda t: _num((ctx[t].get("vs_benchmark") or {}).get("beta_vs_benchmark"))),
    ]

    def m(grp, key, fmt):
        return lambda t: fmt((((yf_data.get(t) or {}).get("metrics") or {}).get(grp) or {}).get(key))

    lines += [
        ("P/E trailing (x)", m("valuation", "pe_trailing", _num)), ("P/E forward (x)", m("valuation", "pe_forward", _num)),
        ("P/S (x)", m("valuation", "ps_ratio", _num)), ("P/B (x)", m("valuation", "pb_ratio", _num)),
        ("PEG (x)", m("valuation", "peg_ratio", _num)), ("Beta (Yahoo, 5y monthly)", m("valuation", "beta", _num)),
        ("Revenue growth, last quarter YoY", m("growth", "revenue_yoy_pct", _pct)),
        ("Expected EPS growth (fwd vs TTM)", m("earnings", "eps_growth_pct", _pct)),
        ("Gross margin", m("profitability", "gross_margin_pct", _pct)),
        ("Operating margin", m("profitability", "operating_margin_pct", _pct)),
        ("Net margin", m("profitability", "profit_margin_pct", _pct)),
        ("ROE", m("profitability", "roe_pct", _pct)), ("ROA", m("profitability", "roa_pct", _pct)),
        ("FCF yield", m("profitability", "fcf_yield_pct", _pct)),
        ("Dividend yield", m("dividend", "dividend_yield_pct", _pct)),
        ("Debt / equity (x)", m("financial_health", "debt_to_equity_x", _num)),
        ("Current ratio (x)", m("financial_health", "current_ratio_x", _num)),
        ("Analyst mean target (USD)", m("analyst", "target_price", _num)),
        ("Implied upside to mean target", m("analyst", "upside_pct", _pct)),
    ]
    head = "".join(f"<th>{html.escape(t)}</th>" for t in tickers)
    body = "".join(f"<tr><td>{lbl}</td>{''.join(f'<td>{html.escape(str(fn(t)))}</td>' for t in tickers)}</tr>"
                   for lbl, fn in lines)
    return f'<table class="fin"><thead><tr><th>Metric</th>{head}</tr></thead><tbody>{body}</tbody></table>'


GLOSSARY = [
    ("Beta", "Sensitivity of the stock to market moves; above 1 moves more than the market."),
    ("Capex", "Capital expenditure: cash spent on property, plant, equipment and similar assets."),
    ("Drawdown", "Percentage fall from the highest price reached so far."),
    ("EPS", "Earnings per share."), ("FCF", "Free cash flow = operating cash flow - capex."),
    ("FCF yield", "Free cash flow / market capitalisation."), ("FY", "Fiscal year."),
    ("P/B, P/E, P/S", "Price to book value, to earnings, to sales."),
    ("PEG", "P/E divided by expected earnings growth."), ("ROE / ROA", "Return on equity / on assets."),
    ("Sharpe ratio", f"Annualised excess return over the risk-free rate ({RISK_FREE_RATE * 100:.1f}%) per unit of volatility."),
    ("TTM", "Trailing twelve months."), ("Volatility", "Annualised standard deviation of daily returns."),
    ("DCF / FCFF", "Discounted cash flow of free cash flow to the firm (before payments to lenders and shareholders)."),
    ("WACC", "Weighted average cost of capital: blended required return of shareholders and lenders."),
    ("CAPM", "Capital asset pricing model: cost of equity = risk-free rate + beta x equity risk premium."),
    ("ERP", "Equity risk premium: extra return investors require for owning stocks over Treasuries."),
    ("Terminal value", "Value of all cash flows after the forecast period, assuming steady growth."),
    ("Reverse DCF", "The growth rate that makes the DCF value equal today's share price."),
    ("Residual income", "Profit above the return shareholders require on book equity."),
    ("DDM", "Dividend discount model: value = present value of future dividends."),
    ("EV / EBITDA", "Enterprise value (equity + net debt) divided by earnings before interest, tax, depreciation and amortisation."),
]

CSS = """
:root { --red:#D71920; --deep:#8C1016; --dark:#262626; --grey:#595959; --line:#D9D9D9; }
* { box-sizing:border-box; }
body { margin:0; background:#fff; color:var(--dark);
       font-family: Arial, "Helvetica Neue", Helvetica, sans-serif; }
.page { max-width:1120px; margin:0 auto; padding:28px 28px 36px; }
.doc-title { font-size:30px; font-weight:800; margin:0 0 6px; letter-spacing:.2px; }
.intro { font-style:italic; font-size:13.5px; line-height:1.5; text-align:justify; margin:0 0 6px; }
.view { font-size:13px; margin:0 0 4px; }
.view b { color:var(--red); text-transform:uppercase; }
.note { font-size:11.5px; color:var(--grey); border-left:3px solid var(--red); padding:2px 8px; margin:4px 0; }
.section { display:grid; grid-template-columns:31% 1fr; gap:22px; margin-top:16px; }
.sidebar figure { margin:0 0 14px; break-inside:avoid; }
.fig-title { background:#3A3A3A; color:#fff; font-weight:700; font-size:11.5px;
             text-align:center; padding:4px 6px; }
figure img { width:100%; display:block; border:1px solid var(--line); border-top:0; }
figcaption { font-size:10px; font-style:italic; text-align:right; color:var(--grey); margin-top:2px; }
.bar { background:var(--red); color:#fff; font-weight:800; text-transform:uppercase;
       font-size:15px; letter-spacing:.4px; padding:4px 10px; margin:0 0 8px; }
.main h3 { color:var(--red); font-size:12.5px; font-weight:800; text-transform:uppercase;
           margin:12px 0 4px; }
.main p { font-size:13px; line-height:1.48; text-align:justify; margin:0 0 8px; }
.main .sec-intro { font-style:italic; }
.lead { font-weight:700; font-style:italic; }
a { color:var(--deep); }
.appendix { margin-top:28px; }
.app-bar { background:var(--red); color:#fff; font-weight:800; text-transform:uppercase;
           text-align:center; font-size:14px; padding:4px; margin:18px 0 8px; }
.table-wrap { overflow-x:auto; }
table.fin { width:100%; border-collapse:collapse; font-size:11.5px; margin:0 0 12px; }
table.fin caption { text-align:left; font-weight:800; font-size:12px; padding:4px 0; }
table.fin th { background:#3A3A3A; color:#fff; text-align:right; padding:4px 6px; white-space:nowrap; }
table.fin th:first-child, table.fin td:first-child { text-align:left; }
table.fin td { text-align:right; padding:3px 6px; border-bottom:1px solid #EFEFEF; white-space:nowrap; }
table.fin tr.total td { font-weight:700; border-top:1px solid var(--grey); }
dl.gloss { display:grid; grid-template-columns:140px 1fr; gap:3px 12px; font-size:12px; margin:0; }
dl.gloss dt { font-weight:700; } dl.gloss dd { margin:0; }
.method { font-size:12px; line-height:1.45; }
.footer { margin-top:24px; border-top:6px solid var(--red); padding-top:8px;
          font-size:10.5px; color:var(--grey); text-align:justify; }
@media (max-width:760px) { .section { grid-template-columns:1fr; } .doc-title { font-size:22px; } }
@media print {
  @page { size:A4; margin:11mm; }
  .page { max-width:none; padding:0; }
  .section { grid-template-columns:31% 1fr; }
  .appendix { break-before:page; }
  a { color:inherit; text-decoration:none; }
  * { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
}
"""


def _render_html(label, tickers, thesis, sections, figs, facts, ctx, yf_data, valuations=None) -> str:
    valid = {f.fid for f in figs}
    by_sec = {k: [f for f in figs if f.section == k] for k, _ in SECTIONS}
    parts = [f"<title>{html.escape(label)} Research Report</title><style>{CSS}{VS.VALUATION_CSS}</style><div class=page>"]

    title = thesis.get("title") or f"{label}: RESEARCH REPORT"
    parts.append(f"<h1 class=doc-title>{html.escape(title)}</h1>")
    if thesis.get("intro"):
        parts.append(f"<p class=intro>{_render_text(thesis['intro'], valid)}</p>")
    if thesis.get("overall_view"):
        parts.append(f"<p class=view>Overall view: <b>{html.escape(thesis['overall_view'])}</b>"
                     f" — {_render_text(thesis.get('view_rationale', ''), valid)}</p>")
    for t in tickers:
        v = (valuations or {}).get(t) or {}
        if v.get("fair_value") and v.get("price"):
            parts.append(f"<p class=fair>{html.escape(t)} model fair value: <b>${v['fair_value']:,.2f}</b> "
                         f"(range ${v['fair_low']:,.0f}–${v['fair_high']:,.0f}) vs price ${v['price']:,.2f}: "
                         f"<b>{v['upside'] * 100:+.1f}%</b> implied. Not a recommendation; see Valuation and Appendix G "
                         "for methods and assumptions.</p>")

    notes = [f"{t}: {n}" for t in tickers for n in (facts.get(t) or {}).get("data_notes", [])]
    for n in notes:
        parts.append(f"<p class=note>Data note — {html.escape(n)}</p>")

    for key, name in SECTIONS:
        if key == "summary":
            body = []
            for th in thesis.get("theses") or []:
                body.append(f"<h3>{_render_text(th.get('heading', ''), valid)}</h3>")
                body.append(_paragraphs(th.get("paragraphs"), valid))
            body = "\n".join(body)
        elif key == "risks":
            body = _risks_body(sections.get("risks"), valid)
        else:
            body = _section_body(sections.get(key) or {}, valid)
        if not body.strip() and not by_sec[key]:
            continue
        side = "".join(_figure_html(f) for f in by_sec[key])
        parts.append(f'<section class=section><aside class=sidebar>{side}</aside>'
                     f'<div class=main><div class=bar>{html.escape(name)}</div>{body}</div></section>')

    # Appendix — financial tables, glossary, methodology
    app = ['<div class=appendix><div class=app-bar>Appendix A – Glossary</div><dl class=gloss>']
    app += [f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in GLOSSARY]
    app.append("</dl>")
    for letter, name, key, lines in (("B", "Income Statement", "income", INCOME_LINES),
                                     ("C", "Balance Sheet", "balance", BALANCE_LINES),
                                     ("D", "Cash Flow Statement", "cashflow", CASHFLOW_LINES)):
        tables = [_table(f"{t}" if len(tickers) > 1 else "",
                         ((yf_data.get(t) or {}).get(key) or {}).get("annual"), lines) for t in tickers]
        tables = [x for x in tables if x]
        if tables:
            app.append(f"<div class=app-bar>Appendix {letter} – {name}</div>")
            app += [f"<div class=table-wrap>{x}</div>" for x in tables]
    app.append("<div class=app-bar>Appendix E – Key Metrics</div>")
    app.append(f"<div class=table-wrap>{_metrics_table(tickers, yf_data, ctx)}</div>")
    vals = {t: v for t, v in (valuations or {}).items() if v}
    if vals:
        app.append("<div class=app-bar>Appendix F – Valuation Detail</div>")
        app.append(VS.appendix_valuation(vals))
        app.append("<div class=app-bar>Appendix G – Valuation Assumptions &amp; Sources</div>")
        app.append(VS.appendix_sources(vals))
    p0 = ctx[tickers[0]]["period"]
    app.append("<div class=app-bar>Appendix H – Methodology &amp; Data Sources</div><div class=method>"
               f"<p>Price data: Yahoo Finance daily, adjusted for splits and dividends, {p0['start']} to {p0['end']}. "
               f"Benchmark: {BENCHMARK_TICKER}. Financial statements and analyst consensus: Yahoo Finance. "
               f"Sharpe ratio is annualised using a {RISK_FREE_RATE * 100:.1f}% risk-free rate. Quarters are "
               "calendar quarters measured from the previous quarter's close. Debt/equity uses total debt. "
               "Comparisons, rankings and risk flags are computed in code; the narrative is written by a "
               "language model constrained to those facts. All facts used are saved in data.json next to this file.</p>"
               + "".join(f"<p>Data note — {html.escape(n)}</p>" for n in notes) + "</div>")
    app.append("</div>")
    parts += app
    parts.append(f"<div class=footer>Generated by FinAgent on {date.today():%d %B %Y}. {html.escape(DISCLAIMER)} "
                 "Figures are drawn from public data believed to be reliable, but accuracy is not guaranteed.</div>")
    parts.append("</div>")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  5. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_research_report(tickers: list, cleaned_data: dict, yf_data: dict,
                          benchmark_df: pd.DataFrame = None, rss_news=None,
                          out_root: Path = None, use_llm: bool = True,
                          use_valuation: bool = True) -> Path:
    """
    Build the HTML research report and return the path to report.html.

    Args:
        tickers:      one ticker (company report) or several (comparison report)
        cleaned_data: {ticker: cleaned price DataFrame}
        yf_data:      {ticker: fetch_all_yfinance_fundamentals(ticker)}
        benchmark_df: cleaned price DataFrame for the benchmark (SPY), optional
        rss_news:     DataFrame from fetch_rss_news(), optional (headlines as context)
        out_root:     folder to create the report folder in (default reports/research)
        use_llm:      False builds a data-only report (no API calls)
        use_valuation: False skips the valuation module (no extra downloads)
    """
    tickers = [t for t in tickers if t in cleaned_data]
    if not tickers:
        raise ValueError("No price data for the requested tickers.")
    label = " vs ".join(tickers)
    out_dir = (out_root or RESEARCH_DIR) / f"{'_'.join(tickers)}_{date.today():%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("fig*.png"):   # re-running the same day replaces the report
        old.unlink()

    ctx = _build_data_context({t: cleaned_data[t] for t in tickers}, benchmark_df=benchmark_df)
    valuations = {}
    if use_valuation:
        for t in tickers:
            print(f"  Valuing {t} (DCF / residual income / DDM / peers as appropriate)...")
            try:
                valuations[t] = value_company(t, yf_data.get(t) or {})
            except Exception as e:   # valuation problems must not stop the report
                logger.warning(f"Valuation for {t} failed: {e}")
    facts = _collect_facts(tickers, ctx, cleaned_data, yf_data, rss_news, valuations)
    figs = _make_figures(tickers, ctx, cleaned_data, yf_data, benchmark_df, out_dir, valuations)

    thesis, sections = {}, {}
    llm_facts = _compact_for_llm(facts)
    if _output_budget(_thesis_prompt(label, llm_facts, figs, BENCHMARK_TICKER), 1800) < 1500:
        llm_facts = _compact_for_llm(facts, level=2)     # comparison reports: tighter facts
    _FIELD_NAMES.clear()
    _collect_keys(llm_facts, _FIELD_NAMES)
    _FIELD_NAMES.difference_update({t.lower() for t in tickers})   # keep "(AAPL)" etc.
    if use_llm:
        print("  Writing investment thesis...")
        tp = _thesis_prompt(label, llm_facts, figs, BENCHMARK_TICKER)
        if _output_budget(tp, 1800) >= 1200:
            thesis = _call_groq_json(tp, SYSTEM_REPORT, max_tokens=_output_budget(tp, 1800))
        else:
            logger.warning("Report data too large for the Groq token limit; writing a data-only report.")
        if thesis.get("theses"):
            print("  Writing report sections...")
            sections = _write_sections(label, llm_facts, figs, thesis)
    if not thesis.get("theses") or not sections:
        thesis_fb, sections_fb = _fallback_narrative(label, facts)
        thesis = thesis if thesis.get("theses") else thesis_fb
        sections = sections or sections_fb

    risk_file = f"fig{len(figs) + 1:02d}.png"
    if F.risk_matrix(sections.get("risks"), out_dir / risk_file):
        figs.append(Figure(f"Fig.{len(figs) + 1:02d}", "risks", "Risk Matrix", risk_file,
                           "likelihood vs impact of the risks listed", "Source: FinAgent analysis"))

    (out_dir / "report.html").write_text(
        _render_html(label, tickers, thesis, sections, figs, facts, ctx, yf_data, valuations), encoding="utf-8")
    (out_dir / "data.json").write_text(json.dumps(
        {"tickers": tickers, "generated": str(date.today()), "facts": facts,
         "figures": [{k: x for k, x in f.__dict__.items() if k != "html"} for f in figs],
         "valuations": {t: _json_safe_valuation(v) for t, v in valuations.items()},
         "thesis": thesis, "sections": sections},
        indent=1, default=str), encoding="utf-8")
    return out_dir / "report.html"


def _json_safe_valuation(v: dict) -> dict:
    """Valuation result without bulky objects (price series), for data.json."""
    out = {k: x for k, x in v.items() if k not in ("history",)}
    if v.get("history"):
        out["history"] = {k: x for k, x in v["history"].items() if k != "series"}
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  6. Optional PDF via the browser already installed (no extra Python packages)
# ─────────────────────────────────────────────────────────────────────────────

def _find_browser():
    """Locate Edge or Chrome; their headless mode can print HTML to PDF."""
    import os
    import shutil
    candidates = [
        os.environ.get("FINAGENT_BROWSER"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ] + [shutil.which(n) for n in ("msedge", "google-chrome", "chrome", "chromium", "chromium-browser")]
    return next((c for c in candidates if c and Path(c).exists()), None)


def export_pdf(html_path: Path):
    """
    Print report.html to report.pdf with headless Edge/Chrome.
    Returns the PDF path, or None if no browser is available (open the HTML
    and use Ctrl+P -> Save as PDF instead).
    """
    import subprocess
    browser = _find_browser()
    if not browser:
        return None
    html_path = Path(html_path).resolve()
    pdf = html_path.with_suffix(".pdf")
    try:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", html_path.as_uri()],
                       capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"PDF export failed: {e}")
        return None
    return pdf if pdf.exists() and pdf.stat().st_size > 0 else None
