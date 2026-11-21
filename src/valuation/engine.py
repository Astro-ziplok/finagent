"""
src/valuation/engine.py — Choose and run the right valuation methods
=====================================================================
value_company(ticker, yf_entry) returns one dict with:
  company_type + rationale     which kind of business this is, and why
  methods                      every method: value range, weight, or the reason it was skipped
  fair_value / range / upside  weighted result (no buy/sell rating)
  assumptions                  every input with its value and source (cited in the report)
  dcf / sensitivity / scenarios / reverse_dcf / comps / regression / history   detail tables

Method choice follows CFA practice: use the models that fit the business,
weight them (config.VALUATION_WEIGHTS), and show the rest as cross-checks.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (VALUATION_ASSUMPTIONS, VALUATION_WEIGHTS, DIVIDEND_MIN_YIELD,
                    DIVIDEND_MIN_PAYOUT, FINANCIAL_INDUSTRY_KEYWORDS)
from src.valuation import models as M
from src.valuation.inputs import risk_free_rate, company_inputs, select_peers, peer_multiples

logger = logging.getLogger(__name__)

METHOD_NAMES = {
    "dcf": "Discounted cash flow (FCFF)",
    "ri": "Residual income",
    "ddm": "Dividend discount model",
    "comps": "Comparable companies",
    "regression": "Peer regression",
    "history": "Own historical multiple",
}
TYPE_LABELS = {
    "cash_generative": "Profitable, cash-generative company",
    "financial": "Bank / insurer / capital-markets firm",
    "dividend": "Mature dividend payer",
    "high_growth": "High-growth or currently unprofitable company",
}
# Which peer multiples suit each company type
TYPE_MULTIPLES = {
    "cash_generative": ["forward_pe", "ev_ebitda"],
    "financial": ["pb", "forward_pe"],
    "dividend": ["forward_pe", "ev_ebitda"],
    "high_growth": ["ev_sales", "forward_pe"],
}
MULTIPLE_LABELS = {"forward_pe": "Forward P/E", "ev_ebitda": "EV/EBITDA", "ev_sales": "EV/Sales", "pb": "P/B"}
MULTIPLE_BOUNDS = {"forward_pe": (1, 150), "ev_ebitda": (1, 100), "ev_sales": (0.1, 50), "pb": (0.1, 60)}


def _a(name, value, source):
    return {"name": name, "value": value, "source": source}


def _pct(x, nd=1):
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def _money_b(x):
    return "n/a" if x is None else ("-" if x < 0 else "") + f"${abs(x) / 1e9:,.1f}B"


def classify(info: dict, cash_rows: list, income_rows: list) -> tuple:
    """Company type for model selection, with a plain-English reason."""
    industry = info.get("industry") or ""
    if any(k in industry for k in FINANCIAL_INDUSTRY_KEYWORDS):
        return "financial", (f"Industry '{industry}': cash flow is not meaningful for lenders and "
                             "insurers, so value is driven by book value and returns on equity.")
    fcfs = [r.get("free_cash_flow") for r in cash_rows[:4] if r.get("free_cash_flow") is not None]
    trailing_eps = info.get("trailingEps")
    if (fcfs and fcfs[0] <= 0) or sum(1 for f in fcfs if f <= 0) >= 2 or (trailing_eps is not None and trailing_eps <= 0):
        return "high_growth", ("Free cash flow or earnings are negative or unstable, so a DCF would be "
                               "unreliable; valued mainly on revenue multiples.")
    yld = (info.get("dividendRate") or 0) / info["currentPrice"] if info.get("currentPrice") else 0
    payout = info.get("payoutRatio") or 0
    if yld >= DIVIDEND_MIN_YIELD and payout >= DIVIDEND_MIN_PAYOUT:
        return "dividend", (f"Dividend yield {yld * 100:.1f}% and payout ratio {payout * 100:.0f}% "
                            "make dividends a reliable basis for value.")
    return "cash_generative", (f"Positive free cash flow in {sum(1 for f in fcfs if f > 0)} of the last "
                               f"{len(fcfs)} years, so cash-flow and earnings-multiple methods apply.")


def _fcf_margin(income_rows, cash_rows, tax_rate):
    """3-year average of (FCF - stock-based pay + after-tax interest) / revenue."""
    rev_by_period = {r["period"]: r.get("total_revenue") for r in income_rows}
    int_by_period = {r["period"]: r.get("interest_expense") for r in income_rows}
    margins = []
    for r in cash_rows[:3]:
        rev, fcf = rev_by_period.get(r["period"]), r.get("free_cash_flow")
        if not rev or fcf is None:
            continue
        interest = abs(int_by_period.get(r["period"]) or 0)
        fcff = fcf - abs(r.get("stock_based_comp") or 0) + interest * (1 - tax_rate)
        margins.append(fcff / rev)
    return (float(np.mean(margins)), len(margins)) if len(margins) >= 2 else (None, len(margins))


def _eps_path(inp, info, g_terminal, years):
    """EPS: analyst consensus for years 1-2, then growth fades to terminal."""
    est = inp.get("eps_est") or {}
    e1 = (est.get("0y") or {}).get("avg") or info.get("forwardEps")
    e2 = (est.get("+1y") or {}).get("avg")
    if not e1:
        return None, None
    if not e2:
        e2 = e1 * (1 + g_terminal)
    g2 = e2 / e1 - 1 if e1 > 0 else g_terminal
    g2 = float(np.clip(g2, -0.2, 0.3))
    path = [e1, e2]
    for g in M.growth_path(g2, g2, g_terminal, years)[2:]:
        path.append(path[-1] * (1 + g))
    n = (est.get("0y") or {}).get("numberOfAnalysts")
    return path, (f"Analyst consensus EPS for the current and next fiscal year (Yahoo Finance"
                  f"{f', {int(n)} analysts' if n else ''}); later years fade to terminal growth.")


def value_company(ticker: str, yf_entry: dict, price: float = None) -> dict:
    """Run the valuation for one company. Never raises for missing data: skipped methods say why."""
    inp = company_inputs(ticker)
    info = inp["info"]
    income = ((yf_entry or {}).get("income") or {}).get("annual") or []
    balance = ((yf_entry or {}).get("balance") or {}).get("annual") or []
    cash = ((yf_entry or {}).get("cashflow") or {}).get("annual") or []
    price = price or info.get("currentPrice") or info.get("regularMarketPrice")
    A = VALUATION_ASSUMPTIONS
    erp, g_t, years = A["equity_risk_premium"]["value"], A["terminal_growth"]["value"], A["forecast_years"]["value"]
    assumptions = []
    out = {"ticker": ticker, "name": info.get("shortName") or ticker, "price": price,
           "price_source": f"Yahoo Finance quote ({inp['as_of']})", "methods": {}, "assumptions": assumptions}

    ctype, why = classify(info, cash, income)
    out["company_type"], out["company_type_label"], out["type_rationale"] = ctype, TYPE_LABELS[ctype], why

    # ── Cost of capital ───────────────────────────────────────────────────────
    rf = risk_free_rate()
    raw_beta = info.get("beta")
    adj = A["beta_adjustment"]["value"]
    if raw_beta is None or not np.isfinite(raw_beta) or raw_beta <= 0:
        raw_beta, beta, beta_src = None, 1.0, "Market beta of 1.0 (Yahoo beta unavailable)"
    else:
        beta = adj * raw_beta + (1 - adj)
        beta_src = (f"Raw beta {raw_beta:.2f} from Yahoo Finance (5-year monthly returns vs S&P 500), "
                    f"adjusted: {A['beta_adjustment']['source']}")
    ke = M.capm(rf["value"], beta, erp)

    latest_inc = income[0] if income else {}
    tax = None
    if latest_inc.get("tax_provision") is not None and (latest_inc.get("pretax_income") or 0) > 0:
        tax = latest_inc["tax_provision"] / latest_inc["pretax_income"]
    if tax is None or not (0 <= tax <= 0.35):
        tax, tax_src = A["tax_rate_fallback"]["value"], A["tax_rate_fallback"]["source"]
    else:
        tax_src = (f"Effective tax rate, FY{latest_inc['period'][:4]} (tax provision / pre-tax income), "
                   "company filings via Yahoo Finance")

    latest_bal = balance[0] if balance else {}
    total_debt = latest_bal.get("total_debt") or info.get("totalDebt") or 0
    cash_sti = latest_bal.get("cash_and_st_investments") or latest_bal.get("cash") or info.get("totalCash") or 0
    interest = abs(latest_inc.get("interest_expense") or 0)
    kd = interest / total_debt if (interest and total_debt) else None
    if kd is None or not (rf["value"] <= kd <= rf["value"] + 0.08):
        kd = rf["value"] + A["credit_spread_fallback"]["value"]
        kd_src = A["credit_spread_fallback"]["source"]
    else:
        kd_src = f"Interest expense / total debt, FY{latest_inc['period'][:4]}, company filings via Yahoo Finance"
    mcap = info.get("marketCap") or (price * (info.get("sharesOutstanding") or 0))
    w = M.wacc(ke, kd, tax, mcap, total_debt)
    shares = info.get("sharesOutstanding") or latest_bal.get("shares_outstanding")
    net_debt = total_debt - cash_sti
    bal_date = latest_bal.get("period", "latest")

    assumptions += [
        _a("Risk-free rate", _pct(rf["value"], 2), rf["source"]),
        _a("Equity risk premium", _pct(erp), A["equity_risk_premium"]["source"]),
        _a("Beta (adjusted)", f"{beta:.2f}", beta_src),
        _a("Cost of equity (CAPM)", _pct(ke, 2), "Risk-free rate + beta x equity risk premium"),
        _a("Pre-tax cost of debt", _pct(kd, 2), kd_src),
        _a("Tax rate", _pct(tax), tax_src),
        _a("Capital weights", f"{w['weight_equity'] * 100:.0f}% equity / {w['weight_debt'] * 100:.0f}% debt",
           f"Market capitalisation (Yahoo Finance) and total debt ({bal_date} balance sheet)"),
        _a("WACC", _pct(w["wacc"], 2), "Weighted average of cost of equity and after-tax cost of debt"),
        _a("Terminal growth", _pct(g_t), A["terminal_growth"]["source"]),
        _a("Forecast horizon", f"{years} years", A["forecast_years"]["source"]),
        _a("Net debt", _money_b(net_debt),
           f"Total debt - cash and short-term investments ({bal_date} balance sheet)"),
        _a("Shares outstanding", "n/a" if not shares else f"{shares / 1e9:,.3f}B", "Yahoo Finance quote data"),
        _a("Current price", "n/a" if not price else f"${price:,.2f}", out["price_source"]),
    ]
    out["cost_of_capital"] = {"risk_free": rf["value"], "beta": beta, "raw_beta": raw_beta, "erp": erp, "cost_equity": ke,
                              "cost_debt": kd, "tax": tax, **w}

    # Statements in a different currency from the share price (e.g. ADRs) can't feed per-share models
    fin_ccy, px_ccy = info.get("financialCurrency"), info.get("currency")
    statements_ok = not (fin_ccy and px_ccy and fin_ccy != px_ccy)
    ccy_reason = f"Financial statements are in {fin_ccy} but the share price is in {px_ccy}."

    # ── DCF (FCFF) + reverse DCF ──────────────────────────────────────────────
    m = out["methods"]
    if ctype == "financial":
        m["dcf"] = {"skipped": "Not used for financial firms: deposits and loans make free cash flow meaningless."}
    elif not statements_ok:
        m["dcf"] = {"skipped": ccy_reason}
    else:
        margin, n_years = _fcf_margin(income, cash, tax)
        rev_est = inp.get("revenue_est") or {}
        base_rev = (rev_est.get("0y") or {}).get("yearAgoRevenue") or (income[0].get("total_revenue") if income else None)
        g1, g2 = (rev_est.get("0y") or {}).get("growth"), (rev_est.get("+1y") or {}).get("growth")
        if g1 is not None and g2 is not None:
            n = (rev_est.get("0y") or {}).get("numberOfAnalysts")
            g_src = (f"Analyst consensus revenue growth for the current and next fiscal year (Yahoo Finance"
                     f"{f', {int(n)} analysts' if n else ''}); years 3-{years} fade to terminal growth")
        else:
            hist = [r.get("total_revenue") for r in income[:2]]
            g1 = g2 = float(np.clip(hist[0] / hist[1] - 1, -0.1, 0.4)) if len(hist) == 2 and all(hist) else g_t
            g_src = "Latest annual revenue growth (no analyst estimates available), bounded -10% to +40%"
        if margin is None or margin <= 0 or not base_rev or not shares:
            m["dcf"] = {"skipped": ("Free cash flow margin is negative or history too short for a DCF."
                                    if not margin or margin <= 0 else "Revenue or share count unavailable.")}
        else:
            growth = M.growth_path(g1, g2, g_t, years)
            try:
                base = M.dcf_fcff(base_rev, growth, margin, w["wacc"], g_t, net_debt, shares)
                sm, gs = A["scenario_margin_shift"]["value"], A["scenario_growth_shift"]["value"]
                bull = M.dcf_fcff(base_rev, [g + gs for g in growth], margin * (1 + sm), w["wacc"], g_t, net_debt, shares)
                bear = M.dcf_fcff(base_rev, [g - gs for g in growth], margin * (1 - sm), w["wacc"], g_t, net_debt, shares)
                m["dcf"] = {"value": base["per_share"], "low": bear["per_share"], "high": bull["per_share"],
                            "range_basis": "bear to bull scenario"}
                out["dcf"] = {**base, "base_revenue": base_rev, "fcf_margin": margin, "growth": growth}
                out["scenarios"] = [
                    {"case": "Bear", "growth_shift": -gs, "margin": margin * (1 - sm), "value": bear["per_share"]},
                    {"case": "Base", "growth_shift": 0.0, "margin": margin, "value": base["per_share"]},
                    {"case": "Bull", "growth_shift": gs, "margin": margin * (1 + sm), "value": bull["per_share"]},
                ]
                out["sensitivity"] = M.dcf_sensitivity(base_rev, growth, margin, w["wacc"], g_t, net_debt, shares)
                assumptions += [
                    _a("Revenue growth, years 1-2", f"{_pct(g1)}, {_pct(g2)}", g_src),
                    _a("Free cash flow margin", _pct(margin),
                       f"{n_years}-year average of (free cash flow - stock-based compensation + after-tax "
                       "interest) / revenue, company filings via Yahoo Finance. Stock-based pay is treated "
                       "as a real cost."),
                    _a("Base-year revenue", _money_b(base_rev), "Last reported fiscal year (Yahoo Finance)"),
                    _a("Scenarios", f"growth {_pct(-gs, 0)}/+{_pct(gs, 0)}, margin x{1 - sm:.1f}/x{1 + sm:.1f}",
                       f"{A['scenario_growth_shift']['source']} {A['scenario_margin_shift']['source']}"),
                    _a("Discounting", "End-of-year", "Simplification: cash flows assumed at each year end."),
                ]
                if price:
                    implied = M.reverse_dcf(price, base_rev, margin, w["wacc"], g_t, net_debt, shares, years)
                    out["reverse_dcf"] = {"implied_growth": implied,
                                          "consensus_growth_y1_y2": [g1, g2],
                                          "years": years}
            except ValueError as e:
                m["dcf"] = {"skipped": str(e)}

    # ── Residual income ───────────────────────────────────────────────────────
    eps_path, eps_src = _eps_path(inp, info, g_t, years)
    bvps = info.get("bookValue")
    payout = float(np.clip(info.get("payoutRatio") or 0, 0, 0.9))
    if ctype != "financial":
        m["ri"] = {"skipped": ("Not used: book value says little about the value of a non-financial "
                               "company" + (f" (here P/B is {info.get('priceToBook'):.1f}x, inflated by buybacks)."
                                            if (info.get("priceToBook") or 0) > 8 else "."))}
    elif not statements_ok:
        m["ri"] = {"skipped": ccy_reason}
    elif not eps_path or not bvps or bvps <= 0:
        m["ri"] = {"skipped": "Book value or EPS estimates unavailable."}
    else:
        try:
            base = M.residual_income(bvps, eps_path, payout, ke, g_t)
            hi = M.residual_income(bvps, eps_path, payout, ke - 0.01, g_t)
            lo = M.residual_income(bvps, eps_path, payout, ke + 0.01, g_t)
            m["ri"] = {"value": base["per_share"], "low": lo["per_share"], "high": hi["per_share"],
                       "range_basis": "cost of equity +/- 1%"}
            out["ri"] = base
            assumptions += [_a("EPS forecast", ", ".join(f"${e:,.2f}" for e in eps_path), eps_src),
                            _a("Book value per share", f"${bvps:,.2f}", "Yahoo Finance quote data"),
                            _a("Payout ratio", _pct(payout, 0), "Yahoo Finance (trailing), bounded 0-90%")]
        except ValueError as e:
            m["ri"] = {"skipped": str(e)}

    # ── Dividend discount model ───────────────────────────────────────────────
    d0 = info.get("dividendRate") or 0
    yld = d0 / price if price else 0
    if ctype not in ("dividend", "financial"):
        m["ddm"] = {"skipped": (f"Not used: dividend yield {yld * 100:.1f}% is too small for dividends to "
                                "explain value" + (" (most cash is returned through buybacks)." if d0 else "."))}
    elif not d0 or yld < DIVIDEND_MIN_YIELD:
        m["ddm"] = {"skipped": f"Dividend yield {yld * 100:.1f}% is below the {DIVIDEND_MIN_YIELD * 100:.1f}% threshold."}
    elif (info.get("payoutRatio") or 0) < DIVIDEND_MIN_PAYOUT:
        m["ddm"] = {"skipped": (f"Payout ratio {(info.get('payoutRatio') or 0) * 100:.0f}% is below "
                                f"{DIVIDEND_MIN_PAYOUT * 100:.0f}%: dividends are a small part of what "
                                "shareholders earn, so a DDM would understate value.")}
    elif not eps_path:
        m["ddm"] = {"skipped": "EPS estimates unavailable for the dividend growth path."}
    else:
        g_div = float(np.clip(eps_path[1] / eps_path[0] - 1 if eps_path[0] > 0 else g_t, 0, 0.10))
        growth = M.growth_path(g_div, g_div, g_t, years)
        try:
            base = M.ddm(d0, growth, ke, g_t)
            hi, lo = M.ddm(d0, growth, ke - 0.01, g_t), M.ddm(d0, growth, ke + 0.01, g_t)
            m["ddm"] = {"value": base["per_share"], "low": lo["per_share"], "high": hi["per_share"],
                        "range_basis": "cost of equity +/- 1%"}
            out["ddm"] = base
            assumptions += [_a("Annual dividend", f"${d0:,.2f}", "Yahoo Finance (indicated annual rate)"),
                            _a("Dividend growth, years 1-2", _pct(g_div),
                               "Analyst EPS growth for next fiscal year, bounded 0-10%, fading to terminal growth")]
        except ValueError as e:
            m["ddm"] = {"skipped": str(e)}

    # ── Comparable companies, regression, own history ─────────────────────────
    _comps(out, ticker, info, ctype, net_debt, shares, price, assumptions)
    _own_history(out, info, income, price, ctype)

    # ── Weighted fair value (no rating) ───────────────────────────────────────
    weights = VALUATION_WEIGHTS[ctype]
    usable = {k: wt for k, wt in weights.items() if "value" in m.get(k, {})}
    total = sum(usable.values())
    for k, v in m.items():
        v["name"] = METHOD_NAMES.get(k, k)
        v["weight"] = usable.get(k, 0) / total if total else 0
        v["role"] = "weighted" if k in usable else ("cross-check" if "value" in v else "not used")
    if total:
        out["fair_value"] = sum(m[k]["value"] * m[k]["weight"] for k in usable)
        out["fair_low"] = sum(m[k]["low"] * m[k]["weight"] for k in usable)
        out["fair_high"] = sum(m[k]["high"] * m[k]["weight"] for k in usable)
        out["upside"] = out["fair_value"] / price - 1 if price else None
        assumptions.append(_a("Method weights", ", ".join(f"{METHOD_NAMES[k]} {m[k]['weight'] * 100:.0f}%" for k in usable),
                              f"FinAgent default for a {TYPE_LABELS[ctype].lower()} (config.VALUATION_WEIGHTS); "
                              "re-scaled if a method could not be computed"))
    else:
        out["fair_value"] = None
    # Flag when the market price sits far outside every model's range
    vals = [v for v in m.values() if "value" in v]
    if price and vals and out.get("fair_value"):
        lo_all = min(min(v["value"], v.get("low") or v["value"]) for v in vals)
        hi_all = max(max(v["value"], v.get("high") or v["value"]) for v in vals)
        if abs(out["upside"]) > 0.5 and not (lo_all <= price <= hi_all):
            if price > hi_all:
                why = ("the market is pricing growth or businesses not visible in current cash flows and "
                       "consensus estimates, or investors apply a lower discount rate than these models. "
                       "The reverse DCF shows the growth the price requires.")
            else:
                why = ("the market doubts that consensus estimates will be met (they may be unusually high), "
                       "investors apply a higher discount rate than these models, or the quote and the "
                       "financial data are out of step (check the data notes). Treat the model values with caution.")
            out["valuation_gap_note"] = (f"The share price (${price:,.2f}) is "
                                         f"{'above' if price > hi_all else 'below'} the range of every method "
                                         f"(${lo_all:,.2f}-${hi_all:,.2f}). This usually means {why}")
    lo52, hi52 = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
    out["range_52w"] = (lo52, hi52) if lo52 and hi52 else None
    return out


def _comps(out, ticker, info, ctype, net_debt, shares, price, assumptions):
    m = out["methods"]
    sel = select_peers(ticker, info.get("sector"), info.get("industry"))
    out["peer_group"] = sel
    if sel["basis"] == "none":
        m["comps"] = {"skipped": sel["rationale"]}
        return
    peers = peer_multiples(sel["peers"])
    out["peers"] = peers
    metric = {"forward_pe": info.get("forwardEps"), "pb": info.get("bookValue"),
              "ev_ebitda": info.get("ebitda"), "ev_sales": info.get("totalRevenue")}
    implied = {}
    for mult in TYPE_MULTIPLES[ctype]:
        lo_b, hi_b = MULTIPLE_BOUNDS[mult]
        stats = M.peer_stats([p.get(mult) for p in peers.values()], lo_b, hi_b)
        base = metric.get(mult)
        if not stats or not base or base <= 0 or not shares:
            continue
        if mult in ("ev_ebitda", "ev_sales"):   # enterprise multiples -> equity per share
            conv = {q: (stats[q] * base - net_debt) / shares for q in ("q1", "median", "q3")}
        else:
            conv = {q: stats[q] * base for q in ("q1", "median", "q3")}
        own = info.get({"forward_pe": "forwardPE", "pb": "priceToBook", "ev_ebitda": "enterpriseToEbitda",
                        "ev_sales": "enterpriseToRevenue"}[mult])
        implied[mult] = {"label": MULTIPLE_LABELS[mult], "peer": stats, "own": own, "implied": conv}
    out["comps"] = implied
    if not implied:
        m["comps"] = {"skipped": "Fewer than 3 peers with usable multiples."}
        return
    m["comps"] = {"value": float(np.mean([v["implied"]["median"] for v in implied.values()])),
                  "low": float(np.mean([v["implied"]["q1"] for v in implied.values()])),
                  "high": float(np.mean([v["implied"]["q3"] for v in implied.values()])),
                  "range_basis": "peer 25th-75th percentile multiples"}
    assumptions.append(_a("Peer group", ", ".join(sel["peers"]), sel["rationale"]))
    assumptions.append(_a("Peer multiples", ", ".join(v["label"] for v in implied.values()),
                          "Yahoo Finance quote data; median applied to the company's forward EPS, EBITDA, "
                          "revenue or book value; negative and extreme values excluded"))

    # Regression cross-check: what multiple do peers get for their growth / returns?
    if ctype == "financial":
        xk, yk, x0, metric0, label = "roe", "pb", info.get("returnOnEquity"), info.get("bookValue"), ("ROE", "P/B")
    else:
        fe, te = info.get("forwardEps"), info.get("trailingEps")
        xk, yk, x0, metric0, label = ("exp_eps_growth", "forward_pe", (fe / te - 1) if (fe and te and te > 0) else None,
                                      fe, ("Expected EPS growth", "Forward P/E"))
    pts = [(p[xk], p[yk]) for p in peers.values()
           if p.get(xk) is not None and p.get(yk) is not None and 0 < p[yk] < 150 and -0.5 < p[xk] < 3]
    if len(pts) >= 5 and x0 is not None and metric0:
        reg = M.regression([a for a, _ in pts], [b for _, b in pts])
        fitted = reg["intercept"] + reg["slope"] * x0
        if fitted > 0:
            m["regression"] = {"value": fitted * metric0, "low": None, "high": None,
                               "range_basis": f"fitted {label[1]} of {fitted:.1f}x"}
        out["regression"] = {**reg, "x_label": label[0], "y_label": label[1], "points": pts,
                             "peer_names": [t for t, p in peers.items() if p.get(xk) is not None
                                            and p.get(yk) is not None and 0 < p[yk] < 150 and -0.5 < p[xk] < 3],
                             "target": (x0, info.get({"pb": "priceToBook", "forward_pe": "forwardPE"}[yk])),
                             "fitted_multiple": fitted}


def _own_history(out, info, income, price, ctype):
    """Today's P/E versus the company's own trailing P/E over recent years."""
    m = out["methods"]
    inp = company_inputs(out["ticker"])
    monthly = inp.get("monthly_close")
    eps_rows = sorted([(pd.Timestamp(r["period"]), r.get("diluted_eps")) for r in income
                       if r.get("diluted_eps")], key=lambda x: x[0])
    ttm = info.get("trailingEps")
    if monthly is None or monthly.empty or len(eps_rows) < 2 or not ttm or ttm <= 0:
        m["history"] = {"skipped": "Not enough positive earnings history for a historical P/E range."}
        return
    splits = inp.get("splits")
    if splits is not None and len(splits):
        d, r = splits.index[-1], splits.iloc[-1]
        m["history"] = {"skipped": (f"Stock split ({r:g}-for-1 on {d.date()}) inside the window: reported EPS "
                                    "before the split is not comparable with split-adjusted prices.")}
        return
    pe = []
    for dt, px in monthly.items():
        past = [e for d, e in eps_rows if d <= dt]
        if past and past[-1] > 0:
            pe.append((dt, px / past[-1]))
    if len(pe) < 12:
        m["history"] = {"skipped": "Fewer than 12 months of P/E history."}
        return
    s = pd.Series(dict(pe))
    mean, std = float(s.mean()), float(s.std())
    if std / mean > 0.6:
        m["history"] = {"skipped": (f"Historical P/E is too unstable to be a guide (from {s.min():.0f}x to "
                                    f"{s.max():.0f}x), e.g. because earnings were near zero.")}
        return
    m["history"] = {"value": mean * ttm, "low": (mean - std) * ttm, "high": (mean + std) * ttm,
                    "range_basis": "own average P/E +/- 1 standard deviation"}
    out["history"] = {"series": s, "mean": mean, "std": std, "current": price / ttm if price else None,
                      "ttm_eps": ttm}
