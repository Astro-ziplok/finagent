"""
src/valuation/models.py — Valuation maths (pure functions, no network)
=======================================================================
Textbook models used by the valuation engine. Everything here is plain
arithmetic on the inputs given, so each formula can be unit-tested.

  capm / wacc            cost of equity and weighted average cost of capital
  growth_path            analyst growth for years 1-2, fading to terminal growth
  dcf_fcff               free cash flow to the firm DCF (enterprise -> equity value)
  reverse_dcf            revenue growth implied by the current share price
  dcf_sensitivity        value per share across WACC x terminal-growth grid
  residual_income        RI model: book value + PV of profits above the cost of equity
  ddm                    multi-stage dividend discount model
  peer_stats / regression  helpers for comparable-company valuation
"""

import numpy as np


def capm(risk_free: float, beta: float, erp: float) -> float:
    """Cost of equity = risk-free rate + beta x equity risk premium."""
    return risk_free + beta * erp


def wacc(cost_equity: float, cost_debt: float, tax_rate: float,
         equity_value: float, debt_value: float) -> dict:
    """Weighted average cost of capital using market equity and book debt weights."""
    total = (equity_value or 0) + (debt_value or 0)
    we = (equity_value or 0) / total if total > 0 else 1.0
    wd = 1 - we
    return {"wacc": we * cost_equity + wd * cost_debt * (1 - tax_rate),
            "weight_equity": we, "weight_debt": wd,
            "after_tax_cost_debt": cost_debt * (1 - tax_rate)}


def growth_path(g1: float, g2: float, g_terminal: float, years: int = 5) -> list:
    """
    Year 1 and 2 growth from analyst consensus; later years fade linearly
    toward terminal growth (reached in the first year after the forecast).
    """
    path = [g1, g2]
    n_fade = max(years - 2, 0)
    for i in range(1, n_fade + 1):
        path.append(g2 + (g_terminal - g2) * i / (n_fade + 1))
    return path[:years]


def dcf_fcff(base_revenue: float, growth: list, fcf_margin: float, wacc_rate: float,
             g_terminal: float, net_debt: float, shares: float) -> dict:
    """
    FCFF DCF: project revenue x free-cash-flow margin, discount at WACC,
    add a Gordon-growth terminal value, then subtract net debt.
    """
    if wacc_rate <= g_terminal + 0.005:
        raise ValueError("WACC must exceed terminal growth by at least 0.5 percentage points.")
    if shares <= 0:
        raise ValueError("Share count must be positive.")
    rows, revenue, pv_sum = [], base_revenue, 0.0
    for t, g in enumerate(growth, 1):
        revenue *= 1 + g
        fcff = revenue * fcf_margin
        df = 1 / (1 + wacc_rate) ** t
        pv_sum += fcff * df
        rows.append({"year": t, "growth": g, "revenue": revenue, "fcff": fcff,
                     "discount_factor": df, "pv_fcff": fcff * df})
    terminal = rows[-1]["fcff"] * (1 + g_terminal) / (wacc_rate - g_terminal)
    pv_terminal = terminal * rows[-1]["discount_factor"]
    ev = pv_sum + pv_terminal
    equity = ev - net_debt
    return {"rows": rows, "pv_fcff": pv_sum, "terminal_value": terminal, "pv_terminal": pv_terminal,
            "enterprise_value": ev, "equity_value": equity, "per_share": equity / shares,
            "terminal_share_of_ev": pv_terminal / ev if ev else None}


def reverse_dcf(price: float, base_revenue: float, fcf_margin: float, wacc_rate: float,
                g_terminal: float, net_debt: float, shares: float, years: int = 5,
                lo: float = -0.5, hi: float = 1.5):
    """
    The constant annual revenue growth (years 1..n) at which the DCF value
    equals today's price, holding margin, WACC and terminal growth fixed.
    Returns None if no growth rate in [lo, hi] matches the price.
    """
    def gap(g):
        return dcf_fcff(base_revenue, [g] * years, fcf_margin, wacc_rate,
                        g_terminal, net_debt, shares)["per_share"] - price

    try:
        if gap(lo) > 0 or gap(hi) < 0:
            return None
    except ValueError:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def dcf_sensitivity(base_revenue, growth, fcf_margin, wacc_rate, g_terminal, net_debt, shares,
                    wacc_steps=(-0.01, -0.005, 0, 0.005, 0.01),
                    g_steps=(-0.01, -0.005, 0, 0.005, 0.01)) -> dict:
    """Value per share for WACC (rows) x terminal growth (columns) around the base case."""
    waccs = [wacc_rate + d for d in wacc_steps]
    gs = [g_terminal + d for d in g_steps]
    grid = []
    for w in waccs:
        row = []
        for g in gs:
            try:
                row.append(dcf_fcff(base_revenue, growth, fcf_margin, w, g, net_debt, shares)["per_share"])
            except ValueError:
                row.append(None)
        grid.append(row)
    return {"waccs": waccs, "growths": gs, "values": grid}


def residual_income(bvps: float, eps_path: list, payout: float, cost_equity: float,
                    g_terminal: float) -> dict:
    """
    Residual income model (CFA Institute): value = book value today + PV of
    future residual income, where RI_t = EPS_t - cost of equity x book value_(t-1).
    Book value grows by retained earnings; RI grows at the terminal rate afterwards.
    """
    if cost_equity <= g_terminal + 0.005:
        raise ValueError("Cost of equity must exceed terminal growth.")
    book, pv_sum, rows = bvps, 0.0, []
    for t, eps in enumerate(eps_path, 1):
        ri = eps - cost_equity * book
        df = 1 / (1 + cost_equity) ** t
        pv_sum += ri * df
        rows.append({"year": t, "eps": eps, "book_begin": book, "residual_income": ri,
                     "discount_factor": df, "pv_ri": ri * df})
        book += eps * (1 - payout)
    terminal = rows[-1]["residual_income"] * (1 + g_terminal) / (cost_equity - g_terminal)
    pv_terminal = terminal * rows[-1]["discount_factor"]
    return {"rows": rows, "book_value": bvps, "pv_ri": pv_sum, "pv_terminal": pv_terminal,
            "per_share": bvps + pv_sum + pv_terminal}


def ddm(d0: float, growth: list, cost_equity: float, g_terminal: float) -> dict:
    """Multi-stage dividend discount model with a Gordon-growth terminal value."""
    if cost_equity <= g_terminal + 0.005:
        raise ValueError("Cost of equity must exceed terminal growth.")
    d, pv_sum, rows = d0, 0.0, []
    for t, g in enumerate(growth, 1):
        d *= 1 + g
        df = 1 / (1 + cost_equity) ** t
        pv_sum += d * df
        rows.append({"year": t, "growth": g, "dividend": d, "discount_factor": df, "pv_dividend": d * df})
    terminal = d * (1 + g_terminal) / (cost_equity - g_terminal)
    pv_terminal = terminal * rows[-1]["discount_factor"]
    return {"rows": rows, "pv_dividends": pv_sum, "pv_terminal": pv_terminal,
            "per_share": pv_sum + pv_terminal}


def peer_stats(values, lo=None, hi=None, min_n: int = 3):
    """Quartiles of a peer multiple after removing missing, negative and extreme values."""
    clean = [float(v) for v in values
             if v is not None and np.isfinite(v) and v > 0
             and (lo is None or v >= lo) and (hi is None or v <= hi)]
    if len(clean) < min_n:
        return None
    q1, med, q3 = np.percentile(clean, [25, 50, 75])
    return {"q1": float(q1), "median": float(med), "q3": float(q3), "n": len(clean)}


def regression(xs, ys) -> dict:
    """Ordinary least squares y = a + b x, with R-squared."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"intercept": float(a), "slope": float(b),
            "r2": 1 - ss_res / ss_tot if ss_tot else 0.0, "n": len(x)}
