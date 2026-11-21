"""
src/valuation/summary.py — Plain-text valuation summary for the terminal
=========================================================================
No language model involved: every line is a computed number or its source.
"""


def _m(x):
    return "n/a" if x is None else f"${x:,.2f}"


def format_valuation(v: dict) -> str:
    t = v["ticker"]
    lines = [f"\n-- {t} VALUATION --",
             f"Company type: {v['company_type_label']}. {v['type_rationale']}", "",
             f"{'Method':<30}{'Role':<13}{'Weight':>7}{'Value':>12}{'Range':>22}"]
    for m in v["methods"].values():
        if "value" in m:
            rng = (f"{_m(m['low'])} - {_m(m['high'])}" if m.get("low") is not None else "")
            lines.append(f"{m['name']:<30}{m['role']:<13}{m['weight'] * 100:>6.0f}%{_m(m['value']):>12}{rng:>22}")
    for m in v["methods"].values():
        if "value" not in m:
            lines.append(f"{m['name']:<30}not used: {m['skipped']}")
    lines.append("")
    if v.get("fair_value"):
        lines.append(f"Model fair value: {_m(v['fair_value'])} (range {_m(v['fair_low'])} - {_m(v['fair_high'])}) "
                     f"vs price {_m(v['price'])} -> {v['upside'] * 100:+.1f}% implied. Not a recommendation.")
    rd = v.get("reverse_dcf") or {}
    if rd.get("implied_growth") is not None:
        g1, g2 = rd["consensus_growth_y1_y2"]
        lines.append(f"Reverse DCF: today's price requires {rd['implied_growth'] * 100:.1f}% revenue growth a year "
                     f"for {rd['years']} years (analyst consensus: {g1 * 100:.1f}% and {g2 * 100:.1f}% for the next two years).")
    if v.get("valuation_gap_note"):
        lines.append(f"Note: {v['valuation_gap_note']}")
    lines += ["", "Assumptions and sources:"]
    lines += [f"  - {a['name']}: {a['value']}  [{a['source']}]" for a in v["assumptions"]]
    return "\n".join(lines)
