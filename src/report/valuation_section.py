"""
src/report/valuation_section.py — Valuation content for the research report
============================================================================
Turns valuation results (src/valuation/engine.py) into:
  valuation_facts()      compact numbers for the language model and data.json
  summary_table()        sidebar: method / role / weight / value per share  (cf. reference Fig.43)
  key_assumptions()      sidebar: cost-of-capital inputs with short sources (cf. reference Fig.44)
  sensitivity_table()    sidebar: DCF value across WACC x terminal growth  (cf. reference Fig.45)
  appendix_valuation()   appendix: method selection, DCF, WACC, scenarios, reverse DCF, peers, RI/DDM
  appendix_sources()     appendix: every assumption with its value and full source
"""

import html


def _m(x, nd=2):
    return "–" if x is None else f"${x:,.{nd}f}"


def _p(x, nd=1, sign=False):
    return "–" if x is None else (f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%")


def _x(x):
    return "–" if x is None else f"{x:.1f}x"


def _e(s):
    return html.escape(str(s))


def valuation_facts(v: dict) -> dict:
    """Short, number-only summary of a valuation for the language model."""
    if not v:
        return {}
    meth = {}
    for k, m in v["methods"].items():
        if "value" in m:
            meth[k] = {"name": m["name"], "role": m["role"], "weight_pct": round(m["weight"] * 100),
                       "value": round(m["value"], 2),
                       "range": [round(m["low"], 2) if m.get("low") is not None else None,
                                 round(m["high"], 2) if m.get("high") is not None else None],
                       "range_basis": m.get("range_basis")}
        else:
            meth[k] = {"name": m["name"], "not_used_because": m["skipped"]}
    cc = v.get("cost_of_capital", {})
    f = {
        "company_type": v["company_type_label"], "why_these_methods": v["type_rationale"],
        "price": v.get("price"), "methods": meth,
        "fair_value": round(v["fair_value"], 2) if v.get("fair_value") else None,
        "fair_value_range": [round(v["fair_low"], 2), round(v["fair_high"], 2)] if v.get("fair_value") else None,
        "upside_vs_price_pct": round(v["upside"] * 100, 1) if v.get("upside") is not None else None,
        "cost_of_equity_pct": round(cc.get("cost_equity", 0) * 100, 2),
        "wacc_pct": round(cc.get("wacc", 0) * 100, 2), "beta_adjusted": round(cc.get("beta", 0), 2),
        "risk_free_pct": round(cc.get("risk_free", 0) * 100, 2),
        "peer_group": (v.get("peer_group") or {}).get("peers"),
        "peer_group_reason": (v.get("peer_group") or {}).get("rationale"),
    }
    if v.get("dcf"):
        f["dcf_terminal_value_share_pct"] = round(v["dcf"]["terminal_share_of_ev"] * 100)
        f["dcf_fcf_margin_pct"] = round(v["dcf"]["fcf_margin"] * 100, 1)
    s = v.get("sensitivity")
    if s:
        mid_w, mid_g = len(s["waccs"]) // 2, len(s["growths"]) // 2
        base = s["values"][mid_w][mid_g]

        def cell(i, j):
            val = s["values"][i][j]
            if val is None or base is None:
                return None
            return {"value_usd": round(val, 2), "change_vs_base_usd": round(val - base, 2),
                    "change_vs_base_pct": round((val / base - 1) * 100, 1)}

        # Pre-computed so the narrative never has to do this arithmetic itself
        f["dcf_sensitivity"] = {"base_value_usd": round(base, 2) if base else None,
                                "wacc_plus_1pt": cell(-1, mid_g), "wacc_minus_1pt": cell(0, mid_g),
                                "terminal_growth_plus_0.5pt": cell(mid_w, mid_g + 1),
                                "terminal_growth_minus_0.5pt": cell(mid_w, mid_g - 1)}
    rd = v.get("reverse_dcf") or {}
    if rd.get("implied_growth") is not None:
        f["reverse_dcf"] = {"price_implies_annual_revenue_growth_pct": round(rd["implied_growth"] * 100, 1),
                            "for_years": rd["years"],
                            "analyst_consensus_growth_next_2y_pct": [round(g * 100, 1) for g in rd["consensus_growth_y1_y2"]]}
    if v.get("comps"):
        f["peer_multiples"] = {c["label"]: {"peer_median": round(c["peer"]["median"], 1),
                                            "company": round(c["own"], 1) if c.get("own") else None}
                               for c in v["comps"].values()}
    if v.get("valuation_gap_note"):
        f["valuation_gap_note"] = v["valuation_gap_note"]
    return f


def summary_table(valuations: dict) -> str:
    rows = []
    multi = len(valuations) > 1
    for t, v in valuations.items():
        if multi:
            rows.append(f'<tr class=grp><td colspan=3>{_e(t)}</td></tr>')
        for k, m in v["methods"].items():
            if "value" not in m:
                continue
            role = f"{m['weight'] * 100:.0f}%" if m["role"] == "weighted" else "check"
            rows.append(f"<tr><td>{_e(m['name'])}</td><td>{role}</td><td>{_m(m['value'])}</td></tr>")
        if v.get("fair_value"):
            rows.append(f"<tr class=total><td>Fair value (weighted)</td><td></td><td>{_m(v['fair_value'])}</td></tr>")
            rows.append(f"<tr><td>Range</td><td></td><td>{_m(v['fair_low'], 0)}–{_m(v['fair_high'], 0)}</td></tr>")
        rows.append(f"<tr><td>Current price</td><td></td><td>{_m(v.get('price'))}</td></tr>")
        if v.get("upside") is not None:
            rows.append(f"<tr class=total><td>Implied upside / downside</td><td></td><td>{_p(v['upside'], 1, True)}</td></tr>")
    return ("<table class=mini><thead><tr><th>Method</th><th>Weight</th><th>Value / share</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def key_assumptions(valuations: dict) -> str:
    keys = ["Risk-free rate", "Equity risk premium", "Beta (adjusted)", "Cost of equity (CAPM)", "WACC",
            "Terminal growth", "Forecast horizon"]
    tick = list(valuations)
    vals = {t: {a["name"]: a["value"] for a in valuations[t]["assumptions"]} for t in tick}
    head = "".join(f"<th>{_e(t)}</th>" for t in tick) if len(tick) > 1 else "<th>Value</th>"
    body = "".join(f"<tr><td>{_e(k)}</td>{''.join(f'<td>{_e(vals[t].get(k, chr(8211)))}</td>' for t in tick)}</tr>"
                   for k in keys)
    return f"<table class=mini><thead><tr><th>Assumption</th>{head}</tr></thead><tbody>{body}</tbody></table>"


def sensitivity_table(v: dict) -> str:
    s = v.get("sensitivity")
    if not s:
        return ""
    price = v.get("price")
    head = "".join(f"<th>{g * 100:.1f}%</th>" for g in s["growths"])
    rows = []
    for i, (w, row) in enumerate(zip(s["waccs"], s["values"])):
        cells = []
        for j, val in enumerate(row):
            cls = " class=base" if (i == len(s["waccs"]) // 2 and j == len(s["growths"]) // 2) else ""
            txt = "–" if val is None else f"{val:,.0f}"
            if val is not None and price:
                txt += f"<small> {(val / price - 1) * 100:+.0f}%</small>"
            cells.append(f"<td{cls}>{txt}</td>")
        rows.append(f"<tr><th>{w * 100:.1f}%</th>{''.join(cells)}</tr>")
    return ("<table class=mini sens><thead><tr><th>WACC \\ g</th>" + head + "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<p class=mini-note>DCF value per share (USD) and % vs current price; base case outlined.</p>")


def _tbl(head, rows, cls="fin"):
    h = "".join(f"<th>{_e(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class=table-wrap><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def appendix_valuation(valuations: dict) -> str:
    parts = []
    for t, v in valuations.items():
        if len(valuations) > 1:
            parts.append(f"<h4>{_e(t)}</h4>")
        parts.append(f"<p class=method><b>Company type:</b> {_e(v['company_type_label'])}. {_e(v['type_rationale'])}</p>")
        rows = []
        for k, m in v["methods"].items():
            if "value" in m:
                rng = f"{_m(m.get('low'), 0)}–{_m(m.get('high'), 0)}" if m.get("low") is not None else "–"
                rows.append([_e(m["name"]), _e(m["role"]), f"{m['weight'] * 100:.0f}%", _m(m["value"]),
                             rng, _e(m.get("range_basis", ""))])
            else:
                rows.append([_e(m["name"]), "not used", "0%", "–", "–", _e(m["skipped"])])
        parts.append("<p class=cap>Method selection</p>" +
                     _tbl(["Method", "Role", "Weight", "Value", "Range", "Basis / reason"], rows))

        cc = v.get("cost_of_capital") or {}
        if cc:
            parts.append("<p class=cap>Cost of capital</p>" + _tbl(
                ["Risk-free", "Beta (adj.)", "ERP", "Cost of equity", "Pre-tax cost of debt", "Tax", "Equity / debt", "WACC"],
                [[_p(cc["risk_free"], 2), f"{cc['beta']:.2f}", _p(cc["erp"]), _p(cc["cost_equity"], 2),
                  _p(cc["cost_debt"], 2), _p(cc["tax"]),
                  f"{cc['weight_equity'] * 100:.0f}% / {cc['weight_debt'] * 100:.0f}%", _p(cc["wacc"], 2)]]))

        d = v.get("dcf")
        if d:
            rows = [[f"Y{r['year']}", _p(r["growth"]), f"{r['revenue'] / 1e9:,.1f}", f"{r['fcff'] / 1e9:,.1f}",
                     f"{r['discount_factor']:.3f}", f"{r['pv_fcff'] / 1e9:,.1f}"] for r in d["rows"]]
            parts.append(f"<p class=cap>DCF projection (USD bn; FCF margin {_p(d['fcf_margin'])} of revenue)</p>" +
                         _tbl(["Year", "Revenue growth", "Revenue", "FCFF", "Discount factor", "PV of FCFF"], rows))
            parts.append(_tbl(["PV of FCFF", "PV of terminal value", "Terminal share of EV", "Enterprise value",
                               "Equity value", "Value per share"],
                              [[f"{d['pv_fcff'] / 1e9:,.1f}", f"{d['pv_terminal'] / 1e9:,.1f}",
                                _p(d["terminal_share_of_ev"], 0), f"{d['enterprise_value'] / 1e9:,.1f}",
                                f"{d['equity_value'] / 1e9:,.1f}", _m(d["per_share"])]]))
        if v.get("scenarios"):
            parts.append("<p class=cap>Scenarios</p>" + _tbl(
                ["Case", "Revenue growth vs base", "FCF margin", "Value per share", "vs price"],
                [[s["case"], _p(s["growth_shift"], 0, True), _p(s["margin"]), _m(s["value"]),
                  _p(s["value"] / v["price"] - 1, 0, True) if v.get("price") else "–"] for s in v["scenarios"]]))
        rd = v.get("reverse_dcf") or {}
        if rd.get("implied_growth") is not None:
            g1, g2 = rd["consensus_growth_y1_y2"]
            parts.append(f"<p class=method><b>Reverse DCF:</b> at the current price of {_m(v['price'])}, the DCF "
                         f"requires revenue growth of <b>{_p(rd['implied_growth'])} a year for {rd['years']} years</b> "
                         f"(same margin, WACC and terminal growth), versus analyst consensus of {_p(g1)} and "
                         f"{_p(g2)} for the next two fiscal years.</p>")
        if v.get("comps"):
            peers = v.get("peers") or {}
            keys = [k for k in ("forward_pe", "ev_ebitda", "ev_sales", "pb") if k in v["comps"]]
            labels = {k: v["comps"][k]["label"] for k in keys}
            rows = [[_e(p), _e(d.get("name", ""))] + [_x(d.get(k)) for k in keys] for p, d in peers.items()]
            rows.append(["<b>Median</b>", ""] + [f"<b>{_x(v['comps'][k]['peer']['median'])}</b>" for k in keys])
            rows.append([f"<b>{_e(t)}</b>", "this company"] + [_x(v["comps"][k].get("own")) for k in keys])
            parts.append(f"<p class=cap>Peer multiples — {_e((v.get('peer_group') or {}).get('rationale', ''))}</p>" +
                         _tbl(["Ticker", "Name"] + [labels[k] for k in keys], rows))
            parts.append(_tbl(["Multiple", "Implied at peer 25th pct", "at median", "at 75th pct"],
                              [[_e(c["label"]), _m(c["implied"]["q1"]), _m(c["implied"]["median"]),
                                _m(c["implied"]["q3"])] for c in v["comps"].values()]))
        for key, title, cols in (("ri", "Residual income model", ("eps", "book_begin", "residual_income", "pv_ri")),
                                 ("ddm", "Dividend discount model", ("dividend", "discount_factor", "pv_dividend"))):
            r = v.get(key)
            if r:
                rows = [[f"Y{x['year']}"] + [f"{x[c]:,.3f}" if c == "discount_factor" else _m(x[c]) for c in cols]
                        for x in r["rows"]]
                parts.append(f"<p class=cap>{title} (per share, USD)</p>" +
                             _tbl(["Year"] + [c.replace("_", " ").title() for c in cols], rows))
        if v.get("valuation_gap_note"):
            parts.append(f"<p class=method><b>Note:</b> {_e(v['valuation_gap_note'])}</p>")
    return "\n".join(parts)


def appendix_sources(valuations: dict) -> str:
    parts = []
    for t, v in valuations.items():
        rows = [[_e(a["name"]), _e(a["value"]), _e(a["source"])] for a in v["assumptions"]]
        parts.append((f"<p class=cap>{_e(t)}</p>" if len(valuations) > 1 else "") +
                     _tbl(["Assumption", "Value", "Source"], rows, cls="fin src"))
    return "\n".join(parts)


VALUATION_CSS = """
table.mini { width:100%; border-collapse:collapse; font-size:10.5px; border:1px solid var(--line); border-top:0; }
table.mini th { background:#F2F2F2; color:var(--dark); text-align:right; padding:3px 5px; }
table.mini th:first-child, table.mini td:first-child { text-align:left; }
table.mini td { text-align:right; padding:2px 5px; border-bottom:1px solid #F2F2F2; }
table.mini tr.total td { font-weight:700; border-top:1px solid var(--grey); }
table.mini tr.grp td { font-weight:800; color:var(--red); background:#FAFAFA; }
table.mini small { color:var(--grey); font-size:9px; }
table.mini td.base { outline:2px solid var(--red); font-weight:700; }
table.sens th:first-child { white-space:nowrap; }
.mini-note { font-size:9.5px; font-style:italic; color:var(--grey); margin:2px 0 0; text-align:right; }
.fair { font-size:13px; margin:2px 0 4px; }
.fair b { color:var(--deep); }
p.cap { font-weight:800; font-size:12px; margin:10px 0 4px; }
.appendix h4 { color:var(--red); margin:14px 0 4px; }
table.src td { white-space:normal; text-align:left; }
table.src td:nth-child(2) { white-space:nowrap; }
"""
