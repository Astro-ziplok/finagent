"""
src/report/figures.py — Sidebar figures for the research report
================================================================
Small, consistently styled charts sized for the report's left sidebar,
in the red/grey palette of a sell-side research note. Every function
returns True if it saved a figure, or False when the data needed for it
is missing (the report then simply leaves that figure out).

Works for one ticker (company report) or several (comparison report).
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("matplotlib").setLevel(logging.ERROR)

RED, DARK, PINK, GREY, DEEP, LIGHT = "#D71920", "#404040", "#F2A0A3", "#A6A6A6", "#8C1016", "#D9D9D9"
SERIES = [RED, DARK, PINK, GREY, DEEP, "#6E6E6E"]
FIGSIZE = (4.2, 2.45)
DPI = 200


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(labelsize=7, colors=DARK, length=2)
    ax.grid(axis="y", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> bool:
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return True


def _legend(ax, ncol=3):
    ax.legend(fontsize=6.5, frameon=False, ncol=ncol, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), handlelength=1.2, columnspacing=1.0)


def _fy(period: str) -> str:
    return f"FY{str(period)[2:4]}"


def _chron(rows: list) -> list:
    """Statement rows come most-recent-first; charts read left to right in time."""
    return list(reversed(rows or []))


def _bn(v):
    return (v or 0) / 1e9


# ─────────────────────────────────────────────────────────────────────────────
#  Market figures
# ─────────────────────────────────────────────────────────────────────────────

def price_vs_benchmark(cleaned: dict, bench_df, bench_label: str, path: Path) -> bool:
    """Share price(s) rebased to 100 against the market benchmark."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    start = max(df.index[0] for df in cleaned.values())
    for i, (t, df) in enumerate(cleaned.items()):
        s = df["Close"][df.index >= start]
        ax.plot(s.index, s / s.iloc[0] * 100, color=SERIES[i % len(SERIES)], lw=1.3, label=t)
    if bench_df is not None and not bench_df.empty:
        b = bench_df["Close"][bench_df.index >= start]
        if len(b) > 1:
            ax.plot(b.index, b / b.iloc[0] * 100, color=GREY, lw=1.1, ls="--", label=bench_label)
    ax.axhline(100, color=DARK, lw=0.5)
    _style(ax)
    ax.tick_params(axis="x", rotation=0)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%y"))
    _legend(ax, ncol=4)
    return _save(fig, path)


def quarterly_returns(quarters: list, bench_quarters: list, ticker: str,
                      bench_label: str, path: Path) -> bool:
    """Calendar-quarter returns for the stock vs the benchmark."""
    if not quarters:
        return False
    labels = [q["period"].replace("20", "", 1).replace("-", "") for q in quarters]
    stock = [q["pct_change"] for q in quarters]
    bmap = {q["period"]: q["pct_change"] for q in (bench_quarters or [])}
    bench = [bmap.get(q["period"], np.nan) for q in quarters]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = 0.38
    ax.bar(x - w / 2, stock, w, color=RED, label=ticker)
    if any(not np.isnan(v) for v in bench):
        ax.bar(x + w / 2, bench, w, color=GREY, label=bench_label)
    for xi, v in zip(x, stock):
        ax.text(xi - w / 2, v + (0.6 if v >= 0 else -0.6), f"{v:.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=6, color=DARK)
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _style(ax)
    _legend(ax)
    return _save(fig, path)


def risk_return(ctx: dict, path: Path) -> bool:
    """Comparison view: annualised volatility vs total return per ticker."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (t, c) in enumerate(ctx.items()):
        vol = c["returns"]["annualised_volatility_pct"]
        ret = c["price_summary"]["total_return_pct"]
        ax.scatter(vol, ret, s=60, color=SERIES[i % len(SERIES)], zorder=3)
        ax.annotate(t, (vol, ret), textcoords="offset points", xytext=(5, 4), fontsize=7, color=DARK)
    ax.axhline(0, color=DARK, lw=0.6)
    ax.margins(x=0.18, y=0.18)  # keep point labels inside the frame
    ax.set_xlabel("Annualised volatility (%)", fontsize=7)
    ax.set_ylabel("Total return (%)", fontsize=7)
    _style(ax)
    return _save(fig, path)


def drawdown(cleaned: dict, bench_df, bench_label: str, path: Path) -> bool:
    """Drawdown from running peak: how deep and how long the falls were."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    single = len(cleaned) == 1
    for i, (t, df) in enumerate(cleaned.items()):
        c = df["Close"]
        dd = (c / c.cummax() - 1) * 100
        color = SERIES[i % len(SERIES)]
        if single:
            ax.fill_between(dd.index, dd, 0, color=PINK, alpha=0.8, lw=0)
        ax.plot(dd.index, dd, color=color, lw=1.0, label=t)
    if bench_df is not None and not bench_df.empty:
        b = bench_df["Close"][bench_df.index >= min(df.index[0] for df in cleaned.values())]
        if len(b) > 1:
            ax.plot(b.index, (b / b.cummax() - 1) * 100, color=GREY, lw=1.0, ls="--", label=bench_label)
    _style(ax)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%y"))
    _legend(ax, ncol=4)
    return _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Fundamental figures (single company: by year; comparison: latest year)
# ─────────────────────────────────────────────────────────────────────────────

def revenue_net_income(yf: dict, path: Path) -> bool:
    if len(yf) == 1:
        t, d = next(iter(yf.items()))
        rows = _chron((d.get("income") or {}).get("annual"))
        rows = [r for r in rows if r.get("total_revenue")]
        if not rows:
            return False
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=FIGSIZE)
        w = 0.38
        ax.bar(x - w / 2, [_bn(r["total_revenue"]) for r in rows], w, color=RED, label="Revenue ($bn)")
        ax.bar(x + w / 2, [_bn(r.get("net_income")) for r in rows], w, color=DARK, label="Net income ($bn)")
        ax.set_xticks(x)
        ax.set_xticklabels([_fy(r["period"]) for r in rows])
        _style(ax)
        ax2 = ax.twinx()
        ax2.plot(x, [r.get("net_margin_pct") or np.nan for r in rows], color=PINK, marker="o",
                 ms=3, lw=1.3, label="Net margin (%, RHS)")
        ax2.tick_params(labelsize=7, colors=DARK, length=2)
        ax2.spines["top"].set_visible(False)
        ax2.set_ylim(bottom=0)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14), handlelength=1.2)
        return _save(fig, path)
    # Comparison: latest revenue growth and net margin per ticker
    names, growth, margin = [], [], []
    for t, d in yf.items():
        inc = d.get("income") or {}
        a = inc.get("annual") or []
        if not a:
            continue
        names.append(t)
        growth.append((inc.get("yoy_growth") or {}).get("revenue_growth_pct") or np.nan)
        margin.append(a[0].get("net_margin_pct") or np.nan)
    if len(names) < 2:
        return False
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = 0.38
    ax.bar(x - w / 2, growth, w, color=RED, label="Revenue growth, latest FY (%)")
    ax.bar(x + w / 2, margin, w, color=DARK, label="Net margin, latest FY (%)")
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    _style(ax)
    _legend(ax, ncol=2)
    return _save(fig, path)


def margins(yf: dict, path: Path) -> bool:
    keys = [("gross_margin_pct", "Gross", RED), ("operating_margin_pct", "Operating", DARK),
            ("net_margin_pct", "Net", PINK)]
    if len(yf) == 1:
        rows = _chron((next(iter(yf.values())).get("income") or {}).get("annual"))
        if not rows:
            return False
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for k, label, color in keys:
            vals = [r.get(k) if r.get(k) is not None else np.nan for r in rows]
            ax.plot(x, vals, color=color, marker="o", ms=3, lw=1.4, label=f"{label} margin")
            if not np.isnan(vals[-1]):
                ax.annotate(f"{vals[-1]:.1f}", (x[-1], vals[-1]), textcoords="offset points",
                            xytext=(4, 0), fontsize=6, color=color, va="center")
        ax.set_xticks(x)
        ax.set_xticklabels([_fy(r["period"]) for r in rows])
        _style(ax)
        _legend(ax)
        return _save(fig, path)
    names = [t for t, d in yf.items() if (d.get("income") or {}).get("annual")]
    if len(names) < 2:
        return False
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (k, label, color) in enumerate(keys):
        vals = [yf[t]["income"]["annual"][0].get(k) or np.nan for t in names]
        ax.bar(x + (i - 1) * w, vals, w, color=color, label=f"{label} margin")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    _style(ax)
    _legend(ax)
    return _save(fig, path)


def cash_flow(yf: dict, path: Path) -> bool:
    if len(yf) == 1:
        rows = _chron((next(iter(yf.values())).get("cashflow") or {}).get("annual"))
        rows = [r for r in rows if r.get("operating_cf") is not None]
        if not rows:
            return False
        x = np.arange(len(rows))
        w = 0.27
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.bar(x - w, [_bn(r["operating_cf"]) for r in rows], w, color=DARK, label="Operating CF")
        ax.bar(x, [abs(_bn(r.get("capex"))) for r in rows], w, color=GREY, label="Capex")
        ax.bar(x + w, [_bn(r.get("free_cash_flow")) for r in rows], w, color=RED, label="Free cash flow")
        ax.axhline(0, color=DARK, lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([_fy(r["period"]) for r in rows])
        _style(ax)
        _legend(ax)
        return _save(fig, path)
    names, fcf_yield = [], []
    for t, d in yf.items():
        fy = ((d.get("metrics") or {}).get("profitability") or {}).get("fcf_yield_pct")
        if fy is not None:
            names.append(t)
            fcf_yield.append(fy)
    if len(names) < 2:
        return False
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(names, fcf_yield, color=[SERIES[i % len(SERIES)] for i in range(len(names))], width=0.5)
    ax.axhline(3, color=DARK, lw=0.7, ls="--")
    ax.text(len(names) - 0.5, 3.1, "3% reference", fontsize=6, ha="right", va="bottom", color=DARK)
    for i, v in enumerate(fcf_yield):
        ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.5)
    _style(ax)
    return _save(fig, path)


def balance_sheet(yf: dict, path: Path) -> bool:
    if len(yf) == 1:
        rows = _chron((next(iter(yf.values())).get("balance") or {}).get("annual"))
        rows = [r for r in rows if r.get("total_debt") is not None or r.get("cash") is not None]
        if not rows:
            return False
        x = np.arange(len(rows))
        w = 0.38
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.bar(x - w / 2, [_bn(r.get("cash")) for r in rows], w, color=GREY, label="Cash ($bn)")
        ax.bar(x + w / 2, [_bn(r.get("total_debt")) for r in rows], w, color=RED, label="Total debt ($bn)")
        ax.set_xticks(x)
        ax.set_xticklabels([_fy(r["period"]) for r in rows])
        _style(ax)
        ax2 = ax.twinx()
        ax2.plot(x, [r.get("debt_to_equity_x") if r.get("debt_to_equity_x") is not None else np.nan
                     for r in rows], color=DARK, marker="o", ms=3, lw=1.3, label="Debt/equity (x, RHS)")
        ax2.tick_params(labelsize=7, colors=DARK, length=2)
        ax2.spines["top"].set_visible(False)
        ax2.set_ylim(bottom=0)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14), handlelength=1.2)
        return _save(fig, path)
    names, de, cr = [], [], []
    for t, d in yf.items():
        fh = (d.get("metrics") or {}).get("financial_health") or {}
        if fh.get("debt_to_equity_x") is None and fh.get("current_ratio_x") is None:
            continue
        names.append(t)
        de.append(fh.get("debt_to_equity_x") if fh.get("debt_to_equity_x") is not None else np.nan)
        cr.append(fh.get("current_ratio_x") if fh.get("current_ratio_x") is not None else np.nan)
    if len(names) < 2:
        return False
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - w / 2, de, w, color=RED, label="Debt/equity (x)")
    ax.bar(x + w / 2, cr, w, color=GREY, label="Current ratio (x)")
    ax.axhline(1, color=DARK, lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    _style(ax)
    _legend(ax, ncol=2)
    return _save(fig, path)


def shareholder_returns(yf: dict, path: Path) -> bool:
    """Single company: buybacks + dividends against free cash flow."""
    if len(yf) != 1:
        return False
    rows = _chron((next(iter(yf.values())).get("cashflow") or {}).get("annual"))
    rows = [r for r in rows if r.get("share_buybacks") or r.get("dividends_paid")]
    if not rows:
        return False
    x = np.arange(len(rows))
    bb = [abs(_bn(r.get("share_buybacks"))) for r in rows]
    dv = [abs(_bn(r.get("dividends_paid"))) for r in rows]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x, bb, 0.5, color=RED, label="Buybacks")
    ax.bar(x, dv, 0.5, bottom=bb, color=PINK, label="Dividends")
    ax.plot(x, [_bn(r.get("free_cash_flow")) for r in rows], color=DARK, marker="o", ms=3,
            lw=1.3, label="Free cash flow")
    ax.set_xticks(x)
    ax.set_xticklabels([_fy(r["period"]) for r in rows])
    _style(ax)
    _legend(ax)
    return _save(fig, path)


def valuation_multiples(yf: dict, path: Path) -> bool:
    keys = [("pe_trailing", "P/E trailing"), ("pe_forward", "P/E forward"),
            ("ps_ratio", "P/S"), ("pb_ratio", "P/B")]
    vals = {t: (d.get("metrics") or {}).get("valuation") or {} for t, d in yf.items()}
    if not any(v.get(k) is not None for v in vals.values() for k, _ in keys):
        return False
    fig, ax = plt.subplots(figsize=FIGSIZE)
    y = np.arange(len(keys))
    n = len(vals)
    h = 0.8 / n
    for i, (t, v) in enumerate(vals.items()):
        data = [v.get(k) if v.get(k) is not None else 0 for k, _ in keys]
        ypos = y + (i - (n - 1) / 2) * h
        ax.barh(ypos, data, h * 0.92, color=SERIES[i % len(SERIES)], label=t, zorder=2)
        for yy, k_v, (k, _) in zip(ypos, data, keys):
            ax.text(k_v + 0.3, yy, "n/a" if v.get(k) is None else f"{k_v:.1f}x",
                    va="center", fontsize=6, color=DARK)
    # Typical S&P 500 P/E range for reference (approximate, 20-25x)
    ax.axvspan(20, 25, ymin=0.5, ymax=1.0, color=LIGHT, alpha=0.9, lw=0, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl in keys])
    ax.invert_yaxis()
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7, colors=DARK, length=2)
    if n > 1:
        _legend(ax, ncol=n)
    return _save(fig, path)


def analyst_targets(yf: dict, path: Path) -> bool:
    """Analyst target range (low / mean / high) as % from the current price."""
    rows = []
    for t, d in yf.items():
        m = d.get("metrics") or {}
        a, v = m.get("analyst") or {}, m.get("valuation") or {}
        price = v.get("current_price")
        if not price or not a.get("target_price"):
            continue
        pct = lambda x: (x / price - 1) * 100 if x else np.nan
        rows.append((t, pct(a.get("target_low")), pct(a.get("target_price")), pct(a.get("target_high"))))
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 0.7 + 0.55 * len(rows) + 0.6))
    for i, (t, lo, mean, hi) in enumerate(rows):
        if not np.isnan(lo) and not np.isnan(hi):
            ax.plot([lo, hi], [i, i], color=GREY, lw=5, solid_capstyle="butt")
            ax.text(lo, i + 0.28, f"{lo:+.0f}%", fontsize=6, ha="center", color=DARK)
            ax.text(hi, i + 0.28, f"{hi:+.0f}%", fontsize=6, ha="center", color=DARK)
        ax.scatter([mean], [i], color=RED, s=40, zorder=3)
        ax.text(mean, i - 0.32, f"mean {mean:+.1f}%", fontsize=6, ha="center", color=RED, va="top")
    ax.axvline(0, color=DARK, lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("% vs current price (0 = today)", fontsize=7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7, colors=DARK, length=0)
    return _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Risk matrix (drawn after the narrative, from the risks the analysis lists)
# ─────────────────────────────────────────────────────────────────────────────

_LEVEL = {"low": 0.5, "medium": 1.5, "high": 2.5}


def risk_matrix(risks: list, path: Path) -> bool:
    pts = [(r.get("code", "?"), _LEVEL.get(str(r.get("impact_level", "")).lower()),
            _LEVEL.get(str(r.get("likelihood", "")).lower())) for r in risks or []]
    pts = [p for p in pts if p[1] is not None and p[2] is not None]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 3.0))
    # Red gradient background: darker = higher impact x likelihood
    grad = np.add.outer(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    ax.imshow(grad, extent=(0, 3, 0, 3), origin="lower", cmap="Reds", alpha=0.85, vmin=0, vmax=2.4)
    placed = {}
    for code, x, y in pts:
        k = (x, y)
        off = placed.get(k, 0)
        placed[k] = off + 1
        cx, cy = x + (off % 3 - 1) * 0.32, y + (off // 3) * 0.3
        ax.scatter([cx], [cy], s=330, color="white", edgecolor=DEEP, lw=1, zorder=3)
        ax.text(cx, cy, code, ha="center", va="center", fontsize=6.5, fontweight="bold", color=DEEP, zorder=4)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Impact", fontsize=7)
    ax.set_ylabel("Likelihood", fontsize=7)
    ax.tick_params(labelsize=7, colors=DARK, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, path)
