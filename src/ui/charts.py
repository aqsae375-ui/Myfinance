"""Plotly figure builders. Each takes analytics/advisor objects and only draws them: no arithmetic on
figures (hover text is formatted with src/ui/format.py so it matches the rest of the app)."""
from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go

from src.analytics.schema import DailySpending, MerchantTotal, MonthComparison, MonthlySummary, MonthRow
from src.ui import format as fmt
from src.ui.theme import MUTED, SPARK_HEIGHT, TEAL, WARNING, apply, category_color, palette, tint

HOVER = "%{customdata}<extra></extra>"
CONFIG = {"displayModeBar": False, "responsive": True}
COMPARE_ROW_PX = 38  # comparison chart height per category, so bars and labels never overlap
COMPARE_LEGEND = "Lighter bar = previous month, solid bar = this month"
PREV_TINT = {False: 0.6, True: 0.45}  # how far last month's bar is blended toward white (light, dark mode)
MERCHANT_MAX_CHARS = 24


def sparkline(months: Sequence[str], values: Sequence[float | None], mode: str, as_pct: bool = False) -> go.Figure:
    pts = [(m, v) for m, v in zip(months, values) if v is not None]
    xs, ys = [m for m, _ in pts], [v for _, v in pts]
    hover = [f"{fmt.month_label(m, short=True)}: {fmt.pct(v) if as_pct else fmt.money(v)}" for m, v in pts]
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines+markers", line=dict(color=TEAL, width=2, shape="spline", smoothing=0.6),
        marker=dict(size=[0] * (len(xs) - 1) + [7], color=TEAL), fill="tozeroy",
        fillcolor="rgba(20,184,166,0.10)", customdata=hover, hovertemplate=HOVER))
    apply(fig, mode, height=SPARK_HEIGHT, margin=dict(l=0, r=0, t=4, b=0), showlegend=False)
    fig.update_xaxes(visible=False)
    lo, hi = (min(ys), max(ys)) if ys else (0, 1)
    pad = (hi - lo) * 0.15 or abs(hi) * 0.1 or 1
    fig.update_yaxes(visible=False, showgrid=False, range=[lo - pad, hi + pad])
    return fig


def category_donut(summary: MonthlySummary, mode: str, legend: bool = True) -> go.Figure:
    p = palette(mode)
    cats = list(summary.categories)
    fig = go.Figure(go.Pie(
        labels=[c.category for c in cats], values=[c.amount for c in cats], hole=0.64, sort=False,
        marker=dict(colors=[category_color(c.category) for c in cats], line=dict(color=p["card"], width=2)),
        textinfo="none", direction="clockwise",
        customdata=[f"<b>{c.category}</b><br>{fmt.money(c.amount)} · {fmt.pct(c.pct_of_spend)} of spending"
                    for c in cats],
        hovertemplate=HOVER))
    apply(fig, mode, margin=dict(l=8, r=8, t=8, b=8), showlegend=legend, height=260 if not legend else 320,
          legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.02, xanchor="left"))
    big, small = (20, 12) if legend else (15, 11)  # a smaller donut (legend beside it) gets a smaller centre
    fig.add_annotation(text=f"<span style='font-size:{small}px;color:{p['muted']}'>Total spent</span><br>"
                            f"<b style='font-size:{big}px'>{fmt.money(summary.total_spent)}</b>",
                       showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font=dict(color=p["text"]))
    return fig


def category_bars(mom: MonthComparison, mode: str) -> go.Figure:
    """This month vs last month per category, as horizontal grouped bars sorted by this month (largest on top).
    Last month = pale tint of the category color, this month = the solid color. Category names are left-aligned
    annotations in a gutter (Plotly tick labels can only be right-aligned); every bar is labelled "Rs 1,650",
    so small categories stay readable next to a large one such as Rent on a zero-based, unclipped axis."""
    p = palette(mode)
    cats = sorted((c for c in mom.categories if c.amount.current or c.amount.previous),
                  key=lambda c: (-(c.amount.current or 0), -(c.amount.previous or 0), c.category))
    names = [c.category for c in cats]
    solid = [category_color(n) for n in names]
    has_prev = bool(mom.previous_month)
    cur_label = fmt.month_label(mom.month, short=True)
    label_font = dict(size=11, color=p["muted"])
    fig = go.Figure()
    if has_prev:
        prev_label = fmt.month_label(mom.previous_month, short=True)
        prev = [c.amount.previous or 0 for c in cats]
        fig.add_bar(name=prev_label, orientation="h", y=names, x=prev,
                    marker=dict(color=[tint(c, PREV_TINT[mode == "dark"]) for c in solid]),
                    text=[fmt.money(v) if v else "" for v in prev], textposition="outside", cliponaxis=False, textfont=label_font,
                    customdata=[f"<b>{c.category}</b><br>{prev_label}: {fmt.money(c.amount.previous or 0)}"
                                for c in cats], hovertemplate=HOVER)
    cur = [c.amount.current or 0 for c in cats]
    fig.add_bar(name=cur_label, orientation="h", y=names, x=cur, marker=dict(color=solid),
                text=[fmt.money(v) if v else "" for v in cur], textposition="outside", cliponaxis=False, textfont=label_font,
                customdata=[f"<b>{c.category}</b><br>{cur_label}: {fmt.money(c.amount.current or 0)}"
                            + (f"<br>vs last month: {fmt.money(c.amount.abs_change, signed=True)}"
                               if c.amount.abs_change is not None else "") for c in cats],
                hovertemplate=HOVER)

    gutter = min(0.4, max(0.16, 0.0135 * max((len(n) for n in names), default=4) + 0.03))
    top = max([v for t in fig.data for v in t.x] or [0]) or 1
    apply(fig, mode, height=max(120, COMPARE_ROW_PX * len(names) + (48 if has_prev else 20)),
          barmode="group", showlegend=False, bargap=0.22, bargroupgap=0.04,
          margin=dict(l=8, r=8, t=34 if has_prev else 8, b=8))
    # Room after the largest bar for its "Rs" label; the axis always starts at zero.
    fig.update_xaxes(domain=[gutter, 1], range=[0, top * 1.28], visible=False)
    fig.update_yaxes(categoryorder="array", categoryarray=names, autorange="reversed", showticklabels=False,
                     showgrid=False)
    for n in names:
        fig.add_annotation(x=0, xref="paper", xanchor="left", y=n, yref="y", text=n, showarrow=False,
                           font=dict(size=12, color=p["text"]), align="left")
    if has_prev:
        fig.add_annotation(x=0, xref="paper", xanchor="left", y=1, yref="paper", yanchor="bottom", yshift=8,
                           text=COMPARE_LEGEND, showarrow=False, font=dict(size=12, color=p["muted"]))
    return fig


def spending_trend(rows: Sequence[MonthRow], selected: str, anomaly_months: Sequence[str], mode: str) -> go.Figure:
    """Total spent per month: open-circle markers, the selected month in a soft teal band with its value,
    months with an unusual charge as orange points labelled "Unusual charge"."""
    p = palette(mode)
    # Short month ticks ("Mar"); "Mar '26" style only when a month name would repeat (over a year of data).
    xs = [fmt.month_tick(r.month) for r in rows]
    if len(set(xs)) < len(xs):
        xs = [fmt.month_tick(r.month, with_year=True) for r in rows]
    ys = [r.total_spent for r in rows]
    hover = [f"<b>{fmt.month_label(r.month)}</b><br>Spent {fmt.money(r.total_spent)}"
             + ("<br>⚠ unusual charge this month" if r.month in anomaly_months else "") for r in rows]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Total spent",
                               line=dict(color=TEAL, width=2.5),
                               marker=dict(size=10, color=p["card"], line=dict(color=TEAL, width=2.5)),
                               customdata=hover, hovertemplate=HOVER, showlegend=False))
    flagged = [(x, y) for r, x, y in zip(rows, xs, ys) if r.month in anomaly_months and r.month != selected]
    if flagged:
        fig.add_scatter(x=[x for x, _ in flagged], y=[y for _, y in flagged], mode="markers", name="Unusual charge",
                        marker=dict(size=11, color=WARNING, line=dict(color=p["card"], width=2)), hoverinfo="skip",
                        showlegend=False)
        for x, y in flagged:
            fig.add_annotation(x=x, y=y, text="<b>Unusual charge</b>", showarrow=False, yshift=16,
                               font=dict(size=11, color=WARNING))
    idx = next((i for i, r in enumerate(rows) if r.month == selected), None)
    if idx is not None:
        fig.add_vrect(x0=idx - 0.32, x1=idx + 0.32, fillcolor=TEAL, opacity=0.12, line_width=0, layer="below")
        fig.add_scatter(x=[xs[idx]], y=[ys[idx]], mode="markers", name="Selected month", hoverinfo="skip",
                        marker=dict(size=11, color=TEAL, line=dict(color=p["card"], width=2)), showlegend=False)
        fig.add_annotation(x=xs[idx], y=ys[idx], text=f"<b>{fmt.money(ys[idx])}</b>", showarrow=False, yshift=-20,
                           font=dict(size=12, color=p["text"]))
    lo, hi = (min(ys), max(ys)) if ys else (0, 1)
    pad = (hi - lo) * 0.25 or abs(hi) * 0.1 or 1
    apply(fig, mode, showlegend=False, hovermode="x", margin=dict(l=8, r=8, t=24, b=8))
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickprefix=f"{fmt.CURRENCY} ", tickformat=",.0f", range=[max(0, lo - pad), hi + pad])
    return fig


def top_merchants(merchants: Sequence[MerchantTotal], mode: str) -> go.Figure:
    rows = list(reversed(merchants))  # largest at the top
    p = palette(mode)
    names = [_short(m.merchant.title(), MERCHANT_MAX_CHARS) for m in rows]
    fig = go.Figure(go.Bar(
        orientation="h", y=names, x=[m.amount for m in rows],
        marker=dict(color=[category_color(m.category) for m in rows]),
        text=[fmt.money(m.amount) for m in rows], textposition="outside", cliponaxis=False,
        textfont=dict(color=p["muted"], size=12),
        customdata=[f"<b>{m.merchant.title()}</b><br>{m.category}<br>{fmt.money(m.amount)} · "
                    f"{m.txn_count} transaction{'s' if m.txn_count != 1 else ''}" for m in rows],
        hovertemplate=HOVER))
    apply(fig, mode, showlegend=False, margin=dict(l=8, r=8, t=8, b=8), bargap=0.35)
    fig.update_xaxes(visible=False, range=[0, max([m.amount for m in rows] or [0]) * 1.3 or 1])  # room for labels
    fig.update_yaxes(showgrid=False, tickfont=dict(color=p["text"]), ticklabelstandoff=6)
    return fig


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def daily_bars(daily: DailySpending, mode: str) -> go.Figure:
    """Spending per day; the highest day is orange (the card subtitle names it)."""
    peak = daily.peak
    fig = go.Figure(go.Bar(
        x=[d.date for d in daily.days], y=[d.amount for d in daily.days],
        marker=dict(color=[WARNING if peak and d.date == peak.date else TEAL for d in daily.days]),
        customdata=[f"<b>{fmt.day_label(d.date)}</b><br>{fmt.money(d.amount)} · {d.txn_count} transaction"
                    f"{'s' if d.txn_count != 1 else ''}" for d in daily.days],
        hovertemplate=HOVER))
    apply(fig, mode, showlegend=False, bargap=0.25)
    fig.update_xaxes(tickformat="%b %-d", dtick=7 * 24 * 3600 * 1000)
    fig.update_yaxes(tickprefix=f"{fmt.CURRENCY} ", tickformat=",.0f")
    return fig


def plan_paths(before: Sequence[float] | None, after: Sequence[float] | None, goal_line: float | None,
               goal_label: str, mode: str) -> go.Figure:
    """Cumulative savings over the next 12 months: current path (grey) vs plan path (teal), each labelled
    with its end value, plus the goal as a dashed orange line."""
    p = palette(mode)
    fig = go.Figure()
    months = list(range(len(after or before or [])))
    end = months[-1] if months else 0
    if before is not None:
        fig.add_scatter(x=months, y=before, name="Current path", mode="lines", line=dict(color=MUTED, width=2.5),
                        customdata=[f"Month {m} · current path: {fmt.money(v)}" for m, v in zip(months, before)],
                        hovertemplate=HOVER)
        fig.add_scatter(x=[end], y=[before[-1]], mode="markers+text", showlegend=False, hoverinfo="skip",
                        marker=dict(size=10, color=p["card"], line=dict(color=MUTED, width=2.5)),
                        text=[fmt.money(before[-1])], textposition="bottom left", textfont=dict(color=MUTED, size=12))
    if after is not None:
        fig.add_scatter(x=months, y=after, name="With plan", mode="lines", line=dict(color=TEAL, width=3),
                        customdata=[f"Month {m} · with plan: {fmt.money(v)}" for m, v in zip(months, after)],
                        hovertemplate=HOVER)
        fig.add_scatter(x=[end], y=[after[-1]], mode="markers+text", showlegend=False, hoverinfo="skip",
                        marker=dict(size=10, color=TEAL), text=[f"<b>{fmt.money(after[-1])}</b>"],
                        textposition="top left", textfont=dict(color=p["text"], size=12))
    if goal_line is not None:
        fig.add_scatter(x=[None], y=[None], mode="lines", name="Goal", line=dict(color=WARNING, width=2, dash="dash"))
        fig.add_hline(y=goal_line, line=dict(color=WARNING, width=2, dash="dash"),
                      annotation_text=f"<b>{goal_label}</b>", annotation_position="top left",
                      annotation_font=dict(color=WARNING, size=12))
    apply(fig, mode, showlegend=True, hovermode="x unified",
          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    if goal_line is not None:  # a shape does not stretch the axis: keep the goal line in view
        top = max([goal_line, *(before or []), *(after or [])])
        fig.update_yaxes(range=[min(0, *(before or [0]), *(after or [0])), top * 1.12])
    ticks = [m for m in months if m % 4 == 0]
    # A little room on both sides so "Now" clears the y labels and the end labels fit.
    fig.update_xaxes(tickvals=ticks, ticktext=["Now" if m == 0 else f"Month {m}" for m in ticks],
                     range=[-0.6, end + 0.9])
    fig.update_yaxes(tickprefix=f"{fmt.CURRENCY} ", tickformat=",.0f")
    if goal_line is None:
        fig.update_yaxes(rangemode="tozero")
    return fig
