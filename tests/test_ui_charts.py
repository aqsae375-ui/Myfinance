"""UI polish: product name, the horizontal month comparison chart, and no rotated labels in any chart."""
import re
from pathlib import Path

import pytest

from src import analytics
from src.ui import charts
from src.ui.theme import APP_NAME, TAGLINE, category_color, tint

ROOT = Path(__file__).resolve().parent.parent
MONTH = "2026-08"


def test_product_name_and_tagline():
    assert (APP_NAME, TAGLINE) == ("Myfinance", "Understand. Manage. Grow.")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "page_title=APP_NAME" in app
    css = (ROOT / "src" / "ui" / "components.py").read_text(encoding="utf-8")
    tag_rule = re.search(r"\.fa-header \.tag \{\{(.*?)\}\}", css, re.DOTALL).group(1)
    assert "white-space:nowrap" in tag_rule  # the tagline never wraps
    for path in [*ROOT.glob("*.py"), *(ROOT / "src").rglob("*.py"), *ROOT.glob("*.md"), *(ROOT / "scripts").glob("*")]:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "[PROJECT NAME]" not in text and "Smart money advice" not in text, path


def _all_figures(df, mode):
    mom = analytics.month_over_month(MONTH, df)
    history = analytics.month_history(df)
    months = [r.month for r in history]
    return {
        "sparkline": charts.sparkline(months, [r.total_spent for r in history], mode),
        "donut": charts.category_donut(mom.current, mode),
        "comparison": charts.category_bars(mom, mode),
        "trend": charts.spending_trend(history, MONTH, ["2026-07"], mode),
        "merchants": charts.top_merchants(analytics.top_merchants(MONTH, txns=df), mode),
        "daily": charts.daily_bars(analytics.daily_spending(MONTH, df), mode),
        "plan": charts.plan_paths([0, 100, 200], [0, 150, 300], 250.0, "Goal", mode),
    }


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_no_chart_uses_a_non_zero_tick_angle(synthetic_df, mode):
    for name, fig in _all_figures(synthetic_df, mode).items():
        axes = [k for k in fig.layout.to_plotly_json() if re.fullmatch(r"[xy]axis\d*", k)]
        assert axes, name
        for ax in axes:
            assert fig.layout[ax].tickangle == 0, (name, ax)
        assert fig.layout.template.layout.xaxis.tickangle == 0 == fig.layout.template.layout.yaxis.tickangle


def test_no_source_file_sets_a_non_zero_tick_angle():
    for path in (ROOT / "src" / "ui").glob("*.py"):
        for value in re.findall(r"tickangle\s*=\s*([^,)\s]+)", path.read_text(encoding="utf-8")):
            assert value == "0", (path.name, value)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_comparison_chart_is_horizontal_sorted_and_labelled(synthetic_df, mode):
    mom = analytics.month_over_month(MONTH, synthetic_df)
    fig = charts.category_bars(mom, mode)
    prev, cur = fig.data
    assert prev.orientation == cur.orientation == "h"
    assert fig.layout.xaxis.tickangle == 0 and fig.layout.yaxis.tickangle == 0

    cats = [c for c in mom.categories if c.amount.current or c.amount.previous]
    names = list(cur.y)
    assert sorted(names) == sorted(c.category for c in cats)
    assert list(cur.x) == sorted(cur.x, reverse=True)                  # this month, largest first
    assert fig.layout.yaxis.autorange == "reversed"                    # first category drawn at the top
    assert list(fig.layout.yaxis.categoryarray) == names

    # Same fixed category colors; previous month is a pale tint of them.
    assert list(cur.marker.color) == [category_color(n) for n in names]
    assert list(prev.marker.color) == [tint(category_color(n), charts.PREV_TINT[mode == "dark"]) for n in names]
    assert all(t == "" or re.fullmatch(r"Rs \d{1,3}(,\d{3})*(\.\d{2})?", t) for t in [*cur.text, *prev.text])
    assert "Rs 1,650" in cur.text                                      # rent, formatted like the rest of the app
    assert cur.cliponaxis is False and cur.textposition == "outside"

    # Legend: plain text at the top, no trace legend squares.
    assert fig.layout.showlegend is False
    legend = [a for a in fig.layout.annotations if a.text == charts.COMPARE_LEGEND]
    assert len(legend) == 1 and legend[0].yref == "paper" and legend[0].y == 1
    # Category names: normal horizontal text, left aligned.
    labels = [a for a in fig.layout.annotations if a.text in names]
    assert len(labels) == len(names) and all(a.xanchor == "left" and not a.textangle for a in labels)

    assert fig.layout.height == charts.COMPARE_ROW_PX * len(names) + 48
    lo, hi = fig.layout.xaxis.range
    assert lo == 0 and hi > max([*cur.x, *prev.x])                     # starts at zero, never clipped


def test_comparison_chart_without_previous_month(synthetic_df):
    mom = analytics.month_over_month("2026-03", synthetic_df)  # first month: nothing to compare with
    fig = charts.category_bars(mom, "light")
    assert len(fig.data) == 1 and fig.data[0].orientation == "h"
    assert not [a for a in fig.layout.annotations if a.text == charts.COMPARE_LEGEND]


def test_trend_uses_short_month_labels(synthetic_df):
    fig = charts.spending_trend(analytics.month_history(synthetic_df), MONTH, [], "light")
    assert list(fig.data[0].x) == ["Mar", "Apr", "May", "Jun", "Jul", "Aug"]


def test_trend_adds_the_year_only_when_month_names_repeat(make_txns):
    rows = [(f"{y}-{m:02d}-05", "SHOP", -10.0, "Shopping") for y, m in [(2025, 9), (2025, 10), (2025, 11),
            (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5), (2026, 6), (2026, 7), (2026, 8),
            (2026, 9)]]
    fig = charts.spending_trend(analytics.month_history(make_txns(rows)), "2026-09", [], "light")
    xs = list(fig.data[0].x)
    assert xs[0] == "Sep '25" and xs[-1] == "Sep '26" and len(set(xs)) == len(xs)


def test_tint_is_paler():
    assert tint("#1E3A8A", 0) == "#1E3A8A" and tint("#1E3A8A", 1) == "#FFFFFF"
    assert tint("#000000", 0.5) == "#808080"
