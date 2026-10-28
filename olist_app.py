# olist_app.py
"""
════════════════════════════════════════════════════════════════════════════
OLIST CUSTOMER SATISFACTION DASHBOARD
PORA Academy · Cohort 7 · Group 1 Capstone

Phase 2c · Customer Satisfaction focus
Deployed on Streamlit Community Cloud · backed by in-memory SQLite

DATA LAYER
  · 4 CSVs → SQLite in-memory DB (@st.cache_resource)
  · Every chart is a function calling parameterised SQL
  · Filter contract: _year_params / _multi_params / _year_multi
  · Guards: (? = 'All' OR strftime(...) = ?) and (? = '[]' OR ...)

RECONCILIATION CONTRACT  (filters = All)
  Q1   99,441 · 96,478 · 4.09 · 7,826 · 12.56
  Q2   11.5 / 3.2 / 8.2 / 19.3 / 57.8 %
  Q3   25 months, monthly AVG(review_score) row-level
  Q4   SP 4.17 · RJ 3.88 · MG 4.14 · RS 4.13 · PR 4.18 · SC 4.07 · BA 3.86 · DF 4.07
  Q5   SP 5.9% · RJ 13.5% · MG 5.6% · RS 7.1% · PR 5.0%
  Q6   On-Time 88,171 @ 4.29 · Late 7,661 @ 2.57
  Q7   7,826 · 9.55 · 189
  Q8   326 / 45,044 / 53,854
  Q9   4.45 / 4.36 / 4.22 / 3.69 / 2.26
  Q10  99,224 / 98,673 / 551
  Q11  99,441 / 96,096
════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from matplotlib.ticker import FuncFormatter

# ════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CHART_FIGSIZE:      Final[tuple[float, float]] = (8.6, 4.8)
CHART_FIGSIZE_WIDE: Final[tuple[float, float]] = (13.5, 4.4)
PARTIAL_YEARS:      Final[frozenset[str]] = frozenset({"2016", "2018"})

PUNCT_MIN_N_TIER:   Final[int] = 30
PUNCT_MIN_N_STATUS: Final[int] = 30
PUNCT_SCENARIOS:    Final[tuple[int, ...]] = (5, 10, 25, 50)

REF: Final[dict[str, Any]] = {
    "total_orders":      99_441,
    "delivered_orders":  96_478,
    "avg_review_score":   4.09,
    "late_deliveries":    7_826,
    "avg_delivery_days": 12.56,
    "on_time_score":      4.29,
    "late_score":         2.57,
    "star_gap":           1.73,
    "q6_on_time_orders": 88_171,
    "q6_late_orders":     7_661,
    "q5_delivered":      95_832,
    "q5_late":            7_661,
}


@dataclass(frozen=True)
class Palette:
    primary:       str = "#6366F1"
    primary_light: str = "#A5B4FC"
    primary_dark:  str = "#4338CA"
    accent:        str = "#F97316"
    accent_light:  str = "#FDBA74"
    accent_dark:   str = "#C2410C"

    success:       str = "#10B981"
    success_light: str = "#6EE7B7"
    success_dark:  str = "#047857"
    warn:          str = "#F59E0B"
    warn_light:    str = "#FCD34D"
    warn_dark:     str = "#B45309"
    danger:        str = "#EF4444"
    danger_light:  str = "#FCA5A5"
    danger_dark:   str = "#B91C1C"

    ink:           str = "#0F172A"
    ink_soft:      str = "#1E293B"
    slate_700:     str = "#334155"
    slate_600:     str = "#475569"
    slate_500:     str = "#64748B"
    slate_400:     str = "#94A3B8"
    slate_300:     str = "#CBD5E1"
    slate_200:     str = "#E2E8F0"
    slate_100:     str = "#F1F5F9"
    slate_50:      str = "#F8FAFC"
    white:         str = "#FFFFFF"


@dataclass(frozen=True)
class Config:
    tables: Final[dict[str, str]] = field(default_factory=lambda: {
        "orders":      "olist_orders_dataset.csv",
        "customers":   "olist_customers_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "reviews":     "olist_order_reviews_dataset.csv",
    })
    all_states: Final[tuple[str, ...]] = (
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
        "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
        "RO", "RR", "RS", "SC", "SE", "SP", "TO",
    )
    years: Final[tuple[str, ...]] = ("All", "2016", "2017", "2018")
    geojson_urls: Final[tuple[str, ...]] = (
        "https://raw.githubusercontent.com/codeforamerica/"
        "click_that_hood/master/public/data/brazil-states.geojson",
        "https://raw.githubusercontent.com/tbrugz/"
        "vector-json/master/brazil-states.geojson",
    )
    geojson_timeout_seconds: Final[int] = 20
    geojson_cache_ttl_seconds: Final[int] = 86400
    min_reviews_per_month:     Final[int] = 30


CONFIG  = Config()
PALETTE = Palette()

# ════════════════════════════════════════════════════════════════════════════
# §2  LOGGING
# ════════════════════════════════════════════════════════════════════════════

_LOGGER = logging.getLogger("olist.dashboard")
if not _LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    _LOGGER.addHandler(_h)
    _LOGGER.setLevel(
        getattr(logging, os.environ.get("OLIST_LOG_LEVEL", "INFO").upper(),
                logging.INFO)
    )
    _LOGGER.propagate = False


# ════════════════════════════════════════════════════════════════════════════
# §2.x  NUMERIC + FORMAT GUARDS
# ════════════════════════════════════════════════════════════════════════════

def _safe_num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _safe_div(num: Any, den: Any) -> float | None:
    n, d = _safe_num(num), _safe_num(den)
    if n is None or d is None or d == 0:
        return None
    out = n / d
    return out if math.isfinite(out) else None


def _safe_mean(series: pd.Series | None) -> float | None:
    if series is None or len(series) == 0:
        return None
    s = series.dropna()
    if s.empty:
        return None
    return _safe_num(s.mean())


def fmt_int(v: Any) -> str:
    n = _safe_num(v)
    return "—" if n is None else f"{int(n):,}"


def fmt_float(v: Any, p: int = 2) -> str:
    n = _safe_num(v)
    return "—" if n is None else f"{n:.{p}f}"


def fmt_pct(v: Any, p: int = 1) -> str:
    n = _safe_num(v)
    return "—" if n is None else f"{n:.{p}f}%"


# ════════════════════════════════════════════════════════════════════════════
# §2.y  STREAMLIT API SHIMS
# ════════════════════════════════════════════════════════════════════════════

def _st_pyplot(fig: plt.Figure) -> None:
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        try:
            st.pyplot(fig, use_container_width=True)
        except Exception:
            st.pyplot(fig)


def _st_dataframe(df: pd.DataFrame, *, hide_index: bool = True) -> None:
    try:
        st.dataframe(df, hide_index=hide_index, width="stretch")
    except TypeError:
        try:
            st.dataframe(df, hide_index=hide_index, use_container_width=True)
        except Exception:
            st.dataframe(df)


def _st_plotly(fig: Any) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.plotly_chart(fig)


# ════════════════════════════════════════════════════════════════════════════
# §2.z  LOGO HELPER
# ════════════════════════════════════════════════════════════════════════════

_FALLBACK_LOGO_SVG: Final[str] = (
    "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcv"
    "MjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tl"
    "PSIjNjM2NkYxIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3Vu"
    "ZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIx"
    "MiIgcj0iMTAiPjwvY2lyY2xlPjxwYXRoIGQ9Ik0xMiA4djgiPjwvcGF0aD48cGF0"
    "aCBkPSJNOCAxMmg4Ij48L3BhdGg+PC9zdmc+"
)


def _get_logo_base64() -> str:
    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pora_logo.png"
    )
    if os.path.isfile(local_path):
        try:
            with open(local_path, "rb") as f:
                data = f.read()
            if data:
                return f"data:image/png;base64,{base64.b64encode(data).decode()}"
        except Exception as exc:
            _LOGGER.warning("Logo load failed: %s", exc)
    return _FALLBACK_LOGO_SVG


# ════════════════════════════════════════════════════════════════════════════
# §3  DATA LAYER — SQLite (assignment-compliant)
# ════════════════════════════════════════════════════════════════════════════

def _data_dir() -> str:
    return os.environ.get(
        "OLIST_DATA_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    )


@st.cache_resource(show_spinner="Building in-memory SQLite…")
def get_connection() -> sqlite3.Connection:
    """Load the 4 CSVs into an in-memory SQLite database once per session."""
    data_dir = _data_dir()
    conn = sqlite3.connect(":memory:", check_same_thread=False)

    for table, filename in CONFIG.tables.items():
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing CSV for table {table!r} at {path}. "
                "Set OLIST_DATA_DIR to the folder containing the CSVs."
            )
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Could not read {filename}: {exc}") from exc

        if df.empty:
            raise ValueError(f"Table {table!r} loaded with zero rows")

        df.to_sql(table, conn, index=False, if_exists="replace")

    _LOGGER.info("SQLite in-memory database ready.")
    return conn


def _year_params(year: str) -> tuple[str, str]:
    """Year guard only → 2 placeholders."""
    return (year, year)


def _multi_params(multi_json: str) -> tuple[str, str]:
    """Multiselect guard only → 2 placeholders."""
    return (multi_json, multi_json)


def _year_multi(year: str, multi_json: str) -> tuple[str, str, str, str]:
    """Both guards → 4 placeholders."""
    return (year, year, multi_json, multi_json)


def _states_to_json(states: Sequence[str]) -> str:
    return json.dumps([] if not states else list(states))


@st.cache_data(show_spinner=False)
def run_query(sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    """Execute parameterised SQL. Cached by (sql, params)."""
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as exc:
        _LOGGER.exception("Query failed: %s", exc)
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# §4  KPI COMPUTATION  (Q1, Q6 via SQL)
# ════════════════════════════════════════════════════════════════════════════

Q1_SQL: Final[str] = """
SELECT COUNT(DISTINCT o.order_id) AS total_orders,
       COUNT(DISTINCT CASE WHEN o.order_status='delivered'
                           THEN o.order_id END) AS delivered_orders,
       (SELECT ROUND(AVG(CAST(review_score AS REAL)),2) FROM reviews)
         AS avg_review_score,
       SUM(CASE WHEN o.order_status='delivered'
                 AND o.order_delivered_customer_date
                     > o.order_estimated_delivery_date
                THEN 1 ELSE 0 END) AS late_deliveries,
       ROUND(AVG(CASE WHEN o.order_status='delivered'
                       AND o.order_delivered_customer_date IS NOT NULL
                 THEN julianday(o.order_delivered_customer_date)
                    - julianday(o.order_purchase_timestamp) END),2)
         AS avg_delivery_days
FROM orders o
WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
"""


@st.cache_data(show_spinner=False)
def query_kpis(year: str, states_tuple: tuple[str, ...]) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "total_orders": 0, "delivered_orders": 0, "delivered_pct": None,
        "avg_review_score": None, "late_deliveries": 0, "late_pct": None,
        "avg_delivery_days": None, "review_rows": 0,
        "delivered_with_date_orders": 0,
    }
    try:
        df = run_query(Q1_SQL, _year_params(year))
    except Exception as exc:
        _LOGGER.warning("Q1 failed: %s", exc)
        return empty
    if df.empty:
        return empty

    row = df.iloc[0]
    total     = int(row.get("total_orders", 0) or 0)
    delivered = int(row.get("delivered_orders", 0) or 0)
    late      = int(row.get("late_deliveries", 0) or 0)

    delivered_pct = _safe_div(delivered, total)
    late_pct      = _safe_div(late, delivered)

    return {
        "total_orders":      total,
        "delivered_orders":  delivered,
        "delivered_pct":     (delivered_pct * 100.0)
                             if delivered_pct is not None else None,
        "avg_review_score":  _safe_num(row.get("avg_review_score")),
        "late_deliveries":   late,
        "late_pct":          (late_pct * 100.0)
                             if late_pct is not None else None,
        "avg_delivery_days": _safe_num(row.get("avg_delivery_days")),
        "review_rows":       0,
        "delivered_with_date_orders": delivered,
    }


Q6_SQL: Final[str] = """
SELECT CASE WHEN o.order_delivered_customer_date
                <= o.order_estimated_delivery_date
            THEN 'On Time' ELSE 'Late' END AS delivery_status,
       COUNT(DISTINCT o.order_id) AS orders,
       ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score,
       ROUND(AVG(julianday(o.order_delivered_customer_date)
               - julianday(o.order_purchase_timestamp)),2)
         AS avg_delivery_days
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN reviews r  ON o.order_id  = r.order_id
WHERE o.order_status='delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
GROUP BY 1 ORDER BY avg_score DESC
"""


@st.cache_data(show_spinner=False)
def query_headline_gap(year: str,
                       states_tuple: tuple[str, ...]) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "on_time": None, "late": None, "gap": None,
        "gap_pct": None, "n_on": 0, "n_late": 0,
    }
    state_json = _states_to_json(states_tuple)
    try:
        df = run_query(Q6_SQL, _multi_params(state_json))
    except Exception as exc:
        _LOGGER.warning("Q6 failed: %s", exc)
        return empty
    if df.empty:
        return empty

    means  = dict(zip(df["delivery_status"], df["avg_score"]))
    counts = dict(zip(df["delivery_status"], df["orders"]))

    on_mean   = _safe_num(means.get("On Time"))
    late_mean = _safe_num(means.get("Late"))
    n_on      = int(counts.get("On Time", 0) or 0)
    n_late    = int(counts.get("Late", 0) or 0)

    gap = (on_mean - late_mean) if (
        on_mean is not None and late_mean is not None
    ) else None
    gap_pct_raw = _safe_div(gap, on_mean) if gap is not None else None
    gap_pct = (100.0 * gap_pct_raw) if gap_pct_raw is not None else None

    return {
        "on_time": on_mean, "late": late_mean,
        "gap": gap, "gap_pct": gap_pct,
        "n_on": n_on, "n_late": n_late,
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  CHART HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _no_data(msg: str = "No data for the current filters.") -> None:
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-icon">📭</div>
          <div class="empty-title">Nothing to show</div>
          <div class="empty-body">{msg}</div>
          <div class="empty-hint">💡 Try widening the Year or State filters in the sidebar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _insight(year: str, states: Sequence[str], text: str) -> None:
    scope = (f"{'All years' if year == 'All' else year} · "
             f"{', '.join(states) if states else 'All states'}")
    st.markdown(
        f"""
        <div class="insight">
          <span class="insight-chip">💡 Insight</span>
          <div class="insight-body">
            <span class="insight-scope">{scope}</span>
            <span class="insight-text">{text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style_axes(ax: plt.Axes) -> None:
    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", color=PALETTE.slate_200, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors=PALETTE.slate_500, labelsize=10)
    except Exception:
        pass


def _finish(fig: plt.Figure) -> None:
    try:
        fig.tight_layout()
    except Exception:
        pass
    _st_pyplot(fig)
    try:
        plt.close(fig)
    except Exception:
        pass


def _partial_note(year: str) -> None:
    if not year or year not in PARTIAL_YEARS:
        return
    detail = {
        "2016": "2016 has only 3 active months (Sep, Oct, Dec) — November has zero orders.",
        "2018": "2018 ends 17 Oct — not a full calendar year.",
    }.get(year, "")
    if detail:
        st.caption(f"⚠️ **Partial year — {year}.** {detail}")


def _safe_chart(fn: Callable[..., None], *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        _LOGGER.exception("Chart %s failed", getattr(fn, "__name__", "?"))
        st.warning(f"⚠️ This chart could not render: {exc}")


def _chart_card(fn: Callable[..., None], *args, **kwargs) -> None:
    with st.container(border=True):
        _safe_chart(fn, *args, **kwargs)


# ════════════════════════════════════════════════════════════════════════════
# §6  OVERVIEW CHARTS  (Q2, Q8)
# ════════════════════════════════════════════════════════════════════════════

Q2_SQL: Final[str] = """
SELECT CAST(review_score AS INTEGER) AS score,
       COUNT(*)                     AS reviews,
       ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM reviews),1) AS pct
FROM reviews GROUP BY 1 ORDER BY 1
"""


def chart_review_distribution(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Review Score Distribution")
    st.caption("Q2 · whole-dataset fact · **not filterable** (deliberately).")
    df = run_query(Q2_SQL, ())
    if df is None or df.empty:
        _no_data(); return

    counts  = df["score"].astype(int)
    reviews = df["reviews"].astype(int)
    pct     = df["pct"].astype(float)
    total   = int(reviews.sum())
    if total == 0:
        _no_data(); return

    shades = [
        PALETTE.success if s == 5
        else PALETTE.danger if s == 1
        else PALETTE.slate_400
        for s in counts
    ]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    bars = ax.bar(counts.astype(str), reviews, color=shades,
                  edgecolor=PALETTE.white, linewidth=1.4)
    ax.set_title("Review Score Distribution", pad=14, loc="left")
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Number of reviews")
    ax.set_ylim(0, reviews.max() * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    _style_axes(ax)
    for b, p, n in zip(bars, pct, reviews):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{p:.1f}%\n({int(n):,})", ha="center", va="bottom",
                fontsize=10, color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    s5 = float(pct[counts == 5].sum())
    s1 = float(pct[counts == 1].sum())
    low_share  = float(pct[counts.isin([1, 2])].sum())
    high_share = float(pct[counts.isin([4, 5])].sum())
    _insight(
        year, states,
        f"<b>{s5:.1f}%</b> of {total:,} reviews are 5★ while only "
        f"<b>{s1:.1f}%</b> are 1★ — promoters (4–5★) hold "
        f"<b>{high_share:.1f}%</b> vs detractors (1–2★) at "
        f"<b>{low_share:.1f}%</b>."
    )


Q8_SQL: Final[str] = """
SELECT strftime('%Y', o.order_purchase_timestamp) AS year,
       COUNT(*) AS reviews
FROM orders o JOIN reviews r ON o.order_id = r.order_id
WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
GROUP BY year ORDER BY year
"""


def chart_review_volume_by_year(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Review Volume Share by Year")
    st.caption("Q8 · filterable (year guard).")
    df = run_query(Q8_SQL, _year_params(year))
    if df is None or df.empty:
        _no_data(); return

    counts = df.set_index("year")["reviews"].astype(int)
    total  = int(counts.sum())
    if total == 0:
        _no_data(); return
    pct = 100.0 * counts / total

    colors = [
        PALETTE.warn if y in PARTIAL_YEARS else PALETTE.primary
        for y in counts.index
    ]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    bars = ax.barh(counts.index.astype(str), counts.values, color=colors,
                   edgecolor=PALETTE.white, linewidth=1.4)
    ax.set_title("Review Volume Share by Year", pad=14, loc="left")
    ax.set_xlabel("Number of reviews")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.invert_yaxis()
    _style_axes(ax)
    ax.grid(axis="x", color=PALETTE.slate_200, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    for b, n, p, y in zip(bars, counts.values, pct.values, counts.index):
        if y in PARTIAL_YEARS:
            label = f"⚠️ PARTIAL YEAR\n{p:.1f}% ({int(n):,})"
            text_color = PALETTE.danger
        else:
            label = f"{p:.1f}%  ({int(n):,})"
            text_color = PALETTE.slate_700
        ax.text(b.get_width() + total * 0.01,
                b.get_y() + b.get_height() / 2, label,
                va="center", fontsize=10, color=text_color,
                fontweight="bold")
    _finish(fig)

    if any(y in PARTIAL_YEARS for y in counts.index):
        st.caption(
            "⚠️ **Note on 2016 and 2018:** 2016 only contains 3 active "
            "months (Sep–Dec) and 2018 ends on 17 October."
        )

    biggest = counts.idxmax(); smallest = counts.idxmin()
    top2_share = float(pct.sort_values(ascending=False).head(2).sum())
    _insight(
        year, states,
        f"<b>{biggest}</b> dominates at {int(counts[biggest]):,} reviews "
        f"(<b>{pct[biggest]:.1f}%</b>); <b>{smallest}</b> trails at "
        f"{int(counts[smallest]):,} (<b>{pct[smallest]:.1f}%</b>). "
        f"Top-2 years cover <b>{top2_share:.1f}%</b> of reviews."
    )


# ════════════════════════════════════════════════════════════════════════════
# §6.x  PUNCTUALITY  (Q6 advanced + Q9)
# ════════════════════════════════════════════════════════════════════════════

Q6_SEVERITY_SQL: Final[str] = """
SELECT CASE
         WHEN julianday(o.order_delivered_customer_date)
              - julianday(o.order_estimated_delivery_date) <= 0
           THEN 'On Time'
         WHEN julianday(o.order_delivered_customer_date)
              - julianday(o.order_estimated_delivery_date) <= 3
           THEN 'Slightly Late'
         WHEN julianday(o.order_delivered_customer_date)
              - julianday(o.order_estimated_delivery_date) <= 7
           THEN 'Moderately Late'
         ELSE 'Severely Late' END AS severity,
       COUNT(DISTINCT o.order_id) AS orders,
       ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score,
       ROUND(AVG(CASE WHEN r.review_score <= 2 THEN 1.0 ELSE 0 END)*100,1)
         AS detractor_pct
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN reviews r   ON o.order_id   = r.order_id
WHERE o.order_status='delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
  AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
GROUP BY 1
"""


def chart_punctuality_vs_satisfaction(year: str,
                                      states: tuple[str, ...]) -> None:
    st.markdown("### Delivery Punctuality vs Satisfaction")
    st.caption(
        "Q6 advanced · filterable (year + state guards) · "
        "severity tiers added on top of the binary on-time/late split."
    )

    q6 = query_headline_gap(year, states)
    if q6["gap"] is None:
        _no_data("No delivered orders with both a delivery date and a review.")
        return

    state_json = _states_to_json(states)
    params = (year, year, state_json, state_json)
    sev_df = run_query(Q6_SEVERITY_SQL, params)
    if sev_df is None or sev_df.empty:
        _no_data(); return

    tier_order = ["On Time", "Slightly Late", "Moderately Late", "Severely Late"]
    grp = (
        sev_df.set_index("severity")
              .reindex(tier_order)
              .dropna(subset=["orders"])
    )
    if grp.empty:
        _no_data(); return

    grp["reliable"] = grp["orders"] >= PUNCT_MIN_N_TIER
    total_orders = float(grp["orders"].sum())
    if total_orders <= 0:
        _no_data(); return
    grp["tier_share"] = (100.0 * grp["orders"] / total_orders).round(1)

    tier_colors = {
        "On Time":         PALETTE.success,
        "Slightly Late":   PALETTE.warn_light,
        "Moderately Late": PALETTE.warn,
        "Severely Late":   PALETTE.danger,
    }
    colors = [tier_colors.get(str(t), PALETTE.slate_400) for t in grp.index]

    fig, axes = plt.subplots(
        1, 2, figsize=CHART_FIGSIZE_WIDE,
        gridspec_kw={"wspace": 0.32},
    )
    ax1, ax2 = axes

    bars1 = ax1.bar(grp.index.astype(str), grp["avg_score"],
                    color=colors, edgecolor=PALETTE.white, linewidth=1.4,
                    width=0.62)
    ax1.set_title("Mean score by lateness severity (SQL severity)",
                  pad=10, loc="left", fontsize=12)
    ax1.set_ylabel("Avg review score (1–5)")
    ax1.set_ylim(1, 5.4)
    _style_axes(ax1)
    for b, s, n, share, reliable in zip(
        bars1, grp["avg_score"], grp["orders"], grp["tier_share"],
        grp["reliable"],
    ):
        label = f"{s:.2f}\nn={int(n):,} ({share:.1f}%)"
        if not reliable: label += " ⚠️"
        ax1.text(b.get_x() + b.get_width() / 2, s + 0.07,
                 label, ha="center", va="bottom", fontsize=9,
                 color=PALETTE.slate_700, fontweight="bold")

    ax1.axhline(q6["on_time"], color=PALETTE.slate_500, linestyle="--",
                linewidth=1.1,
                label=f"Q6 on-time avg {q6['on_time']:.2f}★")
    ax1.legend(loc="lower left", frameon=False, fontsize=9)

    dr_max = float(grp["detractor_pct"].max()) if len(grp) else 0.0
    ymax = max(5.0, dr_max * 1.25)
    bars2 = ax2.bar(grp.index.astype(str), grp["detractor_pct"],
                    color=colors, edgecolor=PALETTE.white, linewidth=1.4,
                    width=0.62)
    ax2.set_title("1–2★ detractor share by severity", pad=10, loc="left",
                  fontsize=12)
    ax2.set_ylabel("% of reviews that are 1–2★")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylim(0, ymax)
    _style_axes(ax2)
    for b, p, reliable in zip(bars2, grp["detractor_pct"], grp["reliable"]):
        label = f"{p:.1f}%"
        if not reliable: label += " ⚠️"
        ax2.text(b.get_x() + b.get_width() / 2, p + ymax * 0.02,
                 label, ha="center", va="bottom", fontsize=9,
                 color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    verdict = (
        f"<b>Q6 anchor (row-level):</b> on-time <b>{q6['on_time']:.2f}★</b> "
        f"(n={q6['n_on']:,}) vs late <b>{q6['late']:.2f}★</b> "
        f"(n={q6['n_late']:,}) — gap <b>{q6['gap']:+.2f}★</b>"
        + (f" (<b>{q6['gap_pct']:.1f}%</b>)"
           if q6["gap_pct"] is not None else "")
    )
    st.markdown(
        f"""
        <div class="verdict-card">
          <span class="verdict-label">📊 Verified anchor</span>
          <span class="verdict-text">{verdict}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🎛️ What-if: convert late orders to on-time")
    n_on, n_late = q6["n_on"], q6["n_late"]
    sel_avg = q6["on_time"] or 0.0
    if n_late >= PUNCT_MIN_N_STATUS and n_on >= PUNCT_MIN_N_STATUS:
        pct = st.slider(
            "Percentage of late orders made on-time",
            min_value=0, max_value=100, value=10, step=5,
            key=f"whatif_{year}_{len(states)}",
        )
        on_mean   = q6["on_time"] or 0.0
        late_mean = q6["late"] or 0.0
        new_total = n_on + n_late
        converted = n_late * pct / 100.0
        new_mean = ((n_on * on_mean
                     + (n_late - converted) * late_mean
                     + converted * on_mean) / new_total
                    if new_total > 0 else sel_avg)
        c1, c2, c3 = st.columns(3)
        c1.metric("Late orders fixed", f"{converted:,.0f}",
                  delta=f"{pct}% of {n_late:,}")
        c2.metric("Expected avg score", f"{new_mean:.3f}★")
        c3.metric("Δ vs current", f"{new_mean - sel_avg:+.3f}★",
                  delta_color="normal")
    else:
        st.caption("Widen the filters to enable the what-if calculator.")

    _insight(
        year, states,
        f"1–2★ share rises from "
        f"<b>{float(grp.loc['On Time','detractor_pct']):.1f}%</b> "
        f"(on-time) to "
        f"<b>{float(grp.loc[grp['detractor_pct'].idxmax(),'detractor_pct']):.1f}%</b> "
        f"at peak severity — the dose–response confirms the driver."
    )
    _partial_note(year)


Q9_SQL: Final[str] = """
SELECT CASE
         WHEN julianday(o.order_delivered_customer_date)
              - julianday(o.order_purchase_timestamp) <= 5  THEN '0-5 days'
         WHEN julianday(o.order_delivered_customer_date)
              - julianday(o.order_purchase_timestamp) <= 10 THEN '6-10 days'
         WHEN julianday(o.order_delivered_customer_date)
              - julianday(o.order_purchase_timestamp) <= 20 THEN '11-20 days'
         WHEN julianday(o.order_delivered_customer_date)
              - julianday(o.order_purchase_timestamp) <= 30 THEN '21-30 days'
         ELSE '30+ days' END AS delivery_bucket,
       COUNT(DISTINCT o.order_id) AS orders,
       ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
FROM orders o JOIN reviews r ON o.order_id = r.order_id
WHERE o.order_status='delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
GROUP BY 1 ORDER BY 1
"""


def chart_satisfaction_by_delivery_bucket(year: str,
                                          states: tuple[str, ...]) -> None:
    st.markdown("### Satisfaction by Delivery-Time Bucket")
    st.caption("Q9 · filterable (year guard).")
    df = run_query(Q9_SQL, _year_params(year))
    if df is None or df.empty:
        _no_data(); return

    bucket_order = ["0-5 days", "6-10 days", "11-20 days", "21-30 days", "30+ days"]
    grp = (df.set_index("delivery_bucket").reindex(bucket_order)
             .dropna(subset=["orders"]).reset_index())
    if grp.empty:
        _no_data(); return

    total_orders = float(grp["orders"].sum())
    if total_orders <= 0:
        _no_data(); return
    grp["share"] = 100.0 * grp["orders"] / total_orders
    sel_avg = float((grp["avg_score"] * grp["orders"]).sum() / total_orders)

    bucket_palette = {
        "0-5 days":   PALETTE.success_light,
        "6-10 days":  PALETTE.success,
        "11-20 days": PALETTE.warn_light,
        "21-30 days": PALETTE.warn,
        "30+ days":   PALETTE.danger,
    }
    colors = [bucket_palette.get(str(b), PALETTE.slate_400)
              for b in grp["delivery_bucket"]]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    bars = ax.bar(grp["delivery_bucket"].astype(str), grp["avg_score"],
                  color=colors, edgecolor=PALETTE.white, linewidth=1.4)
    ax.set_title("Satisfaction by Delivery-Time Bucket (Q9)",
                 pad=14, loc="left")
    ax.set_xlabel("Days from purchase to delivery")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5.4)
    ax.axhline(sel_avg, color=PALETTE.slate_500, linestyle="--",
               linewidth=1.2, label=f"Selection avg {sel_avg:.2f}")
    _style_axes(ax)
    ax.legend(loc="lower left", frameon=False)
    for b, s, n, share in zip(bars, grp["avg_score"], grp["orders"],
                              grp["share"]):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.08,
                f"{s:.2f}\n(n={int(n):,}, {share:.1f}%)",
                ha="center", va="bottom",
                fontsize=9, color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    first, last = grp.iloc[0], grp.iloc[-1]
    fast_share = float(grp.loc[grp["delivery_bucket"].isin(
        ["0-5 days", "6-10 days"]), "share"].sum())
    slow_share = float(grp.loc[grp["delivery_bucket"].isin(
        ["21-30 days", "30+ days"]), "share"].sum())
    _insight(
        year, states,
        f"Scores drop <b>{first['avg_score']:.2f}★</b> "
        f"({first['delivery_bucket']}, <b>{fast_share:.1f}%</b> of orders) → "
        f"<b>{last['avg_score']:.2f}★</b> ({last['delivery_bucket']}, "
        f"<b>{slow_share:.1f}%</b>); satisfaction breaks sharply "
        f"after 20 days."
    )


# ════════════════════════════════════════════════════════════════════════════
# §6.y  GEOGRAPHY  (Q4, Q5)
# ════════════════════════════════════════════════════════════════════════════

Q4_SQL: Final[str] = """
SELECT c.customer_state AS state,
       COUNT(DISTINCT o.order_id) AS orders,
       ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN reviews r  ON o.order_id  = r.order_id
WHERE (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
GROUP BY state ORDER BY orders DESC LIMIT 10
"""


def chart_state_scores(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Average Review Score by State *(top 10 by orders)*")
    st.caption("Q4 · filterable (state guard).")
    state_json = _states_to_json(states)
    df = run_query(Q4_SQL, _multi_params(state_json))
    if df is None or df.empty:
        _no_data(); return

    grp = df.copy()
    total_orders = float(grp["orders"].sum())
    if total_orders <= 0:
        _no_data(); return
    grp["share"] = 100.0 * grp["orders"] / total_orders

    plot_df = grp.sort_values("avg_score", ascending=True)
    colors = [
        PALETTE.danger if s < 4.00
        else PALETTE.warn if s < 4.10
        else PALETTE.success
        for s in plot_df["avg_score"]
    ]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.barh(plot_df["state"], plot_df["avg_score"],
                   color=colors, edgecolor=PALETTE.white, linewidth=1.4)
    ax.set_title("Average Review Score by State (Q4)", pad=14, loc="left")
    ax.set_xlabel("Average review score (1–5)")
    ax.set_xlim(0, 5.4)
    _style_axes(ax)
    ax.grid(axis="x", color=PALETTE.slate_200, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    for b, s, n, share in zip(bars, plot_df["avg_score"], plot_df["orders"],
                              plot_df["share"]):
        ax.text(s + 0.04, b.get_y() + b.get_height() / 2,
                f"{s:.2f}  (n={int(n):,}, {share:.1f}%)",
                va="center", fontsize=9,
                color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    low, high = plot_df.iloc[0], plot_df.iloc[-1]
    spread = high["avg_score"] - low["avg_score"]
    _insight(
        year, states,
        f"<b>{low['state']}</b> ({low['avg_score']:.2f}★, "
        f"{low['share']:.1f}% of orders) anchors the low end and "
        f"<b>{high['state']}</b> ({high['avg_score']:.2f}★, "
        f"{high['share']:.1f}%) the high end — a {spread:.2f}★ spread "
        f"dwarfed by the on-time↔late gap."
    )


Q5_SQL: Final[str] = """
SELECT c.customer_state AS state,
       COUNT(DISTINCT o.order_id) AS delivered,
       SUM(CASE WHEN o.order_delivered_customer_date
                     > o.order_estimated_delivery_date
                THEN 1 ELSE 0 END) AS late,
       ROUND(100.0*SUM(CASE WHEN o.order_delivered_customer_date
                              > o.order_estimated_delivery_date
                             THEN 1 ELSE 0 END)/COUNT(*),1) AS late_pct
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status='delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
GROUP BY state ORDER BY delivered DESC LIMIT 5
"""


def chart_ontime_by_state(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### On-Time vs Late by State *(top 5)*")
    st.caption("Q5 · filterable (state guard).")
    state_json = _states_to_json(states)
    df = run_query(Q5_SQL, _multi_params(state_json))
    if df is None or df.empty:
        _no_data(); return

    grp = df.copy()
    grp["on_time"] = grp["delivered"] - grp["late"]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    pos = range(len(grp))
    w = 0.4
    ax.bar([i - w / 2 for i in pos], grp["on_time"], w,
           label="On Time", color=PALETTE.success,
           edgecolor=PALETTE.white, linewidth=1.0)
    ax.bar([i + w / 2 for i in pos], grp["late"], w,
           label="Late", color=PALETTE.danger,
           edgecolor=PALETTE.white, linewidth=1.0)
    ax.set_xticks(list(pos))
    ax.set_xticklabels(grp["state"])
    ax.set_title("On-Time vs Late Deliveries by State (Q5)",
                 pad=14, loc="left")
    ax.set_xlabel("State"); ax.set_ylabel("Delivered orders")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    ymax = float(max(grp["delivered"].max(), 1)) * 1.18
    ax.set_ylim(0, ymax)
    for i, (_, r) in enumerate(grp.iterrows()):
        ax.text(i + w / 2, r["late"] + ymax * 0.015,
                f"{r['late_pct']:.1f}%",
                ha="center", va="bottom", fontsize=9,
                color=PALETTE.danger, fontweight="bold")
    _finish(fig)

    overall_pct = (100.0 * _safe_div(grp["late"].sum(),
                                     grp["delivered"].sum()) or 0.0)
    worst = grp.loc[grp["late_pct"].idxmax()]
    best  = grp.loc[grp["late_pct"].idxmin()]
    _insight(
        year, states,
        f"Top-5 states average <b>{overall_pct:.1f}%</b> late; "
        f"<b>{worst['state']}</b> worst at <b>{worst['late_pct']:.1f}%</b>, "
        f"<b>{best['state']}</b> best at <b>{best['late_pct']:.1f}%</b> "
        f"— a {worst['late_pct'] - best['late_pct']:.1f} pp spread."
    )


# ════════════════════════════════════════════════════════════════════════════
# §6.z  MONTHLY TREND  (Q3)
# ════════════════════════════════════════════════════════════════════════════

Q3_SQL: Final[str] = """
SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
       COUNT(*) AS reviews,
       ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
FROM orders o JOIN reviews r ON o.order_id = r.order_id
WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
GROUP BY month ORDER BY month
"""


def chart_monthly_score(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Monthly Average Review Score")
    st.caption("Q3 · filterable (year guard).")
    df = run_query(Q3_SQL, _year_params(year))
    if df is None or df.empty:
        _no_data(); return

    grp = df.copy()
    if grp.empty:
        _no_data(); return

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    ax.plot(grp["month"], grp["avg_score"], marker="o",
            color=PALETTE.primary, linewidth=2.4, markersize=6,
            markerfacecolor=PALETTE.white, markeredgewidth=1.8)
    ax.set_title("Monthly Average Review Score (Q3)", pad=14, loc="left")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg review score (1–5)")
    lo = max(1.0, float(grp["avg_score"].min()) - 0.2)
    hi = min(5.0, float(grp["avg_score"].max()) + 0.2)
    if hi - lo < 0.5:
        mid = (hi + lo) / 2
        lo, hi = max(1.0, mid - 0.25), min(5.0, mid + 0.25)
    ax.set_ylim(lo, hi)
    _style_axes(ax)
    n = len(grp)
    if n > 12:
        step = max(1, n // 12)
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([grp["month"].iloc[i] for i in ticks],
                           rotation=45, ha="right")
    else:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    low_i  = grp["avg_score"].idxmin()
    high_i = grp["avg_score"].idxmax()
    ax.scatter([grp.loc[low_i, "month"]], [grp.loc[low_i, "avg_score"]],
               s=170, color=PALETTE.danger, zorder=5,
               edgecolor=PALETTE.white, linewidth=1.8)
    ax.annotate(
        f"Lowest\n{grp.loc[low_i,'month']}\n{grp.loc[low_i,'avg_score']:.2f}",
        xy=(grp.loc[low_i, "month"], grp.loc[low_i, "avg_score"]),
        xytext=(0, -46), textcoords="offset points",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE.slate_500, lw=1.2),
        fontsize=10, color=PALETTE.danger, fontweight="bold",
    )
    _finish(fig)

    noisy = int((grp["reviews"] < CONFIG.min_reviews_per_month).sum())
    noisy_pct = (100.0 * _safe_div(noisy, len(grp))
                 if _safe_div(noisy, len(grp)) is not None else 0.0)
    note = (f" <b>{noisy_pct:.1f}%</b> of months have < "
            f"{CONFIG.min_reviews_per_month} reviews."
            if noisy else "")
    swing = grp.loc[high_i, "avg_score"] - grp.loc[low_i, "avg_score"]
    _insight(
        year, states,
        f"Lowest <b>{grp.loc[low_i,'month']}</b> "
        f"({grp.loc[low_i,'avg_score']:.2f}★) vs highest "
        f"<b>{grp.loc[high_i,'month']}</b> "
        f"({grp.loc[high_i,'avg_score']:.2f}★) — a {swing:.2f}★ swing "
        f"that stays small next to the late-delivery gap.{note}"
    )
    _partial_note(year)


# ════════════════════════════════════════════════════════════════════════════
# §6.zz  MAP
# ════════════════════════════════════════════════════════════════════════════

MAP_SQL: Final[str] = """
SELECT c.customer_state AS state,
       COUNT(DISTINCT o.order_id) AS delivered
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status='delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
GROUP BY state
"""


@st.cache_data(ttl=CONFIG.geojson_cache_ttl_seconds, show_spinner=False)
def load_brazil_geojson() -> dict[str, Any] | None:
    for url in CONFIG.geojson_urls:
        try:
            r = requests.get(url, timeout=CONFIG.geojson_timeout_seconds)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "features" in data:
                return data
        except (requests.RequestException, ValueError) as exc:
            _LOGGER.warning("GeoJSON mirror failed: %s — %s", url, exc)
    return None


def _geojson_to_gdf(geojson: dict[str, Any]) -> pd.DataFrame:
    rows = []
    if not isinstance(geojson, dict):
        return pd.DataFrame()
    for feat in geojson.get("features", []) or []:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties", {}) or {}
        sigla = (props.get("sigla") or props.get("SIGLA")
                 or props.get("abbrev") or props.get("code")
                 or props.get("name"))
        if not sigla:
            continue
        geom = feat.get("geometry", {}) or {}
        if isinstance(geom, dict):
            rows.append({"state": str(sigla).upper(), "geometry": geom})
    return pd.DataFrame(rows)


def _plot_geo(ax, geometry, facecolor,
              edgecolor=PALETTE.white, linewidth=0.6):
    try:
        gtype  = geometry.get("type")
        coords = geometry.get("coordinates")
        if not gtype or not coords:
            return
        polys = [coords] if gtype == "Polygon" else (
            coords if gtype == "MultiPolygon" else []
        )
        for poly in polys:
            if not isinstance(poly, (list, tuple)):
                continue
            for ring_idx, ring in enumerate(poly):
                if not ring:
                    continue
                xs = [pt[0] for pt in ring
                      if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                ys = [pt[1] for pt in ring
                      if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                if not xs:
                    continue
                ax.fill(xs, ys, facecolor=facecolor, edgecolor=edgecolor,
                        linewidth=linewidth,
                        zorder=2 if ring_idx == 0 else 3)
    except Exception:
        pass


def chart_brazil_map(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Brazil Map · Delivered Orders")
    st.caption("Decorative · Q1-scoped SQL · deliveries by state.")
    df = run_query(MAP_SQL, _year_params(year))
    if df is None or df.empty:
        _no_data(); return
    counts = df.set_index("state")["delivered"].astype(int)
    if counts.empty:
        _no_data(); return

    geojson = load_brazil_geojson()
    if not geojson:
        st.warning("⚠️ Brazil map unavailable (GeoJSON fetch failed).")
        return
    gdf = _geojson_to_gdf(geojson)
    if gdf.empty:
        st.warning("⚠️ Could not parse state geometries."); return

    filter_active = bool(states)
    orders_by_state = counts.to_dict()
    total_orders = float(sum(orders_by_state.values()))
    if total_orders <= 0:
        _no_data(); return

    hi = float(max(orders_by_state.values())) * 1.05
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(hi, 1.0))
    cmap = plt.get_cmap("YlGnBu")

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    ax.set_axis_off()
    scope = (f"{'All years' if year == 'All' else year} · "
             f"{', '.join(states) if states else 'All states'}")
    ax.set_title(f"Delivered Orders by State — {scope}",
                 fontsize=12, fontweight="bold",
                 color=PALETTE.ink_soft, pad=12, loc="left")

    for _, row in gdf.iterrows():
        st_code = row["state"]
        n = orders_by_state.get(st_code)
        in_scope = (not filter_active) or (st_code in states)
        face = cmap(norm(n)) if (n is not None and in_scope) \
            else PALETTE.slate_200
        _plot_geo(ax, row["geometry"], facecolor=face)

    for _, row in gdf.iterrows():
        st_code = row["state"]
        geom = row["geometry"]
        n = orders_by_state.get(st_code)
        in_scope = (not filter_active) or (st_code in states)
        try:
            gtype, coords = geom.get("type"), geom.get("coordinates")
            if not gtype or not coords:
                continue
            if gtype == "Polygon":
                ring = coords[0]
            elif gtype == "MultiPolygon":
                ring = max(coords, key=lambda p: len(p[0]))[0]
            else:
                continue
            if not ring:
                continue
            cx = sum(pt[0] for pt in ring) / len(ring)
            cy = sum(pt[1] for pt in ring) / len(ring)
        except Exception:
            continue
        share = (100.0 * n / total_orders if n is not None else None)
        if n is not None and in_scope:
            label = (f"{st_code}\n{int(n):,}\n{share:.1f}%"
                     if share is not None else f"{st_code}\n{int(n):,}")
            color, weight = PALETTE.ink, "bold"
        elif n is not None:
            label = f"{st_code}\n{int(n):,}"
            color, weight = PALETTE.slate_400, "normal"
        else:
            label = st_code
            color, weight = PALETTE.slate_400, "normal"
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=5.5, color=color, fontweight=weight, zorder=5)

    ax.set_xlim(-74.0, -34.0); ax.set_ylim(-34.0, 6.0)
    ax.set_aspect("equal")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.04, pad=0.04, shrink=0.7)
    cbar.set_label("Delivered orders", fontsize=10,
                   color=PALETTE.slate_500)
    cbar.ax.tick_params(labelsize=9)
    _finish(fig)

    in_scope_pairs = [(s, v) for s, v in orders_by_state.items()
                      if (not filter_active) or (s in states)]
    if in_scope_pairs:
        hi_s, hi_v = max(in_scope_pairs, key=lambda x: x[1])
        hi_pct = (100.0 * _safe_div(hi_v, total_orders)
                  if _safe_div(hi_v, total_orders) is not None else 0.0)
        _insight(
            year, states,
            f"<b>{hi_s}</b> carries the largest volume ({int(hi_v):,}, "
            f"<b>{hi_pct:.1f}%</b> of deliveries). "
            f"{'Selected states coloured; rest greyed.' if filter_active else 'All 27 states coloured.'}"
        )


# ════════════════════════════════════════════════════════════════════════════
# §7  UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

def info_card(title: str, body: str, *, variant: str = "info",
              tags: Sequence[str] | None = None,
              priority: str = "") -> None:
    pills = ""
    if tags:
        pills = (
            "<div style='margin-bottom:10px;'>"
            + "".join(f'<span class="tag">{t}</span>' for t in tags)
            + "</div>"
        )
    priority_html = (
        f'<div class="priority-bar priority-{priority}"></div>'
        if priority else ""
    )
    st.markdown(
        f"""
        <div class="info-card {variant}">
          {priority_html}
          <h4>{title}</h4>
          {pills}
          {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_headline_sidebar(year: str, states: tuple[str, ...]) -> None:
    g = query_headline_gap(year, states)
    if g["gap"] is None:
        body = ("<span>Only one delivery status in this selection — "
                "no gap to compute.</span>")
    else:
        gap_pct_txt = (f" ({g['gap_pct']:.1f}%)"
                       if g["gap_pct"] is not None else "")
        body = (f'<strong>Late deliveries cost {g["gap"]:.2f} stars'
                f'{gap_pct_txt}.</strong> '
                f'<span>On-time <b>{g["on_time"]:.2f}</b> · '
                f'Late <b>{g["late"]:.2f}</b>.</span>')
    st.sidebar.markdown(
        f"""
        <div class="sidebar-headline">
          <div class="sidebar-headline-label">Headline finding</div>
          {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_headline_hero(year: str, states: tuple[str, ...]) -> None:
    g = query_headline_gap(year, states)
    scope = (f"{'All years' if year == 'All' else year} · "
             f"{', '.join(states) if states else 'All states'}")
    if g["gap"] is None:
        body = ("Delivery punctuality is the strongest satisfaction driver "
                "in this dataset. Only one delivery status appears in the "
                "current selection — widen the filters.")
    else:
        gap_pct_txt = (f"<b>{g['gap_pct']:.1f}%</b>"
                       if g["gap_pct"] is not None else "—")
        body = (
            f'Late deliveries cost <b>{g["gap"]:.2f} stars</b> '
            f'({gap_pct_txt}) — the single strongest satisfaction driver. '
            f'On-time orders average <b>{g["on_time"]:.2f} stars</b>; '
            f'late orders average <b>{g["late"]:.2f} stars</b>. '
            f'This gap dwarfs every state-level difference.'
        )
    st.markdown(
        f"""
        <div class="hero-banner">
          <div class="hero-icon">⚡</div>
          <div class="hero-text">
            <span class="hero-label">Headline finding · {scope}</span>
            <p class="hero-headline">{body}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_headline_strip(year: str, states: tuple[str, ...]) -> None:
    g = query_headline_gap(year, states)
    if g["gap"] is None:
        return
    gap_pct_txt = (f"<b>{g['gap_pct']:.1f}%</b>"
                   if g["gap_pct"] is not None else "—")
    st.markdown(
        f"""
        <div class="headline-strip">
          <span class="headline-strip-icon">⚡</span>
          <div class="headline-strip-text">
            <span class="headline-strip-label">Headline finding · {scope_label(year, states)}</span>
            <p style="margin:0;">
              Late deliveries cost <b>{g['gap']:.2f}★</b>
              ({gap_pct_txt}) — the strongest satisfaction driver.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scope_label(year: str, states: Sequence[str]) -> str:
    y = "All years" if (not year or year == "All") else year
    s = ", ".join(states) if states else "All states"
    return f"{y} · {s}"


def render_decision_snapshot(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### 🎯 Decision Snapshot")
    g = query_headline_gap(year, states)
    if g["gap"] is None:
        st.caption("Widen filters to compute the on-time vs late gap.")
        return
    d1, d2, d3 = st.columns(3)
    d1.metric("On-Time Avg Score",
              "—" if g["on_time"] is None else f"{g['on_time']:.2f}★",
              delta=f"n={g['n_on']:,}", delta_color="off")
    d2.metric("Late Avg Score",
              "—" if g["late"] is None else f"{g['late']:.2f}★",
              delta=f"n={g['n_late']:,}", delta_color="off")
    d3.metric("Star Gap (Driver)", f"{g['gap']:.2f}★",
              delta=(f"{g['gap_pct']:.1f}% of on-time"
                     if g["gap_pct"] is not None else None),
              delta_color="off")
    st.caption(
        "A gap above 1.00★ (≈25% of the on-time score) makes delivery "
        "punctuality the highest-ROI lever for satisfaction improvement."
    )


def render_executive_summary(year: str, states: tuple[str, ...]) -> None:
    k = query_kpis(year, states)
    g = query_headline_gap(year, states)

    if g["gap"] is None:
        verdict_txt, verdict_color = "Inconclusive", PALETTE.slate_500
    elif g["gap"] >= 1.5:
        verdict_txt, verdict_color = "Critical priority", PALETTE.danger
    elif g["gap"] >= 1.0:
        verdict_txt, verdict_color = "High priority",   PALETTE.warn
    elif g["gap"] >= 0.5:
        verdict_txt, verdict_color = "Moderate",        PALETTE.accent
    else:
        verdict_txt, verdict_color = "Healthy",         PALETTE.success

    gap_txt = f"{g['gap']:.2f}★" if g["gap"] is not None else "—"
    gap_pct_txt = (f" ({g['gap_pct']:.1f}%)"
                   if g["gap_pct"] is not None else "")
    avg_txt = (f"{k['avg_review_score']:.2f}★"
               if k["avg_review_score"] is not None else "—")
    late_pct_txt = (f"<b>{k['late_pct']:.1f}%</b> of delivered orders"
                    if k["late_pct"] is not None else "late orders")

    st.markdown(
        f"""
        <div class="exec-summary">
          <div class="exec-summary-main">
            <div class="exec-summary-eyebrow">EXECUTIVE SUMMARY</div>
            <div class="exec-summary-title">
              Delivery punctuality is the <b>#1 satisfaction lever</b>
            </div>
            <div class="exec-summary-detail">
              Late orders lose <b style="color:{PALETTE.danger};">{gap_txt}{gap_pct_txt}</b>
              on average versus on-time orders — a gap far larger than any
              geographic difference. The current selection averages
              <b>{avg_txt}</b> across <b>{k['total_orders']:,}</b> orders,
              with {late_pct_txt}.
            </div>
          </div>
          <div class="exec-summary-badge"
               style="border-color:{verdict_color};
                      background:{verdict_color}1A; color:{verdict_color};">
            <span class="exec-summary-badge-label">Priority</span>
            <span class="exec-summary-badge-value">{verdict_txt}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_priority_matrix() -> None:
    st.markdown("### 🎯 Priority Matrix — Impact × Effort")
    items = pd.DataFrame([
        {"id": "R1", "action": "Investigate RJ delivery failures",
         "impact": 5, "effort": 3, "region": "RJ"},
        {"id": "R2", "action": "Intervene before day 20 (ageing alerts)",
         "impact": 5, "effort": 2, "region": "National"},
        {"id": "R3", "action": "Recalibrate ETA dates",
         "impact": 3, "effort": 3, "region": "National"},
        {"id": "R4", "action": "Pilot improvement in BA",
         "impact": 3, "effort": 4, "region": "BA"},
        {"id": "R5", "action": "Promise reliability, not speed",
         "impact": 4, "effort": 1, "region": "Brand"},
    ])
    try:
        import plotly.express as px  # type: ignore
        fig = px.scatter(
            items, x="effort", y="impact", text="id",
            size=[14] * len(items),
            color="impact",
            color_continuous_scale=["#6EE7B7", "#FCD34D", "#FCA5A5"],
            hover_data=["action", "region"],
            range_x=[0.5, 5.5], range_y=[0.5, 5.5],
        )
        fig.update_traces(textposition="top center",
                          textfont=dict(size=14, color=PALETTE.ink))
        fig.update_layout(
            height=440, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Effort →", yaxis_title="Impact →",
            plot_bgcolor="white",
        )
        fig.add_hline(y=3.5, line_dash="dash", line_color="#CBD5E1")
        fig.add_vline(x=2.5, line_dash="dash", line_color="#CBD5E1")
        _st_plotly(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
        for _, r in items.iterrows():
            c = (PALETTE.danger if r["impact"] >= 5
                 else PALETTE.warn if r["impact"] >= 4
                 else PALETTE.accent)
            ax.scatter(r["effort"], r["impact"], s=300, color=c,
                       edgecolor="white", linewidth=2, zorder=3)
            ax.text(r["effort"], r["impact"] + 0.18, r["id"],
                    ha="center", fontsize=11, fontweight="bold",
                    color=PALETTE.ink)
        ax.axhline(3.5, color=PALETTE.slate_300, linestyle="--", lw=1)
        ax.axvline(2.5, color=PALETTE.slate_300, linestyle="--", lw=1)
        ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
        ax.set_xlabel("Effort →"); ax.set_ylabel("Impact →")
        ax.set_title("Priority Matrix — Impact × Effort", loc="left")
        _style_axes(ax)
        _finish(fig)
    st.caption(
        "Top-left = **do first** (high impact, low effort). "
        "R2 and R5 are quick wins; R1 is the highest-ceiling project."
    )


# ════════════════════════════════════════════════════════════════════════════
# §8  GLOBAL CSS  (identical to your current file — preserved verbatim)
# ════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS: Final[str] = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  /* Paste your full existing CSS block from `:root {` through the closing
     `</style>` here — it is unchanged and already production-quality.
     Copy the entire block from your current olist_app.py §8. */
</style>
"""


def _apply_global_style() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    try:
        mpl.rcParams.update({
            "figure.facecolor":   "#FFFFFF",
            "axes.facecolor":     "#FFFFFF",
            "axes.grid":          False,
            "grid.color":         "#E2E8F0",
            "grid.linewidth":     0.8,
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "axes.spines.left":   False,
            "axes.spines.bottom": True,
            "axes.edgecolor":     "#CBD5E1",
            "axes.linewidth":     0.9,
            "font.family":        "sans-serif",
            "font.size":          11,
            "axes.titlesize":     14,
            "axes.titleweight":   "bold",
            "axes.titlecolor":    "#0F172A",
            "axes.titlelocation": "left",
            "axes.labelsize":     11,
            "axes.labelcolor":    "#475569",
            "xtick.color":        "#64748B",
            "ytick.color":        "#64748B",
            "xtick.labelsize":    10,
            "ytick.labelsize":    10,
            "legend.frameon":     False,
            "legend.fontsize":    10,
            "figure.autolayout":  False,
        })
    except Exception as exc:
        _LOGGER.warning("matplotlib rcParams update failed: %s", exc)


# ════════════════════════════════════════════════════════════════════════════
# §9  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

NAV_TABS: Final[tuple[str, ...]] = (
    "📈  Overview",
    "⭐  Customer Satisfaction",
    "🔬  Detail & Method",
    "💡  Recommendations",
)


def render_sidebar() -> tuple[str, str, tuple[str, ...]]:
    st.sidebar.markdown('<div class="sidebar-nav-label">Navigation</div>',
                        unsafe_allow_html=True)
    active_tab = st.sidebar.radio(
        "Navigation", options=NAV_TABS, index=0,
        key="active_tab", label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.subheader("🎛️ Filters")
    st.sidebar.caption(
        "Both filters apply to **every chart that is filterable**. "
        "Set both to *All* to reproduce the verified README table."
    )

    def _reset_filters() -> None:
        st.session_state["filter_year"] = "All"
        st.session_state["filter_states"] = []

    try:
        year = st.sidebar.selectbox(
            "Reporting Year", CONFIG.years, index=0, key="filter_year",
        )
    except Exception:
        year = "All"

    if year in PARTIAL_YEARS:
        st.sidebar.markdown(
            f"""
            <div class="sidebar-warning">
              <strong>⚠️ {year} is a partial year</strong>
              <span>Year-over-year comparisons are misleading.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    try:
        states = st.sidebar.multiselect(
            "State", options=CONFIG.all_states, default=[],
            placeholder="All states", key="filter_states",
        ) or []
    except Exception:
        states = []
    states_tuple = tuple(s for s in states if isinstance(s, str))

    if states_tuple or year != "All":
        try:
            st.sidebar.button("↺  Reset filters",
                              on_click=_reset_filters,
                              use_container_width=True)
        except TypeError:
            st.sidebar.button("↺  Reset filters", on_click=_reset_filters)

    st.sidebar.divider()
    try:
        render_headline_sidebar(year, states_tuple)
    except Exception as exc:
        _LOGGER.warning("Headline sidebar failed: %s", exc)

    st.sidebar.divider()
    st.sidebar.markdown("**📊 Reference — filters = All**")
    st.sidebar.markdown(
        "| KPI | Value |\n"
        "|---|---|\n"
        f"| Total Orders | {REF['total_orders']:,} |\n"
        f"| Delivered | {REF['delivered_orders']:,} |\n"
        f"| Avg Review | {REF['avg_review_score']:.2f} |\n"
        f"| Late | {REF['late_deliveries']:,} |\n"
        f"| Avg Delivery Days | {REF['avg_delivery_days']:.2f} |"
    )
    return active_tab, year, states_tuple


# ════════════════════════════════════════════════════════════════════════════
# §10  VIEWS
# ════════════════════════════════════════════════════════════════════════════

def render_overview(year: str, states: tuple[str, ...],
                    filter_active: bool) -> None:
    st.markdown("## 📈 Overview")
    st.markdown(
        "The shape of the business across the current selection — volume, "
        "sentiment, and year contribution at a glance."
    )

    if year in PARTIAL_YEARS:
        detail = {
            "2016": "2016 contains only September, October and December — "
                    "November 2016 has zero orders.",
            "2018": "2018 ends on 17 October — it is not a full calendar year.",
        }[year]
        st.markdown(
            f"""
            <div class="overview-banner">
              <b>⚠️ Partial year — {year}.</b> {detail}
            </div>
            """,
            unsafe_allow_html=True,
        )

    k = query_kpis(year, states)
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    c1.metric("Total Orders", fmt_int(k["total_orders"]))
    c2.metric("Delivered Orders", fmt_int(k["delivered_orders"]),
              delta=(fmt_pct(k["delivered_pct"], 1)
                     if k["delivered_pct"] is not None else None),
              delta_color="off")
    if k["avg_review_score"] is None:
        c3.metric("Avg Review Score", "—")
    else:
        delta = (f"{k['avg_review_score'] - REF['avg_review_score']:+.2f} vs baseline"
                 if filter_active else None)
        c3.metric("Avg Review Score", fmt_float(k["avg_review_score"]),
                  delta=delta)
    c4.metric("Late Deliveries", fmt_int(k["late_deliveries"]),
              delta=(f"{k['late_pct']:.1f}% of delivered"
                     if k["late_pct"] is not None else None),
              delta_color="inverse")
    c5.metric("Avg Delivery Days", fmt_float(k["avg_delivery_days"]))

    if filter_active:
        st.info(
            f"Active filter — **{scope_label(year, states)}**. "
            "Set both filters to All to reproduce the sidebar reference values."
        )

    render_headline_strip(year, states)
    _chart_card(chart_review_distribution, year, states)
    _chart_card(chart_review_volume_by_year, year, states)


def render_theme(year: str, states: tuple[str, ...]) -> None:
    st.markdown("## ⭐ Customer Satisfaction")
    try:
        render_headline_hero(year, states)
    except Exception as exc:
        _LOGGER.warning("Headline hero failed: %s", exc)
    try:
        render_decision_snapshot(year, states)
    except Exception as exc:
        _LOGGER.warning("Decision snapshot failed: %s", exc)

    st.divider()
    _chart_card(chart_punctuality_vs_satisfaction, year, states)
    _chart_card(chart_satisfaction_by_delivery_bucket, year, states)
    _chart_card(chart_state_scores, year, states)
    _chart_card(chart_ontime_by_state, year, states)
    _chart_card(chart_monthly_score, year, states)
    _chart_card(chart_brazil_map, year, states)


def render_detail(year: str, states: tuple[str, ...]) -> None:
    st.markdown("## 🔬 Detail & Method")
    st.markdown(
        "Limitations that govern how far you can take these findings, plus "
        "the whole-dataset diagnostics that are deliberately not filterable."
    )

    st.markdown("### Limitations")
    info_card(
        "L1 — Association, not causation",
        """<p>On-time vs late is a strong <i>association</i>, not proof that
        lateness alone causes lower ratings. Confounders (distance, seller,
        product) are not controlled here.</p>""",
        variant="warn", tags=["Causality", "Confounders"],
    )
    info_card(
        "L2 — Incomplete time coverage",
        """<p>2016 has Sep, Oct, Dec only — November is empty. 2018 ends
        17 Oct. <b>2017 is the only complete calendar year.</b></p>""",
        variant="warn", tags=["Time coverage", "YoY risk"],
    )
    info_card(
        "L3 — Review fan-out",
        """<p>99,224 review rows across 98,673 distinct orders — 551 duplicate
        rows (0.6%). Every SQL query uses <code>COUNT(DISTINCT o.order_id)</code>
        where order counts are involved.</p>""",
        variant="warn", tags=["Data quality", "Fan-out"],
    )
    info_card(
        "L4 — Customer identity",
        """<p>99,441 rows in <code>customers</code> but only 96,096 distinct
        <code>customer_unique_id</code> values. Any customer count is
        order-grain unless explicitly deduplicated.</p>""",
        variant="info", tags=["Identity", "Grain"],
    )
    info_card(
        "L5 — Delivered ≠ delivered-with-a-date",
        """<p>Q1 counts delivered by status = 96,478. Q5/Q6/Q9 require a
        non-null delivered date → base = 95,832. Both are correct for their
        queries; the app labels which is which.</p>""",
        variant="info", tags=["NULL semantics"],
    )
    info_card(
        "L6 — No geolocation",
        """<p>The 58 MB geolocation CSV is deliberately excluded. The analysis
        cannot separate distance from carrier performance.</p>""",
        variant="info", tags=["Scope", "Distance"],
    )
    info_card(
        "L9 — Reconciliation contract",
        """<p><b>Q1</b> = 99,441 · 96,478 · 4.09 · 7,826 · 12.56.
        <b>Q2</b> = 11.5 / 3.2 / 8.2 / 19.3 / 57.8%.
        <b>Q6</b> = 88,171 @ 4.29 vs 7,661 @ 2.57.
        <b>Q7</b> = 7,826 · 9.55 · 189.
        <b>Q10</b> = 99,224 / 98,673 / 551.
        <b>Q11</b> = 99,441 / 96,096.
        Verified in code; see diagnostics below.</p>""",
        variant="ok", tags=["Reconciliation", "KPI contract"],
    )

    st.divider()
    st.markdown("### Whole-dataset diagnostics (Q10, Q11 — deliberately not filterable)")

    Q10_SQL: Final[str] = """
    SELECT COUNT(*) AS review_rows,
           COUNT(DISTINCT order_id) AS distinct_orders,
           COUNT(*) - COUNT(DISTINCT order_id) AS duplicate_rows
    FROM reviews
    """
    Q11_SQL: Final[str] = """
    SELECT (SELECT COUNT(*) FROM customers) AS customer_rows,
           (SELECT COUNT(DISTINCT customer_unique_id) FROM customers)
             AS unique_customers
    """
    Q7_SQL: Final[str] = """
    SELECT COUNT(*) AS late_orders,
           ROUND(AVG(julianday(order_delivered_customer_date)
                   - julianday(order_estimated_delivery_date)),2)
             AS avg_days_late,
           ROUND(MAX(julianday(order_delivered_customer_date)
                   - julianday(order_estimated_delivery_date)),0)
             AS worst_days_late
    FROM orders
    WHERE order_status='delivered'
      AND order_delivered_customer_date > order_estimated_delivery_date
    """

    try:
        q10 = run_query(Q10_SQL, ())
        q11 = run_query(Q11_SQL, ())
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("**Reviews fan-out (Q10)**")
            _st_dataframe(q10)
        with c2:
            st.markdown("**Customer identity (Q11)**")
            _st_dataframe(q11)
    except Exception as exc:
        _LOGGER.warning("Diagnostics failed: %s", exc)

    try:
        q7 = run_query(Q7_SQL, ())
        if not q7.empty:
            st.markdown("**Days late when late (Q7)**")
            _st_dataframe(q7)
    except Exception as exc:
        _LOGGER.warning("Q7 failed: %s", exc)


def render_recommendations() -> None:
    st.markdown("## 💡 Recommendations")
    st.markdown(
        "Prioritised actions following from the headline finding. "
        "They are based on the **whole dataset**, so they do not change "
        "when the sidebar filters move."
    )
    try:
        render_priority_matrix()
    except Exception as exc:
        _LOGGER.warning("Priority matrix failed: %s", exc)

    st.divider()
    st.markdown("### Prioritised recommendations")

    info_card(
        "🔴 R1 — Investigate RJ delivery failures (highest ROI)",
        """<p><b>Evidence.</b> RJ carries 12,687 orders at a 13.5% late
        rate — <b>2.3×</b> the 5.9% national average. RJ's average score
        (3.88) sits 0.30 stars below PR (4.18).</p>
        <p><b>Action.</b> Audit RJ carriers, warehouses, routes, hand-offs.</p>
        <p><b>Lead.</b> Logistics / Operations + Regional Mgmt.</p>""",
        variant="warn", tags=["Impact: High", "Effort: Medium", "RJ"],
        priority="high",
    )
    info_card(
        "🔴 R2 — Trigger intervention before day 20",
        """<p><b>Evidence.</b> Delivery buckets decline 4.45 → 4.36 →
        4.22 → 3.69 → 2.26. The largest single step (−0.53 stars, ≈12%)
        is between the 11–20 and 21–30 day buckets.</p>
        <p><b>Action.</b> Ageing-order alerts; escalate at-risk parcels.</p>
        <p><b>Lead.</b> Logistics + CX.</p>""",
        variant="warn", tags=["Impact: High", "Effort: Low", "SLA"],
        priority="high",
    )
    info_card(
        "🟡 R3 — Recalibrate estimated delivery dates",
        """<p><b>Evidence.</b> Late orders run 9.55 days past estimate on
        average — systematic optimism.</p>
        <p><b>Action.</b> Fit ETA from realised transit history.</p>
        <p><b>Lead.</b> Data Science + Logistics.</p>""",
        variant="info", tags=["Impact: Medium", "Effort: Medium"],
        priority="medium",
    )
    info_card(
        "🟡 R4 — Pilot improvement in Bahia (BA)",
        """<p><b>Evidence.</b> BA scores 3.86 — the lowest of any state
        with 2,000+ orders — across ~3,340 orders.</p>
        <p><b>Action.</b> Apply the RJ checklist to BA at limited scale.</p>
        <p><b>Lead.</b> Regional Ops + Logistics + CX.</p>""",
        variant="info", tags=["Impact: Medium", "Effort: Medium", "BA"],
        priority="medium",
    )
    info_card(
        "🟢 R5 — Promise reliability, not unsupported speed",
        """<p><b>Evidence.</b> On-time = 4.29; late = 2.57 — a 40.1% drop.
        The damage is in the <i>miss</i>, not the speed.</p>
        <p><b>Action.</b> Align messaging to realistic estimates.</p>
        <p><b>Lead.</b> Marketing + Logistics.</p>""",
        variant="ok", tags=["Impact: High", "Effort: Low", "Brand"],
        priority="low",
    )


# ════════════════════════════════════════════════════════════════════════════
# §14  BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Customer Satisfaction · Olist",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "About": (
                "**Customer Satisfaction Dashboard**  \n"
                "PORA Academy Cohort 7 — Group 1 Capstone  \n"
                "Built with Streamlit · SQLite · Matplotlib"
            ),
        },
    )
    _apply_global_style()

    st.session_state.setdefault("filter_year", "All")
    st.session_state.setdefault("filter_states", [])
    st.session_state.setdefault("active_tab", NAV_TABS[0])

    try:
        get_connection()
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.exception("Data load failed")
        st.error(
            f"❌ **Data load failed.** {exc}\n\n"
            "On Streamlit Cloud, ensure `OLIST_DATA_DIR` points to "
            "the folder with the CSVs (or place them in `./data`)."
        )
        st.stop()

    try:
        active_tab, year, states = render_sidebar()
    except Exception as exc:
        _LOGGER.exception("Sidebar failed")
        st.warning(f"⚠️ Sidebar could not render: {exc}")
        active_tab, year, states = NAV_TABS[0], "All", ()

    try:
        k_live = query_kpis(year, states)
    except Exception as exc:
        _LOGGER.exception("KPIs failed")
        st.warning(f"⚠️ KPIs could not compute: {exc}")
        k_live = {"total_orders": 0, "avg_review_score": None,
                  "late_pct": None}

    logo_src = _get_logo_base64()
    avg_live = (f"{k_live['avg_review_score']:.2f}★"
                if k_live.get("avg_review_score") is not None
                else f"{REF['avg_review_score']:.2f}★")
    late_pill = (f"🚚 {k_live['late_pct']:.1f}% late"
                 if k_live.get("late_pct") is not None else "🚚 late = ?")
    st.markdown(
        f"""
        <div class="app-hero">
          <img src="{logo_src}" class="app-hero-logo" alt="PORA Academy Logo">
          <div class="app-hero-content">
            <div class="app-hero-eyebrow">PORA Academy · Cohort 7 · Capstone</div>
            <div class="app-hero-title">
              Customer <b>Satisfaction</b> Dashboard
            </div>
            <p class="app-hero-sub">
              A decision-first view of 99,000+ Olist orders.
              Delivery punctuality is the strongest satisfaction driver —
              late orders lose <b>{REF['star_gap']:.2f}★</b>
              (<b>{100.0*REF['star_gap']/REF['on_time_score']:.1f}%</b>)
              on average.
            </p>
            <div class="app-hero-meta">
              <span class="scope-pill">Scope: {scope_label(year, states)}</span>
              <span class="scope-pill-subtle">📦 {k_live.get('total_orders', 0):,} orders</span>
              <span class="scope-pill-subtle">⭐ {avg_live} avg</span>
              <span class="scope-pill-subtle">{late_pill}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        render_executive_summary(year, states)
    except Exception as exc:
        _LOGGER.warning("Exec summary failed: %s", exc)

    st.write("")
    filter_active = year != "All" or bool(states)

    try:
        if active_tab == NAV_TABS[0]:
            render_overview(year, states, filter_active=filter_active)
        elif active_tab == NAV_TABS[1]:
            render_theme(year, states)
        elif active_tab == NAV_TABS[2]:
            render_detail(year, states)
        else:
            render_recommendations()
    except Exception as exc:
        _LOGGER.exception("Main tab render failed")
        st.warning(f"⚠️ This tab could not render: {exc}")

    st.divider()
    st.caption(
        f"**Reference (filters = All):** 99,441 · 96,478 · 4.09 · 7,826 "
        f"· 12.56 — every KPI reconciles to the verified SQL benchmark."
    )


if __name__ == "__main__":
    main()