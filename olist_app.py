# olist_app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis · Group 1 Capstone

Decision-making dashboard for customer satisfaction.
Every chart responds to the sidebar filters (Reporting Year + State).

Tabs:
    Overview               — KPI row · Review score distribution + Review
                             volume by year (side by side)
    Customer Satisfaction  — Score by state + Brazil map (side by side)
                             On-time vs late + Delivery buckets (side by side)
                             Punctuality vs satisfaction + Monthly trend
                             (side by side)
    Detail & Method        — limitations + whole-dataset diagnostics
    Recommendations        — prioritised actions (whole dataset)

Reference KPIs (filters = All):
    Total Orders 99,441 · Delivered 96,478 · Avg Review 4.09
    Late 7,826 · Avg Delivery Days 12.56

Run:
    streamlit run olist_app.py

Environment:
    OLIST_DATA_DIR   folder holding the 4 CSVs (default ./data)
    OLIST_LOG_LEVEL  logging level (default INFO)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from matplotlib.ticker import FuncFormatter


# ════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CHART_FIGSIZE: Final[tuple[float, float]] = (7.5, 4.5)


@dataclass(frozen=True)
class Palette:
    """Modern, accessible palette tuned for decision dashboards."""
    blue:         str = "#2563EB"
    blue_light:   str = "#60A5FA"
    teal:         str = "#0D9488"
    green:        str = "#10B981"
    green_light:  str = "#6EE7B7"
    orange:       str = "#F59E0B"
    orange_light: str = "#FCD34D"
    red:          str = "#EF4444"
    red_light:    str = "#FCA5A5"
    purple:       str = "#8B5CF6"
    purple_light: str = "#C4B5FD"
    yellow:       str = "#EAB308"
    grey:         str = "#64748B"
    grey_light:   str = "#E2E8F0"
    slate:        str = "#1E293B"
    text:         str = "#334155"
    muted:        str = "#94A3B8"


@dataclass(frozen=True)
class Config:
    tables: Final[dict[str, str]] = field(default_factory=lambda: {
        "orders":      "olist_orders_dataset.csv",
        "customers":   "olist_customers_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "reviews":     "olist_order_reviews_dataset.csv",
    })

    required_columns: Final[dict[str, tuple[str, ...]]] = field(
        default_factory=lambda: {
            "orders": (
                "order_id", "customer_id", "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ),
            "customers": ("customer_id", "customer_unique_id",
                          "customer_state"),
            "order_items": ("order_id", "price", "freight_value"),
            "reviews":   ("review_id", "order_id", "review_score"),
        }
    )

    all_states: Final[tuple[str, ...]] = (
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
        "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
        "RO", "RR", "RS", "SC", "SE", "SP", "TO",
    )
    years: Final[tuple[str, ...]] = ("All", "2016", "2017", "2018")

    geojson_local_path: Final[str] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "brazil-states.geojson",
    )
    geojson_urls: Final[tuple[str, ...]] = (
        "https://raw.githubusercontent.com/codeforamerica/"
        "click_that_hood/master/public/data/brazil-states.geojson",
        "https://raw.githubusercontent.com/tbrugz/"
        "vector-json/master/brazil-states.geojson",
    )
    geojson_timeout_seconds: Final[int] = 8

    data_cache_ttl_seconds:    Final[int] = 3600
    geojson_cache_ttl_seconds: Final[int] = 86400

    min_reviews_per_month: Final[int] = 30


CONFIG  = Config()
PALETTE = Palette()


# ════════════════════════════════════════════════════════════════════════════
# §2  LOGGING
# ════════════════════════════════════════════════════════════════════════════

_LOGGER = logging.getLogger("olist.dashboard")


def _configure_logging() -> None:
    if _LOGGER.handlers:
        return
    level_name = os.environ.get("OLIST_LOG_LEVEL", "INFO").upper()
    handler    = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(getattr(logging, level_name, logging.INFO))
    _LOGGER.propagate = False


_configure_logging()


# ════════════════════════════════════════════════════════════════════════════
# §3  DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

_HERE: Final[str] = os.path.dirname(os.path.abspath(__file__))


def _data_dir() -> str:
    return os.environ.get("OLIST_DATA_DIR", os.path.join(_HERE, "data"))


def _validate_dataframes(frames: dict[str, pd.DataFrame]) -> None:
    for table, required in CONFIG.required_columns.items():
        if table not in frames:
            raise ValueError(f"Missing required table: {table!r}")
        df = frames[table]
        if df.empty:
            raise ValueError(f"Table {table!r} loaded with zero rows")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Table {table!r} missing required columns: {missing}"
            )
    _LOGGER.info(
        "Schema validation passed — %d tables, %d total rows",
        len(frames), sum(len(df) for df in frames.values()),
    )


@st.cache_resource(
    ttl=CONFIG.data_cache_ttl_seconds, show_spinner="Loading Olist data…"
)
def build_connection() -> sqlite3.Connection:
    """
    Load CSVs into an in-memory SQLite database.

    Cloud-safe: never writes to disk, so Streamlit Community Cloud's
    ephemeral filesystem cannot break the app.
    """
    data_dir = _data_dir()
    frames: dict[str, pd.DataFrame] = {}
    for table, filename in CONFIG.tables.items():
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing CSV for table {table!r} at {path}. "
                f"Set OLIST_DATA_DIR to the folder containing the CSVs."
            )
        frames[table] = pd.read_csv(path)

    _validate_dataframes(frames)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for table, df in frames.items():
        df.to_sql(table, conn, if_exists="replace", index=False)

    _LOGGER.info("In-memory database built with %d tables", len(frames))
    return conn


# ── Filter contract ────────────────────────────────────────────────────────

FILTER_SQL: Final[str] = """
          (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN
                        (SELECT value FROM json_each(?)))
"""

YEAR_SQL: Final[str] = """
          (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
"""

DELIVERED_SQL: Final[str] = """
          o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
"""


def year_params(year: str) -> tuple[str, str]:
    return (year, year)


def multi_params(multi_json: str) -> tuple[str, str]:
    return (multi_json, multi_json)


def year_multi(year: str, multi_json: str) -> tuple[str, str, str, str]:
    return (year, year, multi_json, multi_json)


def safe_state_json(raw_states: Iterable[Any] | None) -> str:
    try:
        if raw_states is None:
            return "[]"
        states = [s for s in raw_states if isinstance(s, str) and s]
        if not states or "All" in states:
            return "[]"
        return json.dumps(sorted(set(states)))
    except (TypeError, ValueError) as exc:
        _LOGGER.warning("safe_state_json fallback to []: %s", exc)
        return "[]"


def selected_states(state_json: str) -> list[str]:
    try:
        parsed = json.loads(state_json) if state_json else []
    except (TypeError, json.JSONDecodeError):
        return []
    return [s for s in parsed if isinstance(s, str)]


def scope_label(year: str, state_json: str) -> str:
    years  = "All years" if year == "All" else year
    states = selected_states(state_json)
    return f"{years} · {', '.join(states) if states else 'All states'}"


def safe_query(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
    *,
    label: str = "query",
) -> pd.DataFrame:
    try:
        return pd.read_sql(sql, conn, params=params)
    except (sqlite3.Error, pd.errors.DatabaseError, ValueError) as exc:
        _LOGGER.warning("%s failed gracefully: %s", label, exc)
        st.warning(
            f"⚠️ The **{label}** query could not complete. "
            "An empty result is being shown."
        )
        return pd.DataFrame()


def safe_scalar(df: pd.DataFrame, column: str, default: Any = None) -> Any:
    if df is None or df.empty or column not in df.columns:
        return default
    value = df.iloc[0][column]
    return default if pd.isna(value) else value


def fmt_int(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def fmt_float(value: Any, places: int = 2) -> str:
    return "—" if value is None else f"{value:.{places}f}"


# ════════════════════════════════════════════════════════════════════════════
# §4  KPI COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def query_kpis(
    conn: sqlite3.Connection, year: str, state_json: str
) -> dict[str, Any]:
    params = year_multi(year, state_json)

    df = safe_query(conn, f"""
        SELECT COUNT(DISTINCT o.order_id) AS total_orders,
               COUNT(DISTINCT CASE WHEN o.order_status = 'delivered'
                                    THEN o.order_id END) AS delivered_orders,
               SUM(CASE WHEN o.order_status = 'delivered'
                         AND o.order_delivered_customer_date
                              > o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS late_deliveries,
               ROUND(AVG(CASE WHEN o.order_status = 'delivered'
                                AND o.order_delivered_customer_date IS NOT NULL
                              THEN julianday(o.order_delivered_customer_date)
                                 - julianday(o.order_purchase_timestamp)
                              END), 2) AS avg_delivery_days
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {FILTER_SQL}
    """, params=params, label="KPI row")

    score_df = safe_query(conn, f"""
        SELECT ROUND(AVG(CAST(r.review_score AS REAL)), 2) AS avg_review_score,
               ROUND(100.0 * SUM(CASE WHEN r.review_score >= 4 THEN 1 ELSE 0 END)
                     / NULLIF(COUNT(*), 0), 1) AS positive_review_pct
        FROM reviews r
        JOIN orders o    ON r.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {FILTER_SQL}
    """, params=params, label="KPI avg review score")

    return {
        "total_orders":      safe_scalar(df, "total_orders", 0),
        "delivered_orders":  safe_scalar(df, "delivered_orders", 0),
        "avg_review_score":  safe_scalar(score_df, "avg_review_score"),
        "positive_review_pct": safe_scalar(score_df, "positive_review_pct"),
        "late_deliveries":   safe_scalar(df, "late_deliveries", 0),
        "avg_delivery_days": safe_scalar(df, "avg_delivery_days"),
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  CHART FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _no_data_notice(msg: str = "No data for the current filters.") -> None:
    st.info(f"ℹ️ {msg} Try widening the Year or State filters.")


def _insight(
    year: str,
    state_json: str,
    text: str,
    *,
    level: str = "info",
) -> None:
    """
    Render an actionable insight under a chart.

    level:
        "info"    — neutral context
        "success" — positive finding
        "warning" — problem worth attention
        "error"   — urgent / highest priority
    """
    scope = scope_label(year, state_json)
    md = f"**Selection:** {scope}  \n**Insight.** {text}"
    renderer = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
    }.get(level, st.info)
    renderer(md)


# ── Review score distribution (bar) ──────────────────────────────────────

def chart_review_distribution(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Review Score Distribution")
    df = safe_query(conn, f"""
        SELECT CAST(r.review_score AS INTEGER) AS score,
               COUNT(*) AS reviews
        FROM reviews r
        JOIN orders o    ON r.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {FILTER_SQL}
        GROUP BY 1 ORDER BY 1
    """, params=year_multi(year, state_json), label="review distribution")

    if df.empty:
        _no_data_notice()
        return

    total = int(df["reviews"].sum())
    df["pct"] = 100.0 * df["reviews"] / total

    shades = [
        PALETTE.green  if s == 5
        else PALETTE.red   if s == 1
        else PALETTE.blue_light
        for s in df["score"]
    ]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.bar(df["score"].astype(str), df["reviews"], color=shades)
    ax.set_title("Review Score Distribution", pad=15)
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Number of reviews")
    ax.set_ylim(0, df["reviews"].max() * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar, pct in zip(bars, df["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10,
                color=PALETTE.text, fontweight="bold")

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    def share(score: int) -> float:
        row = df.loc[df["score"] == score, "pct"]
        return float(row.iloc[0]) if not row.empty else 0.0

    low_share = share(1) + share(2)
    level = "warning" if low_share > 15 else "info"
    _insight(
        year, state_json,
        f"{share(5):.1f}% of the {total:,} reviews are 5★ and "
        f"{share(1):.1f}% are 1★. Low scores (1–2★) make up "
        f"{low_share:.1f}% of reviews.",
        level=level,
    )


# ── Monthly average score (line) ─────────────────────────────────────────

def chart_monthly_score(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Monthly Average Review Score")
    df = safe_query(conn, f"""
        SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
               COUNT(*) AS reviews,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE {FILTER_SQL}
        GROUP BY month ORDER BY month
    """, params=year_multi(year, state_json), label="monthly score")

    if df.empty:
        _no_data_notice()
        return

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    ax.plot(df["month"], df["avg_score"], marker="o",
            color=PALETTE.teal, linewidth=2.5, markersize=5)
    ax.set_title("Monthly Average Review Score", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg review score (1–5)")

    lo = max(1.0, float(df["avg_score"].min()) - 0.2)
    hi = min(5.0, float(df["avg_score"].max()) + 0.2)
    if hi - lo < 0.5:
        mid = (hi + lo) / 2
        lo, hi = max(1.0, mid - 0.25), min(5.0, mid + 0.25)
    ax.set_ylim(lo, hi)
    plt.xticks(rotation=45, ha="right")

    low_idx    = df["avg_score"].idxmin()
    low_month  = df.loc[low_idx, "month"]
    low_score  = float(df.loc[low_idx, "avg_score"])
    high_idx   = df["avg_score"].idxmax()
    high_month = df.loc[high_idx, "month"]
    high_score = float(df.loc[high_idx, "avg_score"])

    ax.scatter([low_month], [low_score], s=160, color=PALETTE.red, zorder=5,
               edgecolor="white", linewidth=1.8)
    ax.annotate(
        f"Lowest\n{low_month}\n{low_score:.2f}",
        xy=(low_month, low_score), xytext=(0, -42),
        textcoords="offset points", ha="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE.grey, lw=1.2),
        fontsize=10, color=PALETTE.red, fontweight="bold",
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    noisy = int((df["reviews"] < CONFIG.min_reviews_per_month).sum())
    note = (
        f" {noisy} month(s) have fewer than "
        f"{CONFIG.min_reviews_per_month} reviews, so their averages are "
        "noisy."
        if noisy else ""
    )
    _insight(
        year, state_json,
        f"Lowest month {low_month} ({low_score:.2f}); highest "
        f"{high_month} ({high_score:.2f}). Months with no orders in the "
        f"selection are skipped, not zero.{note}",
    )


# ── Review volume by year (bar) ──────────────────────────────────────────

def chart_review_volume_by_year(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Review Volume by Year")
    df = safe_query(conn, f"""
        SELECT strftime('%Y', o.order_purchase_timestamp) AS year,
               COUNT(*) AS reviews
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r   ON o.order_id  = r.order_id
        WHERE {FILTER_SQL}
        GROUP BY year ORDER BY year
    """, params=year_multi(year, state_json), label="review volume by year")

    if df.empty:
        _no_data_notice()
        return

    total = int(df["reviews"].sum())
    df = df.sort_values("year").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.bar(df["year"].astype(str), df["reviews"], color=PALETTE.teal)
    ax.set_title("Review Volume by Year", pad=15)
    ax.set_xlabel("Purchase year")
    ax.set_ylabel("Number of reviews")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylim(0, df["reviews"].max() * 1.15)
    for bar, count in zip(bars, df["reviews"]):
        pct = 100.0 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count:,}  ·  {pct:.1f}%", ha="center", va="bottom",
                fontsize=9, color=PALETTE.text, fontweight="bold")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    if len(df) == 1:
        row = df.iloc[0]
        _insight(
            year, state_json,
            f"The current selection contains {int(row['reviews']):,} reviews "
            f"from {int(row['year'])}. Select All years to compare year-level "
            "volume; 2016 and 2018 are partial years.",
        )
        return

    biggest  = df.loc[df["reviews"].idxmax()]
    smallest = df.loc[df["reviews"].idxmin()]
    _insight(
        year, state_json,
        f"{int(biggest['year'])} contributes the largest share "
        f"({biggest['reviews']:,} reviews, "
        f"{100 * biggest['reviews'] / total:.1f}%), while "
        f"{int(smallest['year'])} contributes only "
        f"{smallest['reviews']:,} ({100 * smallest['reviews'] / total:.1f}%). "
        "These totals are not like-for-like: 2016 and 2018 are partial years.",
    )


# ── Average review score by state (horizontal bar) ──────────────────────

def chart_state_scores(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Average Review Score by State *(top 10)*")
    df = safe_query(conn, f"""
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE {FILTER_SQL}
        GROUP BY state ORDER BY orders DESC LIMIT 10
    """, params=year_multi(year, state_json), label="state scores")

    if df.empty:
        _no_data_notice()
        return

    plot_df = df.sort_values("avg_score", ascending=True)
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.barh(plot_df["state"], plot_df["avg_score"], color=PALETTE.orange)
    ax.set_title("Average Review Score by State (top 10 by orders)", pad=15)
    ax.set_xlabel("Average review score (1–5)")
    ax.set_ylabel("State")
    ax.set_xlim(0, 5)
    for bar, score, n in zip(bars, plot_df["avg_score"], plot_df["orders"]):
        ax.text(score + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}  (n={n:,})",
                va="center", fontsize=9,
                color=PALETTE.text, fontweight="bold")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    low  = plot_df.iloc[0]
    high = plot_df.iloc[-1]
    if len(plot_df) == 1:
        text = f"{low['state']} averages {low['avg_score']:.2f} stars."
        level = "info"
    else:
        spread = high["avg_score"] - low["avg_score"]
        text = (
            f"{low['state']} ({low['avg_score']:.2f}) anchors the low "
            f"end; {high['state']} ({high['avg_score']:.2f}) the high end "
            f"— a spread of {spread:.2f} stars."
        )
        level = "warning" if low["avg_score"] < 4.0 else "info"
    _insight(year, state_json, text, level=level)


# ── On-time vs late by state (grouped bar) ──────────────────────────────

def chart_ontime_by_state(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### On-Time vs Late Deliveries by State *(top 5)*")
    df = safe_query(conn, f"""
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS delivered,
               SUM(CASE WHEN o.order_delivered_customer_date
                            > o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS late,
               SUM(CASE WHEN o.order_delivered_customer_date
                            <= o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS on_time
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {DELIVERED_SQL} AND {FILTER_SQL}
        GROUP BY state ORDER BY delivered DESC LIMIT 5
    """, params=year_multi(year, state_json), label="on-time by state")

    if df.empty:
        _no_data_notice()
        return

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    positions = range(len(df))
    width = 0.4
    ax.bar([i - width / 2 for i in positions], df["on_time"], width,
           label="On Time", color=PALETTE.green)
    ax.bar([i + width / 2 for i in positions], df["late"], width,
           label="Late", color=PALETTE.red)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(df["state"])
    ax.set_title("On-Time vs Late Deliveries by State (top 5)", pad=15)
    ax.set_xlabel("State")
    ax.set_ylabel("Delivered orders")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(frameon=False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    df = df.assign(late_rate=100.0 * df["late"] / df["delivered"])
    worst = df.loc[df["late_rate"].idxmax()]
    best  = df.loc[df["late_rate"].idxmin()]
    total_delivered = int(df["delivered"].sum())
    total_late      = int(df["late"].sum())
    overall = 100.0 * total_late / total_delivered if total_delivered else 0.0
    if len(df) == 1:
        text = (
            f"{worst['state']}: {int(worst['late']):,} of "
            f"{int(worst['delivered']):,} delivered orders were late "
            f"({worst['late_rate']:.1f}%)."
        )
        level = "warning" if worst["late_rate"] > overall * 1.5 else "info"
    else:
        text = (
            f"Across the states shown, {overall:.1f}% of delivered orders "
            f"were late. {worst['state']} is highest "
            f"({worst['late_rate']:.1f}%), {best['state']} lowest "
            f"({best['late_rate']:.1f}%)."
        )
        level = "warning" if worst["late_rate"] > overall * 1.5 else "info"
    _insight(year, state_json, text, level=level)


# ── Punctuality vs satisfaction ─────────────────────────────────────────

def chart_punctuality_vs_satisfaction(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Delivery Punctuality vs Satisfaction")
    df = safe_query(conn, f"""
        SELECT CASE WHEN o.order_delivered_customer_date
                         <= o.order_estimated_delivery_date
                    THEN 'On Time' ELSE 'Late' END AS delivery_status,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score,
               ROUND(AVG(julianday(o.order_delivered_customer_date)
                       - julianday(o.order_purchase_timestamp)), 2)
                   AS avg_delivery_days
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE {DELIVERED_SQL} AND {FILTER_SQL}
        GROUP BY 1 ORDER BY avg_score DESC
    """, params=year_multi(year, state_json),
         label="punctuality vs satisfaction")

    if df.empty:
        _no_data_notice()
        return

    color_map = {"On Time": PALETTE.green, "Late": PALETTE.red}
    colors = [color_map.get(s, PALETTE.grey) for s in df["delivery_status"]]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.bar(df["delivery_status"], df["avg_score"], color=colors)
    ax.set_title("Delivery Punctuality vs Satisfaction", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5.4)
    for bar, score, n, days in zip(
        bars, df["avg_score"], df["orders"], df["avg_delivery_days"]
    ):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.06,
                f"{score:.2f}\n(n={n:,}; {days}d avg)",
                ha="center", va="bottom", fontsize=10,
                color=PALETTE.text, fontweight="bold")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    scores = dict(zip(df["delivery_status"], df["avg_score"]))
    if "On Time" in scores and "Late" in scores:
        gap = scores["On Time"] - scores["Late"]
        text = (
            f"Late orders average {gap:.2f} stars lower "
            f"({scores['On Time']:.2f} on time vs {scores['Late']:.2f} late). "
            "This is an association, not proof of causation; investigate "
            "delivery reliability alongside other service factors."
        )
        level = "error" if gap >= 1.5 else "warning"
    else:
        only = next(iter(scores))
        text = (
            f"Only {only.lower()} orders exist in this selection, so no "
            "on-time vs late gap can be computed."
        )
        level = "info"
    _insight(year, state_json, text, level=level)


# ── Delivery buckets ────────────────────────────────────────────────────

def chart_satisfaction_by_delivery_bucket(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Satisfaction by Delivery-Time Bucket")
    df = safe_query(conn, f"""
        SELECT CASE
                 WHEN julianday(o.order_delivered_customer_date)
                      - julianday(o.order_purchase_timestamp) <= 5
                   THEN '0-5 days'
                 WHEN julianday(o.order_delivered_customer_date)
                      - julianday(o.order_purchase_timestamp) <= 10
                   THEN '6-10 days'
                 WHEN julianday(o.order_delivered_customer_date)
                      - julianday(o.order_purchase_timestamp) <= 20
                   THEN '11-20 days'
                 WHEN julianday(o.order_delivered_customer_date)
                      - julianday(o.order_purchase_timestamp) <= 30
                   THEN '21-30 days'
                 ELSE '30+ days' END AS delivery_bucket,
               COUNT(DISTINCT o.order_id) AS orders,
               COUNT(*) AS review_rows,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE {DELIVERED_SQL} AND {FILTER_SQL}
        GROUP BY 1
        ORDER BY CASE delivery_bucket
                   WHEN '0-5 days'   THEN 1
                   WHEN '6-10 days'  THEN 2
                   WHEN '11-20 days' THEN 3
                   WHEN '21-30 days' THEN 4
                   ELSE 5 END
    """, params=year_multi(year, state_json), label="delivery buckets")

    if df.empty:
        _no_data_notice()
        return

    selection_avg = float(
        (df["avg_score"] * df["review_rows"]).sum() / df["review_rows"].sum()
    )

    palette_by_bucket = {
        "0-5 days":   PALETTE.green_light,
        "6-10 days":  PALETTE.green,
        "11-20 days": PALETTE.yellow,
        "21-30 days": PALETTE.orange,
        "30+ days":   PALETTE.red,
    }
    colors = [palette_by_bucket.get(b, PALETTE.grey) for b in df["delivery_bucket"]]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.bar(df["delivery_bucket"], df["avg_score"], color=colors)
    ax.set_title("Satisfaction by Delivery-Time Bucket", pad=15)
    ax.set_xlabel("Days from purchase to delivery")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5.4)
    ax.axhline(selection_avg, color=PALETTE.grey, linestyle="--",
               linewidth=1, label=f"Selection avg {selection_avg:.2f}")
    ax.legend(loc="lower left", frameon=False)
    for bar, score, n in zip(bars, df["avg_score"], df["orders"]):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.06,
                f"{score:.2f}\n(n={n:,})",
                ha="center", va="bottom", fontsize=9,
                color=PALETTE.text, fontweight="bold")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    if len(df) == 1:
        text = (
            f"All delivered orders in this selection fall in the "
            f"{df.iloc[0]['delivery_bucket']} bucket "
            f"({df.iloc[0]['avg_score']:.2f} stars)."
        )
        level = "info"
    else:
        first, last = df.iloc[0], df.iloc[-1]
        below = df.loc[df["avg_score"] < selection_avg, "delivery_bucket"]
        below_txt = (
            "Buckets below the selection average: "
            + ", ".join(below) + "."
            if not below.empty else
            "No bucket falls below the selection average."
        )
        text = (
            f"Scores run from {first['avg_score']:.2f} "
            f"({first['delivery_bucket']}) to {last['avg_score']:.2f} "
            f"({last['delivery_bucket']}). {below_txt}"
        )
        level = "warning" if last["avg_score"] < 3.0 else "info"
    _insight(year, state_json, text, level=level)


# ── Brazil map · Delivered Orders (complete) ────────────────────────────

@st.cache_data(ttl=CONFIG.geojson_cache_ttl_seconds, show_spinner=False)
def load_brazil_geojson() -> dict[str, Any]:
    # Try local file first (fastest, no network)
    if os.path.isfile(CONFIG.geojson_local_path):
        try:
            with open(CONFIG.geojson_local_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            _LOGGER.warning("Local GeoJSON failed: %s", exc)

    # Fall back to remote mirrors
    last_err: Exception | None = None
    for url in CONFIG.geojson_urls:
        try:
            response = requests.get(
                url, timeout=CONFIG.geojson_timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            _LOGGER.warning("GeoJSON mirror failed: %s — %s", url, exc)
            last_err = exc
    raise RuntimeError(f"All GeoJSON sources failed. Last error: {last_err}")


def _geojson_to_gdf(geojson: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        sigla = (
            props.get("sigla")
            or props.get("SIGLA")
            or props.get("abbrev")
            or props.get("code")
            or props.get("name")
        )
        if not sigla:
            continue
        rows.append({
            "state": str(sigla).upper(),
            "geometry": feat.get("geometry", {}),
        })
    return pd.DataFrame(rows)


def _plot_geojson_state(
    ax: plt.Axes,
    geometry: dict[str, Any],
    facecolor: str,
    edgecolor: str = "#FFFFFF",
    linewidth: float = 0.6,
) -> None:
    gtype  = geometry.get("type")
    coords = geometry.get("coordinates")
    if not gtype or not coords:
        return

    polygons: list[list[list[tuple[float, float]]]] = []
    if gtype == "Polygon":
        polygons = [coords]
    elif gtype == "MultiPolygon":
        polygons = coords

    for poly in polygons:
        for ring_idx, ring in enumerate(poly):
            if not ring:
                continue
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            ax.fill(
                xs, ys,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=2 if ring_idx == 0 else 3,
            )


def chart_brazil_map(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Brazil Map · Delivered Orders")

    df = safe_query(conn, f"""
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS delivered_orders
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {DELIVERED_SQL} AND {YEAR_SQL}
        GROUP BY state
    """, params=year_params(year), label="Brazil map data")

    if df.empty:
        _no_data_notice("No state-level data for this year.")
        return

    selected      = selected_states(state_json)
    filter_active = len(selected) > 0

    try:
        geojson = load_brazil_geojson()
    except RuntimeError as exc:
        _LOGGER.error("GeoJSON unavailable: %s", exc)
        st.warning(
            "⚠️ The Brazil map is temporarily unavailable. "
            "Add `data/brazil-states.geojson` to the repo, or "
            "verify internet access."
        )
        return

    gdf = _geojson_to_gdf(geojson)
    if gdf.empty:
        st.warning("⚠️ No state geometries could be parsed.")
        return

    orders_by_state = dict(zip(df["state"], df["delivered_orders"]))

    counts_in_scope = [
        orders_by_state[s] for s in orders_by_state
        if orders_by_state[s] is not None
    ]
    if not counts_in_scope:
        _no_data_notice()
        return

    lo_c = 0.0
    hi_c = float(max(counts_in_scope)) * 1.05
    if hi_c - lo_c < 1:
        hi_c = lo_c + 1

    cmap = plt.get_cmap("YlGnBu")
    norm = mpl.colors.Normalize(vmin=lo_c, vmax=hi_c)

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    ax.set_axis_off()
    ax.set_title(
        f"Delivered Orders by State — {scope_label(year, state_json)}",
        fontsize=12, fontweight="bold", color=PALETTE.text, pad=12,
    )

    for _, row in gdf.iterrows():
        state     = row["state"]
        geom      = row["geometry"]
        count     = orders_by_state.get(state)
        in_filter = (not filter_active) or (state in selected)

        if count is not None and in_filter:
            facecolor = cmap(norm(count))
        else:
            facecolor = PALETTE.grey_light

        _plot_geojson_state(ax, geom, facecolor=facecolor)

    for _, row in gdf.iterrows():
        state     = row["state"]
        geom      = row["geometry"]
        count     = orders_by_state.get(state)
        in_filter = (not filter_active) or (state in selected)

        gtype  = geom.get("type")
        coords = geom.get("coordinates")
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

        if count is not None and in_filter:
            label  = f"{state}\n{int(count):,}"
            color  = "#1F2937"
            weight = "bold"
        elif count is not None:
            label  = f"{state}\n{int(count):,}"
            color  = "#9CA3AF"
            weight = "normal"
        else:
            label  = state
            color  = "#9CA3AF"
            weight = "normal"

        ax.text(
            cx, cy, label,
            ha="center", va="center",
            fontsize=5.5, color=color, fontweight=weight, zorder=5,
        )

    ax.set_xlim(-74.0, -34.0)
    ax.set_ylim(-34.0,  6.0)
    ax.set_aspect("equal")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal",
        fraction=0.04, pad=0.04, shrink=0.7,
    )
    cbar.set_label("Delivered orders", fontsize=9, color=PALETTE.text)
    cbar.ax.tick_params(labelsize=8)

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    in_scope = [
        (s, orders_by_state[s]) for s in orders_by_state
        if orders_by_state[s] is not None
        and ((not filter_active) or (s in selected))
    ]
    if not in_scope:
        _insight(year, state_json, "No states match the current filter.")
        return
    lo_s, lo_v = min(in_scope, key=lambda x: x[1])
    hi_s, hi_v = max(in_scope, key=lambda x: x[1])
    mode_note = (
        "Selected states are coloured; the rest are greyed out."
        if filter_active else
        "All 27 states are coloured by their delivered order count."
    )
    _insight(
        year, state_json,
        f"{lo_s} ({int(lo_v):,}) has the fewest delivered orders and "
        f"{hi_s} ({int(hi_v):,}) the most among states in the current "
        f"selection. {mode_note}",
    )


# ════════════════════════════════════════════════════════════════════════════
# §6  UI HELPERS
# ════════════════════════════════════════════════════════════════════════════

def info_card(
    title: str,
    body_html: str,
    *,
    variant: str = "info",
    tags: Sequence[str] | None = None,
) -> None:
    tags_html = ""
    if tags:
        pills = "".join(f'<span class="tag">{t}</span>' for t in tags)
        tags_html = f"<div class='tag-row'>{pills}</div>"

    st.markdown(
        f"""
        <div class="info-card {variant}">
          <h4>{title}</h4>
          {tags_html}
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# §7  PRESENTATION LAYER
# ════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS: Final[str] = """
<style>
    :root {
        --ink: #19323A;
        --muted: #5F7376;
        --line: #DCE7E4;
        --surface: #FFFFFF;
        --canvas: #F5F8F7;
        --accent: #0F766E;
        --coral: #D7664C;
    }

    .stApp { background: var(--canvas); }
    body, button, input, textarea, [data-testid="stMarkdownContainer"] {
        font-family: "Aptos", "Segoe UI", sans-serif;
    }

  /* Main container: cap width and add breathing room */
  .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
        max-width: 1440px;
  }

    /* Typography and vertical rhythm */
  h1, h2, h3, h4 {
    line-height: 1.25;
  }
  h1 { margin-bottom: 0.25rem !important; }
  h2 { margin-top: 1.5rem !important; margin-bottom: 0.75rem !important; }
  h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
        color: var(--ink);
  }

    /* KPI cards emphasize the current decision signals. */
  [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
    border-radius: 8px;
        padding: 15px 16px;
        min-height: 112px;
        box-shadow: 0 2px 5px rgba(25, 50, 58, 0.05);
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
        color: var(--muted) !important;
  }
  [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
    font-weight: 700 !important;
        color: var(--ink) !important;
    line-height: 1.1;
  }
  [data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
  }

  /* Tabs — clean, modern underline */
  button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 1rem;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    border-bottom-width: 3px !important;
  }

  /* Alert boxes — theme-aware, no hard-coded colors */
  [data-testid="stAlert"] {
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.92rem;
    line-height: 1.55;
  }

  /* Scope pill */
  .scope-pill {
    display: inline-block;
    background: #E1F1ED;
    color: #145A54;
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.5rem 0 1rem 0;
  }

  /* Info cards for limitations & recommendations */
  .info-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }
    .info-card.warn    { border-left-color: var(--coral); }
  .info-card.ok      { border-left-color: #10B981; }
  .info-card.info    { border-left-color: #0D9488; }

  .info-card h4 {
    color: var(--ink);
    margin: 0 0 0.6rem 0;
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.35;
  }
    .info-card.warn h4 { color: #A5412F; }
  .info-card.ok h4   { color: #047857; }
  .info-card.info h4 { color: #0F766E; }

  .info-card p, .info-card li {
    color: #334155;
    line-height: 1.6;
    margin-bottom: 0.5rem;
    font-size: 0.92rem;
  }
  .info-card p:last-child { margin-bottom: 0; }

  .tag-row { margin-bottom: 0.6rem; }
  .info-card .tag {
    display: inline-block;
    background: #F1F5F9;
    color: #475569;
    border-radius: 6px;
    padding: 0.2rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
    letter-spacing: 0.02em;
  }

  /* Divider spacing */
  hr { margin: 1.5rem 0 !important; }

  /* Sidebar tightening */
  section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
  }

    @media (max-width: 768px) {
        .block-container {
            padding: 1.25rem 1rem 2rem !important;
        }
        [data-testid="stMetric"] {
            min-height: 100px;
            padding: 12px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }
    }
</style>
"""


def _apply_global_style() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    mpl.rcParams.update({
        "figure.facecolor":   "#FFFFFF",
        "axes.facecolor":     "#FFFFFF",
        "axes.grid":          True,
        "grid.color":         "#F1F5F9",
        "grid.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": True,
        "axes.edgecolor":     "#E2E8F0",
        "axes.linewidth":     0.9,
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.titlecolor":    "#1E293B",
        "axes.titlelocation": "left",
        "axes.labelsize":     9.5,
        "axes.labelcolor":    "#64748B",
        "xtick.color":        "#64748B",
        "ytick.color":        "#64748B",
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.frameon":     False,
        "legend.fontsize":    9,
    })


def render_sidebar() -> tuple[str, list[str]]:
    st.sidebar.title("Filters")
    st.sidebar.caption(
        "Both filters apply to **every KPI and chart**. "
        "Set both to *All* to reproduce the reference values."
    )
    st.session_state.setdefault("filter_year", "All")
    st.session_state.setdefault("filter_states", [])
    if st.sidebar.button("Reset filters", width="stretch"):
        st.session_state["filter_year"] = "All"
        st.session_state["filter_states"] = []

    year = st.sidebar.selectbox(
        "Reporting Year", CONFIG.years, index=None, key="filter_year",
    )
    states = st.sidebar.multiselect(
        "State",
        options=CONFIG.all_states,
        default=None,
        help="Pick one or more states. Leave empty to include all states.",
        key="filter_states",
    )
    if not states:
        states = ["All"]

    with st.sidebar.expander("Reference values · All filters", expanded=False):
        st.markdown(
            "| KPI | Value |\n"
            "|---|---:|\n"
            "| Total orders | 99,441 |\n"
            "| Delivered | 96,478 |\n"
            "| Avg review | 4.09 |\n"
            "| Late deliveries | 7,826 |\n"
            "| Avg delivery | 12.56 days |"
        )
    if year != "All" or "All" not in states:
        st.sidebar.markdown(
            f"**Active filter:** Year = `{year}` · "
            f"States = `{', '.join(states)}`"
        )
    return year, states


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

def render_overview_tab(
    conn: sqlite3.Connection, year: str, state_json: str,
    filter_active: bool,
) -> None:
    st.markdown("## Overview")

    k = query_kpis(conn, year, state_json)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Orders", fmt_int(k["total_orders"]))
    c2.metric("Delivered Orders", fmt_int(k["delivered_orders"]))

    avg_score = k["avg_review_score"]
    if avg_score is None:
        c3.metric("Avg Review Score", "—")
    else:
        delta = (
            f"{avg_score - 4.09:+.2f} vs reference"
            if filter_active else None
        )
        c3.metric("Avg Review Score", fmt_float(avg_score), delta=delta)

    positive_share = k["positive_review_pct"]
    c4.metric(
        "Positive Reviews (4–5★)",
        f"{positive_share:.1f}%" if positive_share is not None else "—",
    )

    late_val      = k["late_deliveries"]
    delivered_val = k["delivered_orders"]
    late_rate = 100 * late_val / delivered_val if delivered_val else None
    c5.metric(
        "Late Delivery Rate",
        f"{late_rate:.1f}%" if late_rate is not None else "—",
        delta=f"{fmt_int(late_val)} orders" if late_rate is not None else None,
        delta_color="inverse",
    )
    c6.metric("Avg Delivery Days", fmt_float(k["avg_delivery_days"]))

    if filter_active:
        st.info(
            f"Active filter — **{scope_label(year, state_json)}**. "
            "Set both filters to All to reproduce the sidebar reference "
            "values."
        )

    st.divider()

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        chart_review_distribution(conn, year, state_json)
    with col_right:
        chart_review_volume_by_year(conn, year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SATISFACTION
# ════════════════════════════════════════════════════════════════════════════

def render_theme_tab(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("## Customer Satisfaction")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        chart_state_scores(conn, year, state_json)
    with col_b:
        chart_brazil_map(conn, year, state_json)

    st.divider()

    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        chart_ontime_by_state(conn, year, state_json)
    with col_d:
        chart_satisfaction_by_delivery_bucket(conn, year, state_json)

    st.divider()

    col_e, col_f = st.columns(2, gap="large")
    with col_e:
        chart_punctuality_vs_satisfaction(conn, year, state_json)
    with col_f:
        chart_monthly_score(conn, year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DETAIL & METHOD
# ════════════════════════════════════════════════════════════════════════════

def render_detail_tab(conn: sqlite3.Connection) -> None:
    st.markdown("## Detail & Method")

    st.markdown("### Data limitations")

    info_card(
        "L1 — Association, not causation",
        """<p>The on-time vs late star gap is a strong <i>association</i>,
        not proof that lateness alone causes lower ratings. Product
        quality, seller performance, distance and communication failures
        all contribute.</p>
        <p>Before claiming a specific intervention will increase
        ratings by a precise amount, use a controlled pilot or a
        regression that controls for confounders.</p>""",
        variant="warn", tags=["Causality", "Confounders"],
    )
    info_card(
        "L2 — Incomplete time coverage",
        """<p>2016 contains only September, October and December, with
        <b>no November orders</b>. 2018 ends on <b>17 October</b>.
        <b>2017 is the only complete calendar year</b>.</p>
        <p>Year-over-year comparisons that treat these periods as
        complete years mislead — the gap is real, do not
        interpolate.</p>""",
        variant="warn", tags=["Time coverage", "YoY risk"],
    )
    info_card(
        "L3 — Review fan-out",
        """<p>The dataset has <b>99,224 review rows across 98,673
        distinct orders</b> — 551 rows of fan-out. The dashboard uses
        <code>COUNT(DISTINCT o.order_id)</code> everywhere to keep
        order counts correct.</p>""",
        variant="warn", tags=["Data quality", "Fan-out"],
    )
    info_card(
        "L4 — Customer identity",
        """<p>The <code>customers</code> table has <b>99,441 rows</b> but
        only <b>96,096 distinct <code>customer_unique_id</code></b>
        values. Any customer count in this dashboard is order-grain;
        a person-grain figure would use the unique identifier.</p>""",
        variant="info", tags=["Identity", "Grain"],
    )
    info_card(
        "L5 — No delivery-distance variable",
        """<p>The large geolocation CSV is deliberately excluded. The
        analysis <b>cannot distinguish long-distance transit from
        carrier underperformance</b>. Adding postal-code centroids and
        a great-circle distance would sharpen the RJ and BA
        diagnostics.</p>""",
        variant="info", tags=["Scope", "Distance"],
    )
    info_card(
        "L6 — Dataset and generalisability boundaries",
        """<p>Findings concern Olist in Brazil during 2016–2018. They
        may not generalise to other countries, marketplaces, logistics
        networks or current operating conditions. The choropleth
        fetches a public GeoJSON at runtime, so map rendering depends
        on external availability.</p>""",
        variant="info", tags=["External validity", "Map proxy"],
    )
    info_card(
        "L7 — Small selections are noisy",
        """<p>Because every chart is filtered, narrow selections (one
        state in one year, for example) can contain only a few hundred
        reviews. Averages from small samples swing widely — read the
        <code>n=</code> labels on the charts before drawing
        conclusions.</p>""",
        variant="info", tags=["Sample size", "Filters"],
    )

    st.divider()
    st.markdown("### Whole-dataset diagnostics")

    reviews_qa = safe_query(conn, """
        SELECT COUNT(*) AS review_rows,
               COUNT(DISTINCT order_id) AS distinct_orders,
               COUNT(*) - COUNT(DISTINCT order_id) AS duplicate_rows
        FROM reviews
    """, label="review fan-out")

    customers_qa = safe_query(conn, """
        SELECT (SELECT COUNT(*) FROM customers) AS customer_rows,
               (SELECT COUNT(DISTINCT customer_unique_id) FROM customers)
                   AS unique_customers
    """, label="customer identity")

    days_late = safe_query(conn, """
        SELECT COUNT(*) AS late_orders,
               ROUND(AVG(julianday(order_delivered_customer_date)
                       - julianday(order_estimated_delivery_date)), 2)
                   AS avg_days_late,
               ROUND(MAX(julianday(order_delivered_customer_date)
                       - julianday(order_estimated_delivery_date)), 0)
                   AS worst_days_late
        FROM orders
        WHERE order_status = 'delivered'
          AND order_delivered_customer_date > order_estimated_delivery_date
    """, label="days-late diagnostic")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Reviews fan-out**")
        if not reviews_qa.empty:
            st.dataframe(reviews_qa, hide_index=True, width="stretch")
    with c2:
        st.markdown("**Customer identity**")
        if not customers_qa.empty:
            st.dataframe(customers_qa, hide_index=True, width="stretch")

    st.markdown("**Days late when late (whole dataset)**")
    if not days_late.empty:
        st.dataframe(days_late, hide_index=True, width="stretch")
        avg_late = safe_scalar(days_late, "avg_days_late")
        worst    = safe_scalar(days_late, "worst_days_late")
        st.caption(
            f"When late, orders run **{fmt_float(avg_late)} days past "
            f"estimate** on average; the worst case is "
            f"**{fmt_int(worst)} days**."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════

def render_recommendations_tab() -> None:
    st.markdown("## Recommendations")
    st.caption(
        "**These actions do not respond to the sidebar filters.** "
        "They are derived from the whole dataset, because a strategic "
        "decision should not flip simply because someone narrowed the view."
    )

    st.divider()
    st.markdown("### Prioritised recommendations")

    info_card(
        "🔴 R1 — Investigate RJ delivery failures (highest ROI)",
        """<p><b>Evidence.</b> RJ carries 12,687 orders at a 13.5% late
        rate — 2.3× the 5.9% national average. RJ's average score
        (3.88) sits 0.29 stars below PR (4.18).</p>
        <p><b>Action.</b> Audit RJ carriers, warehouse locations,
        last-mile routes and hand-off delays. Run a controlled pilot
        before scaling.</p>
        <p><b>Lead.</b> Logistics / Operations + Regional Management.</p>
        <p><b>Measures.</b> Late-delivery rate; on-time rate; median
        and 90th-percentile transit time; average review score.</p>""",
        variant="warn", tags=["High", "Operations", "RJ"],
    )
    info_card(
        "🔴 R2 — Trigger intervention before day 20",
        """<p><b>Evidence.</b> Delivery buckets decline 4.45 → 4.36 →
        4.22 → 3.69 → 2.26. The largest single step (−0.53 stars) is
        between 11–20 days and 21–30 days.</p>
        <p><b>Action.</b> Ageing-order alerts; escalate at-risk
        parcels; revised ETAs.</p>
        <p><b>Lead.</b> Logistics / Operations + Customer Experience.</p>
        <p><b>Measures.</b> Orders over 20 days; alert-to-resolution
        time; complaint rate.</p>""",
        variant="warn", tags=["High", "SLA", "Day 20"],
    )
    info_card(
        "🟡 R3 — Recalibrate estimated delivery dates",
        """<p><b>Evidence.</b> Late orders run 9.55 days past estimate
        on average — systematic optimism, not random slippage.</p>
        <p><b>Action.</b> Fit ETA from realised transit history by
        carrier and destination state; back-test predicted vs actual.</p>
        <p><b>Lead.</b> Data Science / Analytics + Logistics.</p>
        <p><b>Measures.</b> ETA error (days); % delivered by promised
        date.</p>""",
        variant="info", tags=["Medium", "Forecasting"],
    )
    info_card(
        "🟡 R4 — Pilot improvement in Bahia (BA)",
        """<p><b>Evidence.</b> BA scores 3.86 — the lowest of any state
        with 2,000+ orders — across ~3,340 orders.</p>
        <p><b>Action.</b> Apply the RJ diagnostic checklist to BA; test
        changes at limited scale.</p>
        <p><b>Lead.</b> Regional Operations + Logistics + CX.</p>
        <p><b>Measures.</b> BA avg score; late rate; complaints.</p>""",
        variant="info", tags=["Medium", "BA", "Pilot"],
    )
    info_card(
        "🟢 R5 — Promise reliability, not unsupported speed",
        """<p><b>Evidence.</b> On-time = 4.29; late = 2.57. The damage is
        in the <i>miss</i>, not the speed.</p>
        <p><b>Action.</b> Align delivery messaging with realistic
        estimates the network can meet.</p>
        <p><b>Lead.</b> Marketing / Customer Communications + Logistics.</p>
        <p><b>Measures.</b> On-time delivery; ETA accuracy; contacts
        about delivery; review score.</p>""",
        variant="ok", tags=["Strategic", "Brand"],
    )


# ════════════════════════════════════════════════════════════════════════════
# §8  BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Customer Satisfaction Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_global_style()

    try:
        conn = build_connection()
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.exception("Data load failed")
        st.error(f"❌ **Data load failed.** {exc}")
        st.stop()

    year, states  = render_sidebar()
    state_json    = safe_state_json(states)
    filter_active = year != "All" or "All" not in states

    st.title("Customer Satisfaction Dashboard")
    st.caption(
        "PORA Academy Cohort 7 · Group 1 Capstone · Brazil e-commerce, 2016–2018"
    )
    st.markdown(
        f'<span class="scope-pill">Scope: {scope_label(year, state_json)}</span>',
        unsafe_allow_html=True,
    )

    (
        tab_overview,
        tab_theme,
        tab_detail,
        tab_recommendations,
    ) = st.tabs([
        "Overview",
        "Customer Satisfaction",
        "Detail & Method",
        "Recommendations",
    ])

    with tab_overview:
        render_overview_tab(conn, year, state_json, filter_active)
    with tab_theme:
        render_theme_tab(conn, year, state_json)
    with tab_detail:
        render_detail_tab(conn)
    with tab_recommendations:
        render_recommendations_tab()

    st.divider()
    st.caption(
        "**Reference (filters = All):** Total Orders 99,441 · Delivered "
        "96,478 · Avg Review 4.09 · Late 7,826 · Avg Delivery Days 12.56. "
        "Every KPI, chart and insight line responds to the Year and State "
        "filters and reconciles to the verified README tables when both "
        "are set to All."
    )


if __name__ == "__main__":
    main()