# olist_app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis · Group 1 Capstone

Two dashboards in one app:

  📊 DATA ANALYSIS DASHBOARD — what happened?
     Overview · Customer Satisfaction · Detail & Method

  🎯 DECISION DASHBOARD — what should we do?
     Recommendations · Priorities & Ownership · Decision Record

Chart map (each chart appears exactly once):
     Review score distribution          → Overview tab (unfiltered)
     Monthly average review score       → Customer Satisfaction tab
     Average review score by state      → Customer Satisfaction tab
     On-time vs late by state           → Customer Satisfaction tab
     Punctuality vs satisfaction        → Customer Satisfaction tab (advanced · Q6)
     Satisfaction by delivery bucket    → Customer Satisfaction tab (bonus · Q9)
     Brazil map (top-10 states)         → Customer Satisfaction tab (bonus)

Data tables loaded (4 — the minimum Group 1 requires):
    orders, customers, order_items, reviews

Reference KPIs (filters = All):
    Total Orders 99,441 · Delivered 96,478 · Avg Review 4.09
    Late 7,826 · Avg Delivery Days 12.56
    On-time 4.29 · Late 2.57 · Gap 1.73 stars

Run:
    streamlit run olist_app.py

Environment:
    OLIST_DATA_DIR   folder holding the 4 CSVs (default ./data)
    OLIST_DB_PATH    optional explicit SQLite path
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
import plotly.express as px
import requests
import streamlit as st
from matplotlib.ticker import FuncFormatter


# ════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Palette:
    purple:       str = "#8B5CF6"
    purple_dark:  str = "#6D3FB5"
    purple_light: str = "#C4B5FD"
    red:          str = "#D64545"
    red_light:    str = "#F0A0A0"
    green:        str = "#16A34A"
    blue:         str = "#0EA5E9"
    grey:         str = "#9CA3AF"


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

    reference_total_orders:      Final[int]   = 99_441
    reference_delivered_orders:  Final[int]   = 96_478
    reference_national_avg:      Final[float] = 4.09
    reference_late_deliveries:   Final[int]   = 7_826
    reference_avg_delivery_days: Final[float] = 12.56
    reference_on_time_avg:       Final[float] = 4.29
    reference_late_avg:          Final[float] = 2.57
    reference_star_gap:          Final[float] = 1.73
    reference_avg_days_late:     Final[float] = 9.55

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

    data_cache_ttl_seconds:    Final[int] = 3600
    geojson_cache_ttl_seconds: Final[int] = 86400

    chart_width_wide:   Final[float] = 11.0
    chart_width_normal: Final[float] = 10.0
    chart_width_narrow: Final[float] =  9.0
    chart_height:       Final[float] =  5.5
    chart_height_short: Final[float] =  4.5

    delivery_bucket_order: Final[tuple[str, ...]] = (
        "0-5 days", "6-10 days", "11-20 days", "21-30 days", "30+ days",
    )


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


def _resolve_db_path() -> str:
    custom = os.environ.get("OLIST_DB_PATH")
    if custom:
        return custom
    if os.path.isdir("/content"):
        return "/content/olist.db"
    return os.path.join(_HERE, "olist.db")


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
    """Build the shared file-backed SQLite DB from the 4 CSVs."""
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

    db_path = _resolve_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    for table, df in frames.items():
        df.to_sql(table, conn, if_exists="replace", index=False)
    _LOGGER.info("Database built at %s", db_path)
    return conn


# ── Filter contract helpers ────────────────────────────────────────────────

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
# §4  KPI COMPUTATION  — Q1
# ════════════════════════════════════════════════════════════════════════════

def query_kpis(
    conn: sqlite3.Connection, year: str, state_json: str
) -> dict[str, Any]:
    """Q1 — headline KPIs. Filters = All reproduces the README table."""
    df = safe_query(conn, """
        SELECT COUNT(DISTINCT o.order_id) AS total_orders,
               COUNT(DISTINCT CASE WHEN o.order_status = 'delivered'
                                    THEN o.order_id END) AS delivered_orders,
               (SELECT ROUND(AVG(CAST(review_score AS REAL)), 2)
                  FROM reviews) AS avg_review_score,
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
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN
                        (SELECT value FROM json_each(?)))
    """, params=year_multi(year, state_json), label="KPI row")

    return {
        "total_orders":      safe_scalar(df, "total_orders", 0),
        "delivered_orders":  safe_scalar(df, "delivered_orders", 0),
        "avg_review_score":  safe_scalar(df, "avg_review_score"),
        "late_deliveries":   safe_scalar(df, "late_deliveries", 0),
        "avg_delivery_days": safe_scalar(df, "avg_delivery_days"),
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  CHART FUNCTIONS  (each chart appears exactly once)
# ════════════════════════════════════════════════════════════════════════════

def _no_data_notice(msg: str = "No data for the current filters.") -> None:
    st.info(f"ℹ️ {msg} Try widening the Year or State filters.")


# ── Review score distribution (Q2 · Overview · unfiltered) ────────────────

def chart_review_distribution(conn: sqlite3.Connection) -> None:
    """Whole-dataset structural fact — intentionally unfiltered."""
    st.markdown("### Review Score Distribution *(whole dataset)*")
    df = safe_query(conn, """
        SELECT CAST(review_score AS INTEGER) AS score,
               COUNT(*) AS reviews,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM reviews), 1) AS pct
        FROM reviews
        GROUP BY 1
        ORDER BY 1
    """, label="review distribution")

    if df.empty:
        _no_data_notice("No reviews in the dataset.")
        return

    shades = [
        PALETTE.purple_dark if s == 5
        else PALETTE.red   if s == 1
        else PALETTE.purple_light
        for s in df["score"]
    ]

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_normal,
                                    CONFIG.chart_height_short))
    bars = ax.bar(df["score"].astype(str), df["reviews"], color=shades)
    ax.set_title("Review Score Distribution (whole dataset)")
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Number of reviews")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar, pct in zip(bars, df["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10,
                color=PALETTE.purple_dark, fontweight="bold")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Insight.** Bimodal — 57.8% of reviews are 5★, 11.5% are 1★. "
        "Deliberately unfiltered: a whole-dataset structural fact."
    )


# ── Monthly average score (Q3 · filterable) ───────────────────────────────

def chart_monthly_score(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Monthly Average Review Score")
    df = safe_query(conn, """
        SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
               COUNT(*) AS reviews,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN
                        (SELECT value FROM json_each(?)))
        GROUP BY month
        ORDER BY month
    """, params=year_multi(year, state_json), label="monthly score")

    if df.empty:
        _no_data_notice()
        return

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_wide,
                                    CONFIG.chart_height_short))
    ax.plot(df["month"], df["avg_score"], marker="o",
            color=PALETTE.purple_dark, linewidth=2.5, markersize=5)
    ax.set_title("Monthly Average Review Score")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(3.5, 4.6)
    plt.xticks(rotation=45, ha="right")

    if "2017-11" in df["month"].values:
        nov = df.loc[df["month"] == "2017-11", "avg_score"].iloc[0]
        ax.scatter(["2017-11"], [nov], s=160, color=PALETTE.red, zorder=5,
                   edgecolor="white", linewidth=1.8)
        ax.annotate(
            f"Nov 2017\n{nov:.2f}",
            xy=("2017-11", nov), xytext=(-55, -35),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=PALETTE.grey, lw=1.2),
            fontsize=10, color=PALETTE.red, fontweight="bold",
        )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Insight.** Scores hold in the 4.0–4.2 band except Nov 2017. "
        "Nov 2016 is empty — zero orders that month; the gap is real."
    )


# ── Average review score by state (Q4 · horizontal bar) ───────────────────

def chart_state_scores(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown(
        "### Average Review Score by State *(top 10 by orders)*"
    )
    df = safe_query(conn, """
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN
                        (SELECT value FROM json_each(?)))
        GROUP BY state
        ORDER BY orders DESC
        LIMIT 10
    """, params=year_multi(year, state_json), label="state scores")

    if df.empty:
        _no_data_notice()
        return

    plot_df = df.sort_values("avg_score", ascending=True)
    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_normal,
                                    CONFIG.chart_height))
    bars = ax.barh(plot_df["state"], plot_df["avg_score"],
                   color=PALETTE.purple)
    ax.set_title("Average Review Score by State (top 10 by orders)")
    ax.set_xlabel("Average review score (1–5)")
    ax.set_ylabel("State")
    ax.set_xlim(0, 5)
    for bar, score, n in zip(bars, plot_df["avg_score"], plot_df["orders"]):
        ax.text(score + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}  (n={n:,})",
                va="center", fontsize=10,
                color=PALETTE.purple_dark, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Insight.** BA (3.86) and RJ (3.88) anchor the low end; "
        "PR (4.18) and SP (4.17) the high end."
    )


# ── On-time vs late by state (Q5 · grouped bar) ───────────────────────────

def chart_ontime_by_state(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown(
        "### On-Time vs Late Deliveries by State *(top 5)*"
    )
    df = safe_query(conn, """
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
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN
                        (SELECT value FROM json_each(?)))
        GROUP BY state
        ORDER BY delivered DESC
        LIMIT 5
    """, params=year_multi(year, state_json), label="on-time by state")

    if df.empty:
        _no_data_notice()
        return

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_normal,
                                    CONFIG.chart_height))
    positions = range(len(df))
    width = 0.4
    ax.bar([i - width / 2 for i in positions], df["on_time"], width,
           label="On Time", color=PALETTE.purple)
    ax.bar([i + width / 2 for i in positions], df["late"], width,
           label="Late", color=PALETTE.red)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(df["state"])
    ax.set_title("On-Time vs Late Deliveries by State (top 5)")
    ax.set_xlabel("State")
    ax.set_ylabel("Delivered orders")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(frameon=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Insight.** RJ's late rate is 13.5% — 2.3× the 5.9% national "
        "average. SP has the highest volume at a healthy 5.9%."
    )


# ── Punctuality vs satisfaction (Q6 · advanced feature) ───────────────────

def chart_punctuality_vs_satisfaction(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown(
        "### Delivery Punctuality vs Satisfaction *(advanced feature · Q6)*"
    )
    df = safe_query(conn, """
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
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN
                        (SELECT value FROM json_each(?)))
        GROUP BY 1
        ORDER BY avg_score DESC
    """, params=year_multi(year, state_json),
         label="punctuality vs satisfaction")

    if df.empty:
        _no_data_notice()
        return

    color_map = {"On Time": PALETTE.purple, "Late": PALETTE.red}
    colors = [color_map.get(s, PALETTE.grey) for s in df["delivery_status"]]

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_narrow,
                                    CONFIG.chart_height))
    bars = ax.bar(df["delivery_status"], df["avg_score"], color=colors)
    ax.set_title("Delivery Punctuality vs Satisfaction")
    ax.set_xlabel("")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(2, 4.8)
    for bar, score, n, days in zip(
        bars, df["avg_score"], df["orders"], df["avg_delivery_days"]
    ):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.06,
                f"{score:.2f}\n(n={n:,}; {days}d avg)",
                ha="center", va="bottom", fontsize=11,
                color=PALETTE.purple_dark, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Insight.** Late deliveries cost **1.73 stars** — 4.29 on-time "
        "vs 2.57 late. The single strongest satisfaction driver in the "
        "dataset."
    )


# ── Delivery buckets (Q9 · bonus) ─────────────────────────────────────────

def chart_satisfaction_by_delivery_bucket(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown(
        "### Satisfaction by Delivery-Time Bucket *(bonus · Q9)*"
    )
    df = safe_query(conn, """
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
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN
                        (SELECT value FROM json_each(?)))
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

    gradient = [
        PALETTE.purple_dark, PALETTE.purple,
        "#A78BFA", "#F0A0A0", PALETTE.red,
    ]
    colors = gradient[: len(df)]

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_normal,
                                    CONFIG.chart_height))
    bars = ax.bar(df["delivery_bucket"], df["avg_score"], color=colors)
    ax.set_title("Satisfaction by Delivery-Time Bucket")
    ax.set_xlabel("Days from purchase to delivery")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5)
    ax.axhline(CONFIG.reference_national_avg, color=PALETTE.grey,
               linestyle="--", linewidth=1,
               label=f"National avg {CONFIG.reference_national_avg}")
    ax.legend(loc="lower left", frameon=False)
    for bar, score, n in zip(bars, df["avg_score"], df["orders"]):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.06,
                f"{score:.2f}\n(n={n:,})",
                ha="center", va="bottom", fontsize=10,
                color=PALETTE.purple_dark, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Insight.** Smooth decline, not a cliff: 4.45 → 4.36 → 4.22 "
        "→ 3.69 → 2.26. Every bucket past 20 days falls below the 4.09 "
        "national average — the case for a 20-day SLA."
    )


# ── Brazil map (Q4 · bonus) ───────────────────────────────────────────────

@st.cache_data(ttl=CONFIG.geojson_cache_ttl_seconds, show_spinner=False)
def load_brazil_geojson() -> dict[str, Any]:
    last_err: Exception | None = None
    for url in CONFIG.geojson_urls:
        try:
            response = requests.get(url, timeout=CONFIG.geojson_timeout_seconds)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            _LOGGER.warning("GeoJSON mirror failed: %s — %s", url, exc)
            last_err = exc
    raise RuntimeError(f"All GeoJSON mirrors failed. Last error: {last_err}")


@st.cache_data(show_spinner=False)
def state_centroids() -> dict[str, tuple[float, float]]:
    return {
        "AC": (-9.0238,   -70.8120), "AL": (-9.5713,   -36.7820),
        "AM": (-3.4168,   -65.8561), "AP": ( 1.4100,   -51.7700),
        "BA": (-12.5797,  -41.7007), "CE": (-5.4984,   -39.3206),
        "DF": (-15.7998,  -47.8645), "ES": (-19.1834,  -40.3089),
        "GO": (-15.8270,  -49.8362), "MA": (-4.9609,   -45.2744),
        "MG": (-18.5122,  -44.5550), "MS": (-20.7722,  -54.7852),
        "MT": (-12.6819,  -56.9211), "PA": (-3.7900,   -52.4800),
        "PB": (-7.2400,   -36.7820), "PE": (-8.8137,   -36.9541),
        "PI": (-7.7183,   -42.7289), "PR": (-24.8934,  -51.5300),
        "RJ": (-22.9068,  -43.1729), "RN": (-5.7945,   -36.5298),
        "RO": (-11.5057,  -63.5806), "RR": ( 2.7376,   -62.0751),
        "RS": (-30.0346,  -51.2177), "SC": (-27.2423,  -50.2189),
        "SE": (-10.5741,  -37.3857), "SP": (-23.5505,  -46.6333),
        "TO": (-10.1753,  -48.2982),
    }


def chart_brazil_map(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown(
        "### Brazil Map — Avg Review Score "
        "*(top 10 states by volume · bonus)*"
    )
    df = safe_query(conn, """
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
        GROUP BY state
        ORDER BY orders DESC
        LIMIT 10
    """, params=year_params(year), label="Brazil map")

    if df.empty:
        _no_data_notice("No state-level data for this year.")
        return

    try:
        selected = json.loads(state_json) if state_json else []
    except (TypeError, json.JSONDecodeError):
        selected = []
    filter_active = len(selected) > 0

    top10       = df["state"].tolist()
    keep_states = (
        sorted(set(top10) | set(selected)) if filter_active else top10
    )

    grid = pd.DataFrame({"state": keep_states}).merge(
        df[["state", "orders", "avg_score"]], on="state", how="left",
    )
    grid["orders"]    = grid["orders"].fillna(0).astype(int)
    grid["avg_score"] = grid["avg_score"].fillna(0)

    grid["display_score"] = (
        grid.apply(
            lambda r: r["avg_score"] if r["state"] in selected else -1,
            axis=1,
        ) if filter_active else grid["avg_score"]
    )
    grid["hover"] = grid.apply(
        lambda r: (
            "Not in current filter" if filter_active and r["state"] not in selected
            else "No orders"        if r["orders"] == 0
            else f"{r['avg_score']:.2f} avg · {r['orders']:,} orders"
        ),
        axis=1,
    )

    centroids = state_centroids()
    grid["lat"] = grid["state"].map(lambda s: centroids.get(s, (None, None))[0])
    grid["lon"] = grid["state"].map(lambda s: centroids.get(s, (None, None))[1])

    label_df = (
        grid[grid["state"].isin(selected)] if filter_active else grid
    ).dropna(subset=["lat", "lon"])
    label_df = label_df.assign(
        label=label_df.apply(
            lambda r: (f"{r['state']}\n{r['avg_score']:.2f}"
                       if r["orders"] > 0 else r["state"]),
            axis=1,
        )
    )

    try:
        geojson = load_brazil_geojson()
    except RuntimeError as exc:
        _LOGGER.error("GeoJSON unavailable: %s", exc)
        st.warning("⚠️ The Brazil map is temporarily unavailable.")
        return

    color_scale = [
        [0.00, "#3F3F5A"], [0.05, "#3F3F5A"], [0.10, "#B83232"],
        [0.45, "#F0A0A0"], [0.65, PALETTE.purple_light],
        [1.00, PALETTE.purple_dark],
    ]

    try:
        fig = px.choropleth(
            grid,
            geojson=geojson,
            locations="state",
            featureidkey="properties.sigla",
            color="display_score",
            color_continuous_scale=color_scale,
            range_color=(-1, 4.3),
            scope="south america",
            labels={"display_score": "Avg review score",
                    "state": "State", "orders": "Orders"},
            hover_name="state",
            custom_data=["hover"],
        )
        fig.update_traces(
            hovertemplate="<b>%{location}</b><br>%{customdata[0]}<extra></extra>",
            marker_line_width=1.0,
            marker_line_color="#FFFFFF",
        )
        if not label_df.empty:
            fig.add_scattergeo(
                lat=label_df["lat"], lon=label_df["lon"],
                text=label_df["label"], mode="text",
                textfont=dict(size=11, color="#1F2937",
                              family="sans-serif"),
                hoverinfo="skip", showlegend=False,
            )
        fig.update_geos(
            fitbounds="locations", visible=False,
            bgcolor="rgba(0,0,0,0)", showcountries=False,
            showcoastlines=False, showland=False, showlakes=False,
            projection_type="mercator",
        )
        fig.update_layout(
            title=dict(
                text="Average Review Score — Top 10 States (Brazil)",
                x=0.5, xanchor="center",
                font=dict(size=14, color=PALETTE.purple_dark,
                          family="sans-serif"),
            ),
            margin=dict(l=0, r=0, t=48, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(
                title="Avg score",
                tickfont=dict(color=PALETTE.purple_dark),
                title_font=dict(color=PALETTE.purple_dark),
                thickness=14, len=0.7,
                tickvals=[-1, 3.9, 4.0, 4.1, 4.2, 4.3],
                ticktext=["Not selected", "3.9", "4.0", "4.1",
                          "4.2", "4.3"],
            ),
            font=dict(family="sans-serif", color="#3F3D56"),
            height=620,
        )
        st.plotly_chart(fig, use_container_width=True)
    except (ValueError, KeyError, TypeError) as exc:
        _LOGGER.exception("Map render failed: %s", exc)
        st.warning("⚠️ The map could not be rendered for this selection.")
        return

    st.caption(
        "**Insight.** Top 10 states by order volume. BA (3.86) and RJ "
        "(3.88) anchor the low end; PR (4.18) and SP (4.17) hold the "
        "high end. RJ's low score against its wealthier Southeast "
        "neighbours signals an operational problem."
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
        tags_html = f"<div style='margin-bottom:10px;'>{pills}</div>"

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
  .stApp { background: linear-gradient(160deg, #EFE9F7 0%, #E8E1F2 100%); }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6D3FB5 0%, #8B5CF6 100%);
    border-right: none;
  }
  section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important; font-weight: 700;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label {
    color: #EDE9FE !important; font-weight: 600;
    text-transform: uppercase; font-size: 12px; letter-spacing: 0.06em;
  }
  section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2) !important;
  }

  h2, h3 { color: #4C1D95; letter-spacing: -0.01em; font-weight: 700; }
  p, li, span, label { color: #3F3D56; }
  .stCaption, [data-testid="stCaptionContainer"] { color: #6B7280; }

  [data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5DEF7;
    border-top: 4px solid #8B5CF6;
    border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 4px 14px rgba(109, 63, 181, 0.08);
  }
  [data-testid="stMetricLabel"] {
    color: #6D3FB5; font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  [data-testid="stMetricValue"] {
    color: #1F2937; font-weight: 700; font-size: 28px;
  }

  button[data-baseweb="tab"] {
    color: #6B7280; font-weight: 600; font-size: 14px;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #6D3FB5; border-bottom-color: #8B5CF6;
    border-bottom-width: 3px;
  }

  hr { border-color: #E5DEF7; }
  [data-testid="stAlert"] {
    background-color: #F5F0FF;
    border-left: 4px solid #8B5CF6;
    color: #4C1D95;
  }
  code {
    background-color: #F5F0FF; color: #6D3FB5;
    padding: 2px 6px; border-radius: 4px;
  }

  .info-card {
    background: #FFFFFF; border: 1px solid #E5DEF7;
    border-left: 5px solid #8B5CF6; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 16px;
    box-shadow: 0 3px 12px rgba(109, 63, 181, 0.08);
  }
  .info-card.warn { border-left-color: #D64545; }
  .info-card.ok   { border-left-color: #16A34A; }
  .info-card.info { border-left-color: #0EA5E9; }
  .info-card h4 {
    color: #4C1D95; margin-top: 0; margin-bottom: 8px; font-size: 16px;
  }
  .info-card p, .info-card li { color: #3F3D56; line-height: 1.7; }
  .info-card .tag {
    display: inline-block; background: #F5F0FF; color: #6D3FB5;
    border-radius: 6px; padding: 2px 10px; font-size: 12px;
    font-weight: 700; margin-right: 6px; margin-bottom: 6px;
  }

  .report-body {
    background: #FFFFFF; border: 1px solid #E5DEF7;
    border-radius: 12px; padding: 26px 34px; margin-bottom: 20px;
    box-shadow: 0 3px 14px rgba(109, 63, 181, 0.08);
    line-height: 1.75; color: #3F3D56;
  }
  .report-body table {
    width: 100%; border-collapse: collapse; margin: 12px 0;
    font-size: 14px;
  }
  .report-body table th {
    background: #F5F0FF; color: #4C1D95; padding: 10px 12px;
    text-align: left; border-bottom: 2px solid #EDE7FA;
  }
  .report-body table td {
    padding: 10px 12px; border-bottom: 1px solid #F0EBFA;
    vertical-align: top;
  }

  .mode-badge {
    display: inline-block; padding: 6px 14px;
    border-radius: 20px; font-size: 12px; font-weight: 800;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 12px;
  }
  .mode-analysis {
    background: #EDE9FE; color: #4C1D95;
    border: 1px solid #C4B5FD;
  }
  .mode-decision {
    background: #FEF3C7; color: #92400E;
    border: 1px solid #FDE68A;
  }

  .headline-story {
    background: linear-gradient(120deg, #FBF8FF 0%, #F2ECFB 60%, #EFE9F7 100%);
    border: 1px solid #E5DEF7; border-left: 5px solid #8B5CF6;
    border-radius: 12px; padding: 22px 26px; margin: 8px 0 20px 0;
    box-shadow: 0 4px 16px rgba(109, 63, 181, 0.10);
    line-height: 1.75;
  }
  .headline-story p { margin: 0 0 10px 0; font-size: 16px; color: #3F3D56; }
  .headline-story p:last-child { margin-bottom: 0; }
  .headline-story .big {
    font-size: 22px; font-weight: 700; color: #3F3D56;
  }
  .headline-story .star {
    color: #D64545; font-size: 26px; font-weight: 800;
  }
  .headline-story .hl { color: #6D3FB5; font-weight: 700; }
  .headline-story .takeaway {
    color: #4C1D95; font-style: italic; font-weight: 600;
  }
</style>
"""


def _apply_global_style() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    mpl.rcParams.update({
        "figure.facecolor":   "#FFFFFF",
        "axes.facecolor":     "#FFFFFF",
        "axes.grid":          True,
        "grid.color":         "#EDE7FA",
        "grid.linewidth":     0.9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": True,
        "axes.edgecolor":     "#D8CDF0",
        "axes.linewidth":     0.9,
        "font.family":        "sans-serif",
        "font.size":          11,
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "axes.titlecolor":    "#4C1D95",
        "axes.titlelocation": "center",
        "axes.labelsize":     11,
        "axes.labelcolor":    "#6B7280",
        "xtick.color":        "#6B7280",
        "ytick.color":        "#6B7280",
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.frameon":     False,
        "legend.fontsize":    10,
    })


def render_sidebar() -> tuple[str, list[str]]:
    st.sidebar.title("📊 Filters")
    st.sidebar.markdown(
        "Both filters apply to every filterable KPI card and every "
        "chart on the **Data Analysis Dashboard**. Set to *All* to "
        "reproduce the unfiltered reference values."
    )

    year = st.sidebar.selectbox(
        "Reporting Year", CONFIG.years, index=0, key="filter_year",
    )
    states = st.sidebar.multiselect(
        "State",
        options=["All", *CONFIG.all_states],
        default=["All"],
        help='Pick one or more states. "All" applies no state filter.',
        key="filter_states",
    )
    if not states:
        states = ["All"]
    if "All" in states and len(states) > 1:
        states = ["All"]

    st.sidebar.divider()
    st.sidebar.markdown("**Reference — filters = All**")
    st.sidebar.markdown(
        "| KPI | Value |\n"
        "|---|---|\n"
        "| Total Orders | 99,441 |\n"
        "| Delivered | 96,478 |\n"
        "| Avg Review | 4.09 |\n"
        "| Late | 7,826 |\n"
        "| Avg Delivery Days | 12.56 |"
    )
    if year != "All" or "All" not in states:
        st.sidebar.markdown(
            f"**Active filter:** Year = `{year}` · "
            f"States = `{', '.join(states)}`"
        )
    return year, states


# ════════════════════════════════════════════════════════════════════════════
# 📊 DATA ANALYSIS DASHBOARD  — Tabs 1–3
# ════════════════════════════════════════════════════════════════════════════

def render_overview_tab(
    conn: sqlite3.Connection, year: str, state_json: str,
    filter_active: bool,
) -> None:
    """Overview — KPI row + review-score distribution (unfiltered)."""
    st.markdown(
        '<span class="mode-badge mode-analysis">📊 Data Analysis</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Overview")
    st.markdown(
        "Five headline KPIs plus the review-score distribution — the "
        "structural fact every other chart builds on."
    )

    k = query_kpis(conn, year, state_json)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders", fmt_int(k["total_orders"]))
    c2.metric("Delivered Orders", fmt_int(k["delivered_orders"]))

    avg_score = k["avg_review_score"]
    if avg_score is None:
        c3.metric("Avg Review Score", "—")
    else:
        delta = (
            f"{avg_score - CONFIG.reference_national_avg:+.2f} vs reference"
            if filter_active else None
        )
        c3.metric("Avg Review Score", fmt_float(avg_score), delta=delta)

    late_val      = k["late_deliveries"]
    delivered_val = k["delivered_orders"]
    late_delta = (
        f"{100 * late_val / delivered_val:.1f}% of delivered"
        if delivered_val and delivered_val > 0 else None
    )
    c4.metric("Late Deliveries", fmt_int(late_val),
              delta=late_delta, delta_color="inverse")
    c5.metric("Avg Delivery Days", fmt_float(k["avg_delivery_days"]))

    if filter_active:
        st.info(
            f"Active filter — **Year:** {year}. "
            "Toggle to All to reproduce the sidebar reference values."
        )

    st.divider()
    chart_review_distribution(conn)


def render_theme_tab(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    """Customer Satisfaction — all filterable charts, each exactly once."""
    st.markdown(
        '<span class="mode-badge mode-analysis">📊 Data Analysis</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Customer Satisfaction")
    st.markdown(
        "Charts respond to the **Year** and **State** filters. Each "
        "chart appears exactly once across the whole dashboard."
    )

    st.divider()
    st.markdown(
        """
        <div class="headline-story">
          <p class="big">
            Late deliveries cost you <span class="star">1.73 stars</span>.
          </p>
          <p>
            On-time orders average <span class="hl">4.294</span>;
            late ones average <span class="hl">2.566</span>.
          </p>
          <p class="takeaway">
            That is the single strongest satisfaction driver in this
            dataset, and it is the insight your presentation should be
            built around.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Three supporting facts")
    fact_cols = st.columns(3)
    fact_cols[0].metric(
        "Late deliveries",
        "7,826",
        delta="8.1% of delivered orders",
        delta_color="inverse",
    )
    fact_cols[1].metric(
        "Avg days past estimate (when late)",
        "9.55 d",
        delta="worst case 189 d",
        delta_color="inverse",
    )
    fact_cols[2].metric(
        "Reviews fan-out",
        "99,224 rows",
        delta="98,673 distinct orders",
        delta_color="off",
    )
    st.caption(
        "Use `COUNT(DISTINCT order_id)` across any join — plain "
        "`COUNT(*)` overstates order volume by 551 rows."
    )

    st.info(
        "**Split for the headline, buckets for the nuance.** The binary "
        "on-time/late split is the headline (4.29 vs 2.57). The "
        "delivery-time buckets below show the score decline more "
        "smoothly (4.45 → 4.36 → 4.22 → 3.69 → 2.26). Use both."
    )

    st.divider()
    chart_monthly_score(conn, year, state_json)
    st.divider()
    chart_state_scores(conn, year, state_json)
    st.divider()
    chart_ontime_by_state(conn, year, state_json)
    st.divider()
    chart_punctuality_vs_satisfaction(conn, year, state_json)
    st.divider()
    chart_satisfaction_by_delivery_bucket(conn, year, state_json)
    st.divider()
    chart_brazil_map(conn, year, state_json)


def render_detail_tab(conn: sqlite3.Connection) -> None:
    """Detail & Method — limitations + diagnostics (no charts)."""
    st.markdown(
        '<span class="mode-badge mode-analysis">📊 Data Analysis</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Detail & Method")
    st.markdown(
        "The data-limitation note required by the brief, plus the "
        "whole-dataset diagnostics behind the reference numbers."
    )

    st.markdown("### Data limitations")

    info_card(
        "L1 — Association, not causation",
        """<p>The <b>1.73-star</b> difference between on-time and late
        deliveries is a strong <i>association</i>, not proof that
        lateness alone causes lower ratings. Product quality, seller
        performance, distance and communication failures all
        contribute.</p>
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

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Reviews fan-out**")
        if not reviews_qa.empty:
            st.dataframe(reviews_qa, hide_index=True,
                         use_container_width=True)
    with c2:
        st.markdown("**Customer identity**")
        if not customers_qa.empty:
            st.dataframe(customers_qa, hide_index=True,
                         use_container_width=True)

    st.markdown("**Days late when late**")
    if not days_late.empty:
        st.dataframe(days_late, hide_index=True, use_container_width=True)
        st.caption(
            "When late, orders run **9.55 days past estimate** on "
            "average; the worst case is **189 days**."
        )


# ════════════════════════════════════════════════════════════════════════════
# 🎯 DECISION DASHBOARD  — Tabs 4–6
# ════════════════════════════════════════════════════════════════════════════

def render_recommendations_tab() -> None:
    """Recommendations — headline + supporting facts + prioritised actions."""
    st.markdown(
        '<span class="mode-badge mode-decision">🎯 Decision</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Recommendations")
    st.markdown(
        "The headline finding, the three supporting facts, and the "
        "prioritised recommendations that follow from them."
    )

    st.markdown("### Headline")
    st.markdown(
        """
        <div class="headline-story">
          <p class="big">
            Late deliveries cost you <span class="star">1.73 stars</span>.
          </p>
          <p>
            On-time orders average <span class="hl">4.294</span>;
            late ones average <span class="hl">2.566</span>.
          </p>
          <p class="takeaway">
            That is the single strongest satisfaction driver in this
            dataset, and it is the insight your presentation should be
            built around.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Three supporting facts")
    st.markdown(
        """
        <div class="info-card">
          <ol style="margin-bottom:0; color:#3F3D56; line-height:1.9;">
            <li><b>7,826 delivered orders (8.1%)</b> arrived after their
                estimated delivery date.</li>
            <li>When late, orders average <b>9.55 days past the
                estimate</b>.</li>
            <li>Reviews are a fan-out table — <b>99,224 rows for
                98,673 orders</b>. Use <code>COUNT(DISTINCT order_id)</code>
                or your order counts will be wrong.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "**Q9 (delivery-time buckets) shows the score decline more "
        "smoothly than the binary on-time/late split.** Use both — the "
        "split for the headline, the buckets for the nuance."
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


def render_priorities_tab() -> None:
    """Priorities & Ownership — department matrix + 90-day sequence."""
    st.markdown(
        '<span class="mode-badge mode-decision">🎯 Decision</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Priorities & Ownership")
    st.markdown(
        "Who owns which issue, which recommendation resolves it, and "
        "which chart they can open to watch the metric move."
    )

    st.markdown("### Department ownership matrix")
    st.markdown(
        """
        <div class="report-body">
        <table>
          <thead>
            <tr>
              <th>Department</th>
              <th>Owns</th>
              <th>Rec.</th>
              <th>Chart to monitor</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>Logistics / Operations</b></td>
              <td>RJ late rate (13.5%); nationwide 20-day tail</td>
              <td>R1, R2</td>
              <td>On-Time vs Late by State · Delivery Buckets</td>
            </tr>
            <tr>
              <td><b>Logistics + CX</b></td>
              <td>Proactive SLA on orders nearing day 20</td>
              <td>R2</td>
              <td>Punctuality vs Satisfaction · Delivery Buckets</td>
            </tr>
            <tr>
              <td><b>Data Science + Logistics</b></td>
              <td>EDD model recalibration</td>
              <td>R3</td>
              <td>Monthly Avg Review Score · Days-Late Diagnostic</td>
            </tr>
            <tr>
              <td><b>Regional Operations</b></td>
              <td>BA (3.86) satisfaction</td>
              <td>R4</td>
              <td>Avg Score by State · Brazil Map</td>
            </tr>
            <tr>
              <td><b>CX Analytics</b></td>
              <td>1★/2★ review-text themes</td>
              <td>(future)</td>
              <td>Review Score Distribution</td>
            </tr>
            <tr>
              <td><b>Data Engineering</b></td>
              <td>Identity resolution (customer_unique_id)</td>
              <td>(foundation)</td>
              <td>Backend; feeds every tab</td>
            </tr>
            <tr>
              <td><b>Marketing + Logistics</b></td>
              <td>Delivery-certainty positioning</td>
              <td>R5</td>
              <td>Punctuality vs Satisfaction · Review Distribution</td>
            </tr>
            <tr>
              <td><b>Executive / Steering</b></td>
              <td>Cross-departmental coordination</td>
              <td>All</td>
              <td>Overview KPI row</td>
            </tr>
          </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Governance & accountability")
    st.markdown(
        """
        - **Executive sponsor:** Head of Operations — resolves
          cross-functional blockers, reviews monthly results.
        - **Operational owner:** Logistics / Operations — accountable
          for the delivery improvement plan and regional pilots.
        - **Customer outcome owner:** Customer Experience — proactive
          updates, complaint categorisation, service recovery.
        - **Analytical support:** Data Science / Analytics — reliable
          metrics, ETA evaluation, review-text analysis.
        - **Data controls:** Data Engineering / Data Governance —
          identity keys, data definitions, join-quality checks.
        - **Commercial alignment:** Marketing — customer-facing promises
          match delivery capability.
        """
    )

    st.markdown("### 90-day delivery sequence")
    phase_cols = st.columns(3)
    with phase_cols[0]:
        st.markdown("**Days 1–30 · Diagnose**")
        st.markdown(
            "- Validate dashboard metrics against the reference.\n"
            "- Build an RJ root-cause view (carrier × route × city).\n"
            "- Define baseline KPIs and alert thresholds."
        )
    with phase_cols[1]:
        st.markdown("**Days 31–60 · Pilot**")
        st.markdown(
            "- Ageing-order alerts in RJ.\n"
            "- Revised ETAs on orders past day 20.\n"
            "- Investigate BA and low-score review comments."
        )
    with phase_cols[2]:
        st.markdown("**Days 61–90 · Scale or stop**")
        st.markdown(
            "- Compare with baseline; quantify the gap closed.\n"
            "- Refine ETA model; decide what scales.\n"
            "- Report back to steering."
        )


def render_decision_record_tab(
    year: str, states: list[str], kpis: dict[str, Any]
) -> None:
    """Decision Record — the formal decision this dashboard supports."""
    st.markdown(
        '<span class="mode-badge mode-decision">🎯 Decision</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Decision Record")
    st.markdown(
        "The formal decision this dashboard supports, the evidence chain "
        "behind it, the assumptions the reviewer must acknowledge, and "
        "the review cadence."
    )

    st.markdown("### Context")
    st.markdown(
        f"""
        **Active filter:** Year = `{year}` ·
        States = `{', '.join(states)}`

        **Reference (filters = All):** Total Orders
        **{fmt_int(kpis['total_orders'])}** · Delivered
        **{fmt_int(kpis['delivered_orders'])}** · Avg Review
        **{fmt_float(kpis['avg_review_score'])}** · Late
        **{fmt_int(kpis['late_deliveries'])}** · Avg Delivery Days
        **{fmt_float(kpis['avg_delivery_days'])}**
        """
    )

    st.markdown("### Decision requested")
    st.markdown(
        """
        Approve a **90-day evidence-led delivery reliability initiative**,
        starting with an RJ diagnostic and a day-20 escalation process.
        """
    )

    st.markdown("### Decision details")
    st.markdown(
        """
        <div class="report-body">
        <table>
          <tbody>
            <tr><td><b>Decision</b></td>
                <td>Approve 90-day delivery-reliability initiative</td></tr>
            <tr><td><b>Sponsor</b></td>
                <td>Head of Operations</td></tr>
            <tr><td><b>Accountable lead</b></td>
                <td>Logistics / Operations</td></tr>
            <tr><td><b>Supporting teams</b></td>
                <td>Customer Experience · Data Science · Data Engineering
                    · Regional Management · Marketing</td></tr>
            <tr><td><b>Pilot market</b></td>
                <td>BA (3.86, ~3,340 orders) alongside the RJ diagnostic</td></tr>
            <tr><td><b>Primary success metric</b></td>
                <td>Late-delivery rate (national: currently 8.1%)</td></tr>
            <tr><td><b>Secondary metrics</b></td>
                <td>On-time delivery; avg review score; % orders &gt; 20
                    days; ETA error</td></tr>
            <tr><td><b>Review cadence</b></td>
                <td>Monthly at steering, with a 90-day formal gate</td></tr>
          </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Evidence chain")
    st.markdown(
        """
        1. **Headline:** On-time orders average 4.294 ★, late orders
           average 2.566 ★ — a **1.73-star gap**.
        2. **Breadth:** 7,826 orders (8.1% of delivered) arrive late,
           so the effect touches a minority of customers but is large
           enough to move the national average.
        3. **Geography:** RJ accounts for the worst excess — 13.5% late
           vs 5.9% national, across 12,687 orders.
        4. **Threshold:** Satisfaction drops below the 4.09 national
           average at every delivery bucket beyond 20 days.
        5. **Severity:** When late, orders average 9.55 days past
           estimate, worst case 189 days.
        """
    )

    st.markdown("### Assumptions and caveats")
    st.markdown(
        """
        - The 1.73-star gap is an **association, not proof of
          causation**. Product quality, seller behaviour, distance and
          communication all contribute.
        - 2016 and 2018 are **partial years**; comparisons across
          periods should account for this.
        - Reviews are **voluntary**; scores reflect reviewers, not the
          full customer base.
        - The dashboard does **not** include a delivery-distance
          feature; the RJ diagnosis should add one before scaling.
        - Score improvements should be measured against a **validated
          baseline**, not against projections.
        """
    )

    st.markdown("### Next review")
    st.markdown(
        """
        - **+30 days:** Baseline KPI validation complete; RJ root-cause
          view in place.
        - **+60 days:** Pilot live in RJ; day-20 alert threshold tuned.
        - **+90 days:** Formal decision gate — scale, reshape, or stop.
        """
    )

    st.divider()
    st.caption(
        "**Sign-off block:** This decision record is based on the "
        "verified KPIs and charts reproduced in the Data Analysis "
        "Dashboard tabs with filters = All. Any deviation from the "
        "reference numbers in the sidebar should be investigated before "
        "signature."
    )


# ════════════════════════════════════════════════════════════════════════════
# §8  BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
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

    st.markdown(
        """
        <h1 style="
            background: linear-gradient(90deg, #6D3FB5 0%, #8B5CF6 50%, #D64545 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; font-size: 42px; font-weight: 800;
            letter-spacing: -0.02em; text-align: center;
            margin-bottom: 0; padding-bottom: 0;">
          Customer Satisfaction Dashboard
        </h1>
        <p style='color:#6B7280; font-size:14px; margin-top:4px;
                  text-align:center;'>
          PORA Academy Cohort 7 — Group 1 Capstone Project
        </p>
        <p style='color:#9CA3AF; font-size:13px; margin-top:2px;
                  text-align:center;'>
          📊 Data Analysis Dashboard · 🎯 Decision Dashboard
        </p>
        """,
        unsafe_allow_html=True,
    )

    (
        tab_overview,
        tab_theme,
        tab_detail,
        tab_recommendations,
        tab_priorities,
        tab_decision,
    ) = st.tabs([
        "📊 Overview",
        "📊 Customer Satisfaction",
        "📊 Detail & Method",
        "🎯 Recommendations",
        "🎯 Priorities & Ownership",
        "🎯 Decision Record",
    ])

    with tab_overview:
        render_overview_tab(conn, year, state_json, filter_active)
    with tab_theme:
        render_theme_tab(conn, year, state_json)
    with tab_detail:
        render_detail_tab(conn)
    with tab_recommendations:
        render_recommendations_tab()
    with tab_priorities:
        render_priorities_tab()
    with tab_decision:
        kpis = query_kpis(conn, year, state_json)
        render_decision_record_tab(year, states, kpis)

    st.divider()
    st.caption(
        "**Reference (filters = All):** Total Orders 99,441 · Delivered "
        "96,478 · Avg Review 4.09 · Late 7,826 · Avg Delivery Days 12.56. "
        "Review Score Distribution is intentionally unfiltered — a "
        "whole-dataset fact. Every filterable chart reconciles to the "
        "verified README tables with filters = All."
    )


if __name__ == "__main__":
    main()