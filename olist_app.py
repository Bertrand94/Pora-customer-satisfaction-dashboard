# olist_app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis · Group 1 Capstone

EVERY chart responds to the sidebar filters (Reporting Year + State).

Tabs:
    📊 Overview               — KPI row · Chart 1 (score distribution bar)
                                Chart 2 (monthly trend)
                                Chart 3 (score pie)
    📊 Customer Satisfaction  — Chart 4 (score by state) · Chart 5 (on-time vs late)
                                Chart 6 (punctuality vs satisfaction)
                                Chart 7 (delivery buckets)
                                Chart 8 (Brazil map · complete)
    📊 Detail & Method        — Chart 9 (delivery punctuality pie)
                                limitations + diagnostics
    🎯 Recommendations        — prioritised actions (whole dataset)

Reference KPIs (filters = All):
    Total Orders 99,441 · Delivered 96,478 · Avg Review 4.09
    Late 7,826 · Avg Delivery Days 12.56

Run:
    streamlit run olist_app.py

Environment:
    OLIST_DATA_DIR   folder holding the 4 CSVs (default ./data)
    OLIST_DB_PATH    optional explicit SQLite path
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

@dataclass(frozen=True)
class Palette:
    purple:       str = "#8B5CF6"
    purple_dark:  str = "#6D3FB5"
    purple_light: str = "#C4B5FD"
    red:          str = "#D64545"
    red_light:    str = "#F0A0A0"
    green:        str = "#16A34A"
    green_light:  str = "#86EFAC"
    blue:         str = "#0EA5E9"
    grey:         str = "#9CA3AF"
    grey_light:   str = "#E5E7EB"
    slate:        str = "#3F3F5A"


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
    pie_height:         Final[float] =  5.0
    map_height:         Final[float] =  8.5

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


def fmt_pct(value: Any, places: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{places}f}%"


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
        SELECT ROUND(AVG(CAST(r.review_score AS REAL)), 2) AS avg_review_score
        FROM reviews r
        JOIN orders o    ON r.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {FILTER_SQL}
    """, params=params, label="KPI avg review score")

    return {
        "total_orders":      safe_scalar(df, "total_orders", 0),
        "delivered_orders":  safe_scalar(df, "delivered_orders", 0),
        "avg_review_score":  safe_scalar(score_df, "avg_review_score"),
        "late_deliveries":   safe_scalar(df, "late_deliveries", 0),
        "avg_delivery_days": safe_scalar(df, "avg_delivery_days"),
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  CHART FUNCTIONS  (9 unique charts, no duplication)
# ════════════════════════════════════════════════════════════════════════════

def _no_data_notice(msg: str = "No data for the current filters.") -> None:
    st.info(f"ℹ️ {msg} Try widening the Year or State filters.")


def _insight(year: str, state_json: str, text: str) -> None:
    st.caption(
        f"**Selection:** {scope_label(year, state_json)}  ·  "
        f"**Insight.** {text}"
    )


# ── Chart 1 · Review score distribution (bar) ─────────────────────────────

def chart_review_distribution(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Chart 1 — Review Score Distribution")
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
        PALETTE.purple_dark if s == 5
        else PALETTE.red   if s == 1
        else PALETTE.purple_light
        for s in df["score"]
    ]

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_normal,
                                    CONFIG.chart_height_short))
    bars = ax.bar(df["score"].astype(str), df["reviews"], color=shades)
    ax.set_title("Review Score Distribution")
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Number of reviews")
    ax.set_ylim(0, df["reviews"].max() * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar, pct in zip(bars, df["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10,
                color=PALETTE.purple_dark, fontweight="bold")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    def share(score: int) -> float:
        row = df.loc[df["score"] == score, "pct"]
        return float(row.iloc[0]) if not row.empty else 0.0

    _insight(
        year, state_json,
        f"{share(5):.1f}% of the {total:,} reviews are 5★ and "
        f"{share(1):.1f}% are 1★.",
    )


# ── Chart 2 · Monthly average score (line) ───────────────────────────────

def chart_monthly_score(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Chart 2 — Monthly Average Review Score")
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

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_wide,
                                    CONFIG.chart_height_short))
    ax.plot(df["month"], df["avg_score"], marker="o",
            color=PALETTE.purple_dark, linewidth=2.5, markersize=5)
    ax.set_title("Monthly Average Review Score")
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

    st.pyplot(fig, use_container_width=True)
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


# ── Chart 3 · Review score share (pie) ───────────────────────────────────

def chart_score_pie(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Chart 3 — Review Score Share *(pie)*")
    df = safe_query(conn, f"""
        SELECT CAST(r.review_score AS INTEGER) AS score,
               COUNT(*) AS reviews
        FROM reviews r
        JOIN orders o    ON r.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {FILTER_SQL}
        GROUP BY 1 ORDER BY 1
    """, params=year_multi(year, state_json), label="score pie")

    if df.empty:
        _no_data_notice()
        return

    total = int(df["reviews"].sum())
    color_map = {
        1: PALETTE.red, 2: PALETTE.red_light,
        3: PALETTE.grey,
        4: PALETTE.purple_light, 5: PALETTE.purple_dark,
    }
    colors = [color_map.get(int(s), PALETTE.grey) for s in df["score"]]

    fig, ax = plt.subplots(figsize=(CONFIG.pie_height, CONFIG.pie_height))
    ax.pie(
        df["reviews"],
        labels=[f"{s}★" for s in df["score"]],
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=colors,
        explode=[0.06 if s in (1, 5) else 0 for s in df["score"]],
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    ax.set_title("Review Score Share", fontsize=13, fontweight="bold",
                 color=PALETTE.purple_dark)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    five = df.loc[df["score"] == 5, "reviews"].sum()
    one  = df.loc[df["score"] == 1, "reviews"].sum()
    _insight(
        year, state_json,
        f"{100 * five / total:.1f}% are 5★, "
        f"{100 * one / total:.1f}% are 1★ — {five:,} vs {one:,}.",
    )


# ── Chart 4 · Average review score by state (horizontal bar) ─────────────

def chart_state_scores(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Chart 4 — Average Review Score by State *(top 10)*")
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

    low  = plot_df.iloc[0]
    high = plot_df.iloc[-1]
    if len(plot_df) == 1:
        text = f"{low['state']} averages {low['avg_score']:.2f} stars."
    else:
        text = (
            f"{low['state']} ({low['avg_score']:.2f}) anchors the low "
            f"end; {high['state']} ({high['avg_score']:.2f}) the high end "
            f"— a spread of {high['avg_score'] - low['avg_score']:.2f} stars."
        )
    _insight(year, state_json, text)


# ── Chart 5 · On-time vs late by state (grouped bar) ─────────────────────

def chart_ontime_by_state(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Chart 5 — On-Time vs Late Deliveries by State *(top 5)*")
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
    else:
        text = (
            f"Across the states shown, {overall:.1f}% of delivered orders "
            f"were late. {worst['state']} is highest "
            f"({worst['late_rate']:.1f}%), {best['state']} lowest "
            f"({best['late_rate']:.1f}%)."
        )
    _insight(year, state_json, text)


# ── Chart 6 · Punctuality vs satisfaction ────────────────────────────────

def chart_punctuality_vs_satisfaction(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Chart 6 — Delivery Punctuality vs Satisfaction")
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

    color_map = {"On Time": PALETTE.purple, "Late": PALETTE.red}
    colors = [color_map.get(s, PALETTE.grey) for s in df["delivery_status"]]

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_narrow,
                                    CONFIG.chart_height))
    bars = ax.bar(df["delivery_status"], df["avg_score"], color=colors)
    ax.set_title("Delivery Punctuality vs Satisfaction")
    ax.set_xlabel("")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5.4)
    for bar, score, n, days in zip(
        bars, df["avg_score"], df["orders"], df["avg_delivery_days"]
    ):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.06,
                f"{score:.2f}\n(n={n:,}; {days}d avg)",
                ha="center", va="bottom", fontsize=11,
                color=PALETTE.purple_dark, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    scores = dict(zip(df["delivery_status"], df["avg_score"]))
    if "On Time" in scores and "Late" in scores:
        gap = scores["On Time"] - scores["Late"]
        text = (
            f"Late deliveries cost **{gap:.2f} stars** — "
            f"{scores['On Time']:.2f} on time vs {scores['Late']:.2f} late."
        )
    else:
        only = next(iter(scores))
        text = (
            f"Only {only.lower()} orders exist in this selection, so no "
            "on-time vs late gap can be computed."
        )
    _insight(year, state_json, text)


# ── Chart 7 · Delivery buckets ───────────────────────────────────────────

def chart_satisfaction_by_delivery_bucket(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Chart 7 — Satisfaction by Delivery-Time Bucket")
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
        "0-5 days":   PALETTE.purple_dark,
        "6-10 days":  PALETTE.purple,
        "11-20 days": "#A78BFA",
        "21-30 days": "#F0A0A0",
        "30+ days":   PALETTE.red,
    }
    colors = [palette_by_bucket.get(b, PALETTE.grey)
              for b in df["delivery_bucket"]]

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_normal,
                                    CONFIG.chart_height))
    bars = ax.bar(df["delivery_bucket"], df["avg_score"], color=colors)
    ax.set_title("Satisfaction by Delivery-Time Bucket")
    ax.set_xlabel("Days from purchase to delivery")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5.4)
    ax.axhline(selection_avg, color=PALETTE.grey, linestyle="--",
               linewidth=1, label=f"Selection avg {selection_avg:.2f}")
    ax.legend(loc="lower left", frameon=False)
    for bar, score, n in zip(bars, df["avg_score"], df["orders"]):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.06,
                f"{score:.2f}\n(n={n:,})",
                ha="center", va="bottom", fontsize=10,
                color=PALETTE.purple_dark, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if len(df) == 1:
        text = (
            f"All delivered orders in this selection fall in the "
            f"{df.iloc[0]['delivery_bucket']} bucket "
            f"({df.iloc[0]['avg_score']:.2f} stars)."
        )
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
    _insight(year, state_json, text)


# ── Chart 8 · Brazil map (COMPLETE · matplotlib) ─────────────────────────

@st.cache_data(ttl=CONFIG.geojson_cache_ttl_seconds, show_spinner=False)
def load_brazil_geojson() -> dict[str, Any]:
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
    raise RuntimeError(f"All GeoJSON mirrors failed. Last error: {last_err}")


def _geojson_to_gdf(geojson: dict[str, Any]) -> pd.DataFrame:
    """Convert a Brazil-states GeoJSON FeatureCollection into a dataframe
    with one row per state.  Uses shapely + matplotlib PathPatch, avoiding
    a hard geopandas dependency."""
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
    linewidth: float = 0.8,
) -> None:
    """Draw a single GeoJSON geometry (Polygon / MultiPolygon) on ax."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not gtype or not coords:
        return

    polygons: list[list[list[tuple[float, float]]]] = []
    if gtype == "Polygon":
        polygons = [coords]
    elif gtype == "MultiPolygon":
        polygons = coords

    for poly in polygons:
        # poly is a list of rings; ring 0 is the exterior
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
    """Chart 8 · Complete Brazil map — all 27 states shown, colored by the
    average review score for the current selection.  States not in the
    current filter are rendered in a light grey but remain visible."""
    st.markdown("### Chart 8 — Brazil Map · Avg Review Score *(complete)*")

    df = safe_query(conn, f"""
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE {YEAR_SQL}
        GROUP BY state
    """, params=year_params(year), label="Brazil map data")

    if df.empty:
        _no_data_notice("No state-level data for this year.")
        return

    selected       = selected_states(state_json)
    filter_active  = len(selected) > 0

    try:
        geojson = load_brazil_geojson()
    except RuntimeError as exc:
        _LOGGER.error("GeoJSON unavailable: %s", exc)
        st.warning("⚠️ The Brazil map is temporarily unavailable.")
        return

    gdf = _geojson_to_gdf(geojson)
    if gdf.empty:
        st.warning("⚠️ No state geometries could be parsed.")
        return

    score_by_state = dict(zip(df["state"], df["avg_score"]))
    orders_by_state = dict(zip(df["state"], df["orders"]))

    # Determine colour scale domain from the states that actually have data
    scores_in_scope = [
        score_by_state[s] for s in score_by_state
        if score_by_state[s] is not None
    ]
    if not scores_in_scope:
        _no_data_notice()
        return

    lo_c = max(0.5, float(min(scores_in_scope)) - 0.05)
    hi_c = float(max(scores_in_scope)) + 0.05
    if hi_c - lo_c < 0.1:
        hi_c = lo_c + 0.1

    cmap = plt.get_cmap("RdYlGn")
    norm = mpl.colors.Normalize(vmin=lo_c, vmax=hi_c)

    fig, ax = plt.subplots(figsize=(CONFIG.chart_width_wide,
                                    CONFIG.map_height))
    ax.set_axis_off()
    ax.set_title(
        f"Average Review Score by State — {scope_label(year, state_json)}",
        fontsize=13, fontweight="bold", color=PALETTE.purple_dark,
        pad=14,
    )

    # First pass: draw all states so the outline of Brazil is complete
    for _, row in gdf.iterrows():
        state  = row["state"]
        geom   = row["geometry"]
        score  = score_by_state.get(state)
        in_filter = (not filter_active) or (state in selected)

        if score is not None and in_filter:
            facecolor = cmap(norm(score))
        else:
            facecolor = PALETTE.grey_light

        _plot_geojson_state(ax, geom, facecolor=facecolor)

    # Second pass: label the centroid of every state
    for _, row in gdf.iterrows():
        state = row["state"]
        geom  = row["geometry"]
        score = score_by_state.get(state)
        in_filter = (not filter_active) or (state in selected)

        # Compute a rough centroid from the outer ring
        gtype = geom.get("type")
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

        if score is not None and in_filter:
            label   = f"{state}\n{score:.2f}"
            color   = "#1F2937"
            weight  = "bold"
        elif score is not None:
            label   = f"{state}\n{score:.2f}"
            color   = "#9CA3AF"
            weight  = "normal"
        else:
            label   = state
            color   = "#9CA3AF"
            weight  = "normal"

        ax.text(
            cx, cy, label,
            ha="center", va="center",
            fontsize=7.5, color=color, fontweight=weight, zorder=5,
        )

    # Brazil bounds — fixed so the map is always complete
    ax.set_xlim(-74.0, -34.0)
    ax.set_ylim(-34.0,  6.0)
    ax.set_aspect("equal")

    # Horizontal colour bar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal",
        fraction=0.04, pad=0.04, shrink=0.7,
    )
    cbar.set_label("Average review score (1–5)", fontsize=10,
                   color=PALETTE.purple_dark)
    cbar.ax.tick_params(labelsize=9)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    in_scope = [
        (s, score_by_state[s]) for s in score_by_state
        if score_by_state[s] is not None
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
        "All 27 states are coloured by their average review score."
    )
    _insight(
        year, state_json,
        f"{lo_s} ({lo_v:.2f}) is lowest and {hi_s} ({hi_v:.2f}) is "
        f"highest among states in the current selection. {mode_note}",
    )


# ── Chart 9 · Delivery punctuality pie ───────────────────────────────────

def chart_punctuality_pie(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown("### Chart 9 — Delivery Punctuality Share *(pie)*")
    df = safe_query(conn, f"""
        SELECT CASE WHEN o.order_delivered_customer_date
                         <= o.order_estimated_delivery_date
                    THEN 'On Time' ELSE 'Late' END AS delivery_status,
               COUNT(DISTINCT o.order_id) AS orders
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {DELIVERED_SQL} AND {FILTER_SQL}
        GROUP BY 1
    """, params=year_multi(year, state_json), label="punctuality pie")

    if df.empty:
        _no_data_notice()
        return

    total = int(df["orders"].sum())
    color_map = {"On Time": PALETTE.green, "Late": PALETTE.red}
    colors = [color_map.get(s, PALETTE.grey) for s in df["delivery_status"]]

    fig, ax = plt.subplots(figsize=(CONFIG.pie_height, CONFIG.pie_height))
    ax.pie(
        df["orders"],
        labels=df["delivery_status"],
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * total)):,})",
        startangle=90,
        colors=colors,
        explode=[0.08 if s == "Late" else 0 for s in df["delivery_status"]],
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title("On-Time vs Late Delivery Share",
                 fontsize=13, fontweight="bold", color=PALETTE.purple_dark)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    late = int(df.loc[df["delivery_status"] == "Late", "orders"].sum())
    late_pct = 100.0 * late / total if total else 0.0
    _insight(
        year, state_json,
        f"{late:,} of {total:,} delivered orders ({late_pct:.1f}%) "
        f"arrived after the estimated date.",
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

  .scope-pill {
    display: inline-block; background: #FFFFFF; color: #6D3FB5;
    border: 1px solid #C4B5FD; border-radius: 16px;
    padding: 3px 12px; font-size: 12px; font-weight: 700;
    margin-bottom: 6px;
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
        "Both filters apply to **every KPI and chart**. "
        "Set both to *All* to reproduce the reference values."
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
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

def render_overview_tab(
    conn: sqlite3.Connection, year: str, state_json: str,
    filter_active: bool,
) -> None:
    st.markdown(
        '<span class="mode-badge mode-analysis">📊 Data Analysis</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Overview")

    k = query_kpis(conn, year, state_json)
    c1, c2, c3, c4, c5 = st.columns(5)
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
            f"Active filter — **{scope_label(year, state_json)}**. "
            "Set both filters to All to reproduce the sidebar reference "
            "values."
        )

    st.divider()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        chart_review_distribution(conn, year, state_json)
    with col_right:
        chart_monthly_score(conn, year, state_json)

    st.divider()

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        chart_score_pie(conn, year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SATISFACTION
# ════════════════════════════════════════════════════════════════════════════

def render_theme_tab(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown(
        '<span class="mode-badge mode-analysis">📊 Data Analysis</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Customer Satisfaction")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        chart_state_scores(conn, year, state_json)
    with col_b:
        chart_ontime_by_state(conn, year, state_json)

    st.divider()
    chart_punctuality_vs_satisfaction(conn, year, state_json)

    st.divider()
    col_c, col_d = st.columns([1, 1])
    with col_c:
        chart_satisfaction_by_delivery_bucket(conn, year, state_json)
    with col_d:
        st.empty()  # placeholder to keep 2-col rhythm

    st.divider()
    chart_brazil_map(conn, year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DETAIL & METHOD
# ════════════════════════════════════════════════════════════════════════════

def render_detail_tab(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    st.markdown(
        '<span class="mode-badge mode-analysis">📊 Data Analysis</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Detail & Method")

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        chart_punctuality_pie(conn, year, state_json)

    st.divider()

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

    st.markdown("**Days late when late (whole dataset)**")
    if not days_late.empty:
        st.dataframe(days_late, hide_index=True, use_container_width=True)
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
    st.markdown(
        '<span class="mode-badge mode-decision">🎯 Decision</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Recommendations")
    st.markdown(
        "Prioritised actions following from the headline finding. "
        "They are based on the **whole dataset**, so they do not change "
        "when the sidebar filters move — a decision should not flip "
        "because someone narrowed the view."
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

    st.markdown(
        """
        <h1 style="
            background: linear-gradient(90deg, #6D3FB5 0%, #8B5CF6 50%, #D64545 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; font-size: 40px; font-weight: 800;
            letter-spacing: -0.02em; text-align: center;
            margin-bottom: 0; padding-bottom: 0;">
          Customer Satisfaction Dashboard
        </h1>
        <p style='color:#6B7280; font-size:14px; margin-top:4px;
                  text-align:center;'>
          PORA Academy Cohort 7 — Group 1 Capstone Project
        </p>
        """,
        unsafe_allow_html=True,
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
        "📊 Overview",
        "📊 Customer Satisfaction",
        "📊 Detail & Method",
        "🎯 Recommendations",
    ])

    with tab_overview:
        render_overview_tab(conn, year, state_json, filter_active)
    with tab_theme:
        render_theme_tab(conn, year, state_json)
    with tab_detail:
        render_detail_tab(conn, year, state_json)
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