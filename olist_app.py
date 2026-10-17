# app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis · Group 1 Capstone

Production-grade Streamlit dashboard for the Olist customer-satisfaction
analysis. Six curated charts, seven tabs, animated headline, and
fully-connected Year + State filters on every KPI and chart.

Architecture
------------
The module is organised into numbered sections:

    §1  Configuration         — tunables, palettes, constants
    §2  Logging               — structured logging setup
    §3  Data layer            — connection, schema validation, loaders
    §4  KPI computation       — pure functions, filter-aware
    §5  Chart functions       — pure data + plotting, each in isolation
    §6  UI helpers            — reusable rendering primitives
    §7  Presentation layer    — page layout, tabs, report content
    §8  Bootstrap             — main() entry point

Design principles
-----------------
*   **Single source of truth** — all tunables live in `CONFIG`.
*   **Fail gracefully** — every query is guarded; empty results render
    an informative notice instead of crashing.
*   **Pure data functions** — `query_*` functions take a connection and
    filters, return a DataFrame. They are unit-testable without Streamlit.
*   **No string interpolation in SQL** — all parameters are bound.
*   **Explicit types** — every function signature is annotated.
*   **No dead code** — unused helpers removed; imports tightened.

Run
---
    streamlit run app.py

Environment
-----------
    OLIST_DATA_DIR   Path to the folder containing the five CSVs
                     (default: ./data next to this file)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import closing
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
    """Brand colours used across charts and UI chrome."""
    purple:       str = "#8B5CF6"
    purple_dark:  str = "#6D3FB5"
    red:          str = "#D64545"
    grey:         str = "#9CA3AF"
    green:        str = "#16A34A"
    blue:         str = "#0EA5E9"
    purple_light: str = "#C4B5FD"


@dataclass(frozen=True)
class Config:
    """All tunables in one place. Import-free, self-documenting."""
    # ── Data layer ─────────────────────────────────────────────────────
    tables: Final[dict[str, str]] = field(default_factory=lambda: {
        "orders":      "olist_orders_dataset.csv",
        "customers":   "olist_customers_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments":    "olist_order_payments_dataset.csv",
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
            "reviews":   ("review_id", "order_id", "review_score"),
        }
    )

    # ── Reference values from the brief ────────────────────────────────
    reference_national_avg: Final[float] = 4.09
    reference_on_time_avg:  Final[float] = 4.29
    reference_late_avg:     Final[float] = 2.57

    # ── Filter contract ────────────────────────────────────────────────
    all_states: Final[tuple[str, ...]] = (
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
        "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
        "RO", "RR", "RS", "SC", "SE", "SP", "TO",
    )
    years: Final[tuple[str, ...]] = ("All", "2016", "2017", "2018")

    # ── External resource ──────────────────────────────────────────────
    geojson_urls: Final[tuple[str, ...]] = (
        "https://raw.githubusercontent.com/codeforamerica/"
        "click_that_hood/master/public/data/brazil-states.geojson",
        "https://raw.githubusercontent.com/tbrugz/"
        "vector-json/master/brazil-states.geojson",
    )
    geojson_timeout_seconds: Final[int] = 20

    # ── Cache / TTL ────────────────────────────────────────────────────
    data_cache_ttl_seconds: Final[int] = 3600
    geojson_cache_ttl_seconds: Final[int] = 86400

    # ── Chart geometry ─────────────────────────────────────────────────
    chart_width_wide:   Final[float] = 11.0
    chart_width_normal: Final[float] = 10.0
    chart_width_narrow: Final[float] =  9.0
    chart_height:       Final[float] =  5.5
    chart_height_short: Final[float] =  4.5

    # ── Taxonomies ─────────────────────────────────────────────────────
    delivery_buckets: Final[tuple[tuple[int, str], ...]] = (
        (5,  "0-5 days"),
        (10, "6-10 days"),
        (20, "11-20 days"),
        (30, "21-30 days"),
    )
    delivery_bucket_overflow: Final[str] = "30+ days"


CONFIG    = Config()
PALETTE   = Palette()


# ════════════════════════════════════════════════════════════════════════════
# §2  LOGGING
# ════════════════════════════════════════════════════════════════════════════

_LOGGER = logging.getLogger("olist.dashboard")


def _configure_logging() -> None:
    """Configure module-level logging once. Idempotent."""
    if _LOGGER.handlers:
        return
    level_name = os.environ.get("OLIST_LOG_LEVEL", "INFO").upper()
    level      = getattr(logging, level_name, logging.INFO)
    handler    = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(level)
    _LOGGER.propagate = False


_configure_logging()


# ════════════════════════════════════════════════════════════════════════════
# §3  DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

_HERE: Final[str] = os.path.dirname(os.path.abspath(__file__))


def _data_dir() -> str:
    """Resolve the CSV directory from env or local default."""
    return os.environ.get("OLIST_DATA_DIR", os.path.join(_HERE, "data"))


def _resolve_db_path() -> str:
    """Return the SQLite database path (Colab-safe, local-safe)."""
    custom = os.environ.get("OLIST_DB_PATH")
    if custom:
        return custom
    on_colab = os.path.isdir("/content")
    return "/content/olist.db" if on_colab else os.path.join(
        _HERE, "olist.db"
    )


def _validate_dataframes(frames: dict[str, pd.DataFrame]) -> None:
    """
    Validate the loaded dataframes against the expected schema.

    Raises
    ------
    ValueError
        If a required table or column is missing, or a table is empty.
    """
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
        len(frames),
        sum(len(df) for df in frames.values()),
    )


@st.cache_resource(
    ttl=CONFIG.data_cache_ttl_seconds, show_spinner="Loading Olist data…"
)
def build_connection() -> sqlite3.Connection:
    """
    Build a file-backed SQLite database from the five Olist CSVs.

    The database is file-backed rather than in-memory because
    jupysql-style tooling requires a stable path. The connection is
    shared across sessions and cached for one hour.
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
        _LOGGER.debug("Loaded %s: %d rows from %s", table,
                      len(frames[table]), path)

    _validate_dataframes(frames)

    db_path = _resolve_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    for table, df in frames.items():
        df.to_sql(table, conn, if_exists="replace", index=False)

    _LOGGER.info("Database built at %s", db_path)
    return conn


# ── Filter parameter helpers ───────────────────────────────────────────────

def year_params(year: str) -> tuple[str, str]:
    """Return the two bind variables used by every year-guarded query."""
    return (year, year)


def multi_params(state_json: str) -> tuple[str, str]:
    """Return the two bind variables used by every state-guarded query."""
    return (state_json, state_json)


def year_multi(year: str, state_json: str) -> tuple[str, str, str, str]:
    """Return the four bind variables used by combined-guard queries."""
    return (year, year, state_json, state_json)


def safe_state_json(raw_states: Iterable[Any] | None) -> str:
    """
    Serialise a state selection to a JSON array.

    Guards against None, non-iterable values, empty lists, malformed
    elements, and the "All" sentinel. Always returns a valid JSON array.
    """
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


# ── Safe query wrapper ────────────────────────────────────────────────────

def safe_query(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
    *,
    label: str = "query",
) -> pd.DataFrame:
    """
    Execute a parameterised query, always returning a DataFrame.

    On error, logs the failure and returns an empty DataFrame so that
    downstream visualisations degrade gracefully.
    """
    try:
        return pd.read_sql(sql, conn, params=params)
    except (sqlite3.Error, pd.errors.DatabaseError, ValueError) as exc:
        _LOGGER.warning("%s failed gracefully: %s", label, exc)
        st.warning(
            f"⚠️ The **{label}** query could not complete. "
            "An empty result is being shown."
        )
        return pd.DataFrame()


def safe_scalar(
    df: pd.DataFrame, column: str, default: Any = None
) -> Any:
    """Return the first-row value of a column, or `default`."""
    if df is None or df.empty or column not in df.columns:
        return default
    value = df.iloc[0][column]
    return default if pd.isna(value) else value


def safe_metric_display(
    value: Any, fmt: str = "{:,.0f}", default: str = "—"
) -> str:
    """Format a metric value safely, returning `default` on failure."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


# ════════════════════════════════════════════════════════════════════════════
# §4  KPI COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def query_kpis(
    conn: sqlite3.Connection, year: str, state_json: str
) -> dict[str, Any]:
    """
    Compute the five headline KPIs, scoped to the active filters.

    Returns
    -------
    dict
        Keys: ``total_orders``, ``delivered_orders``,
        ``avg_review_score``, ``late_deliveries``,
        ``avg_delivery_days``. Missing values are ``0`` for counts and
        ``None`` for averages.
    """
    df = safe_query(conn, """
        SELECT COUNT(DISTINCT o.order_id) AS total_orders,
               COUNT(DISTINCT CASE WHEN o.order_status = 'delivered'
                                    THEN o.order_id END)
                   AS delivered_orders,
               ROUND(AVG(CAST(r.review_score AS REAL)), 2)
                   AS avg_review_score,
               SUM(CASE WHEN o.order_status = 'delivered'
                         AND o.order_delivered_customer_date
                              > o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS late_deliveries,
               ROUND(AVG(CASE WHEN o.order_status = 'delivered'
                                AND o.order_delivered_customer_date
                                    IS NOT NULL
                              THEN julianday(o.order_delivered_customer_date)
                                 - julianday(o.order_purchase_timestamp)
                              END), 2) AS avg_delivery_days
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
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
# §5  CHART FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _no_data_notice(message: str = "No data for the current filters.") -> None:
    """Render a standardised no-data advisory."""
    st.info(f"ℹ️ {message} Try widening the Year or State filters.")


# ── Chart 1 — Review score distribution (whole dataset) ───────────────────

def chart_review_distribution(conn: sqlite3.Connection) -> None:
    """
    Plot the whole-dataset review score distribution.

    This chart is deliberately unfiltered — it is a structural fact about
    the dataset that should not change when the user slices by year or
    state.
    """
    df = safe_query(conn, """
        SELECT CAST(review_score AS INTEGER) AS score,
               COUNT(*)                     AS reviews,
               ROUND(100.0 * COUNT(*)
                     / (SELECT COUNT(*) FROM reviews), 1) AS pct
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


# ── Chart 2 — Monthly average review score ────────────────────────────────

def chart_monthly_score(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    """Plot the monthly average review score across the active filters."""
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


# ── Chart 3 — Brazil choropleth ────────────────────────────────────────────

@st.cache_data(ttl=CONFIG.geojson_cache_ttl_seconds, show_spinner=False)
def load_brazil_geojson() -> dict[str, Any]:
    """
    Fetch and cache the Brazil-states GeoJSON from public mirrors.

    Tries each mirror in turn. Raises ``RuntimeError`` if all fail.
    """
    last_err: Exception | None = None
    for url in CONFIG.geojson_urls:
        try:
            response = requests.get(url, timeout=CONFIG.geojson_timeout_seconds)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            _LOGGER.warning("GeoJSON mirror failed: %s — %s", url, exc)
            last_err = exc
    raise RuntimeError(
        f"All GeoJSON mirrors failed. Last error: {last_err}"
    )


@st.cache_data(show_spinner=False)
def state_centroids() -> dict[str, tuple[float, float]]:
    """Return approximate (lat, lon) centroids for all 27 states."""
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
    """
    Render a Brazil choropleth showing the top 10 states by order volume.

    Selected states from the sidebar appear in colour; other states (if
    they happen to be in the top 10 by volume) appear greyed. Each
    polygon is labelled with the state code and its average score.
    """
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

    top10        = df["state"].tolist()
    keep_states  = (
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


# ── Chart 4 — On-time vs late by state ────────────────────────────────────

def chart_ontime_by_state(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    """Plot on-time vs late delivery counts for the top 5 states."""
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


# ── Chart 5 — Punctuality vs satisfaction (headline) ──────────────────────

def chart_punctuality_vs_satisfaction(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    """Plot the satisfaction gap between on-time and late deliveries."""
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


# ── Chart 6 — Satisfaction by delivery-time bucket ────────────────────────

def chart_satisfaction_by_delivery_bucket(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    """Plot average review score across delivery-time buckets."""
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
    """
    Render a styled card used across Limitations, Recommendations and
    Future Research tabs.

    Parameters
    ----------
    title
        Card heading.
    body_html
        Body HTML (already trusted — built by us, not user input).
    variant
        One of ``'info'``, ``'warn'``, ``'ok'``. Controls the left
        accent bar.
    tags
        Optional pill labels shown above the body.
    """
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


def render_projection_headline() -> None:
    """
    Render the animated headline banner.

    The three headline facts are presented as compact horizontal
    blocks — label, headline number, and supporting detail on a single
    row (wrapping to two lines on narrow viewports). One ``st.markdown``
    call per block keeps CSS animations intact across Streamlit
    versions and avoids whitespace-induced layout gaps.
    """
    with st.container():
        st.markdown(
            '<div class="projection-frame"><div class="projection-content">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="projection-title">Headline</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="projection-underline"></div>',
            unsafe_allow_html=True,
        )

        # ── Block 1 — Punctuality gap ────────────────────────────────
        st.markdown(
            '<div class="projection-block b1">'
            '<span class="block-label">Punctuality gap</span>'
            '<span class="block-number">1.73'
            '<span class="block-unit"> stars</span></span>'
            '<span class="block-detail">'
            'On-time deliveries <span class="hl">4.29</span> vs late '
            '<span class="hl">2.57</span> — the strongest satisfaction '
            'driver in the dataset.'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Block 2 — Late-delivery exposure ─────────────────────────
        st.markdown(
            '<div class="projection-block b2">'
            '<span class="block-label">Late-delivery exposure</span>'
            '<span class="block-number">8.1'
            '<span class="block-unit">% of delivered orders</span></span>'
            '<span class="block-detail">'
            'Late orders run <span class="hl">9.55 days</span> beyond '
            'the estimated delivery date.'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Block 3 — Regional outlier ───────────────────────────────
        st.markdown(
            '<div class="projection-block b3">'
            '<span class="block-label">Regional outlier</span>'
            '<span class="block-number">13.5'
            '<span class="block-unit">% late in RJ</span></span>'
            '<span class="block-detail">'
            'vs <span class="hl">5.9%</span> nationally — 2.3× the '
            'benchmark.'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Priority bar + caveat ────────────────────────────────────
        st.markdown(
            '<div class="projection-priority">'
            '<span class="priority-label">PRIORITY ·</span> '
            '<span class="priority-text">'
            'Improve delivery reliability, realistic delivery estimates, '
            'proactive communication, and regional logistics.'
            '</span>'
            '<span class="projection-caveat">'
            'These findings show an association, not proof of causation.'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '</div></div>',
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
    color: #EDE9FE !important;
    font-weight: 600; text-transform: uppercase;
    font-size: 12px; letter-spacing: 0.06em;
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
  .report-body h2 {
    color: #4C1D95; border-bottom: 2px solid #EDE7FA;
    padding-bottom: 6px; margin-top: 28px; margin-bottom: 14px;
  }
  .report-body h3 {
    color: #6D3FB5; margin-top: 22px; margin-bottom: 10px;
  }
  .report-body table {
    width: 100%; border-collapse: collapse;
    margin: 12px 0; font-size: 14px;
  }
  .report-body table th {
    background: #F5F0FF; color: #4C1D95; padding: 10px 12px;
    text-align: left; border-bottom: 2px solid #EDE7FA;
  }
  .report-body table td {
    padding: 10px 12px; border-bottom: 1px solid #F0EBFA;
    vertical-align: top;
  }
  .report-body .callout {
    background: linear-gradient(90deg, #F5F0FF 0%, #EDE7FA 100%);
    border-left: 4px solid #6D3FB5; padding: 14px 20px;
    border-radius: 8px; margin: 16px 0;
  }

  @keyframes projectionFadeIn {
    0%   { opacity: 0; transform: translateY(14px) scale(0.97);
           filter: blur(6px); }
    60%  { opacity: 0.9; filter: blur(1px); }
    100% { opacity: 1; transform: translateY(0) scale(1);
           filter: blur(0); }
  }
  @keyframes projectionGlow {
    0%   { box-shadow: 0 0 0 rgba(139, 92, 246, 0.0),
                       inset 0 0 40px rgba(139, 92, 246, 0.05); }
    50%  { box-shadow: 0 0 32px rgba(139, 92, 246, 0.20),
                       inset 0 0 60px rgba(139, 92, 246, 0.08); }
    100% { box-shadow: 0 0 0 rgba(139, 92, 246, 0.0),
                       inset 0 0 40px rgba(139, 92, 246, 0.05); }
  }
  @keyframes scanLine {
    0%   { top: -8%; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 108%; opacity: 0; }
  }
  @keyframes numberPulse {
    0%   { text-shadow: 0 0 0 rgba(214, 69, 69, 0.0); }
    30%  { text-shadow: 0 0 18px rgba(214, 69, 69, 0.45),
                        0 0 32px rgba(214, 69, 69, 0.25); }
    100% { text-shadow: 0 0 0 rgba(214, 69, 69, 0.0); }
  }
  @keyframes slideInLeft {
    0%   { opacity: 0; transform: translateX(-24px); }
    100% { opacity: 1; transform: translateX(0); }
  }
  @keyframes underlineSweep {
    0%   { transform: scaleX(0); transform-origin: left; }
    100% { transform: scaleX(1); transform-origin: left; }
  }

  .projection-frame {
    position: relative; overflow: hidden;
    background:
      radial-gradient(circle at 15% 20%,
                      rgba(139,92,246,0.10) 0%, transparent 45%),
      radial-gradient(circle at 85% 80%,
                      rgba(214,69,69,0.07) 0%, transparent 45%),
      linear-gradient(120deg, #FBF8FF 0%, #F2ECFB 60%, #EFE9F7 100%);
    border: 1px solid #E5DEF7; border-radius: 14px;
    padding: 26px 32px 24px 32px; margin: 8px 0 22px 0;
    animation: projectionGlow 4.5s ease-in-out infinite;
  }
  .projection-frame::after {
    content: ""; position: absolute; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent 0%,
                rgba(139,92,246,0.55) 50%, transparent 100%);
    filter: blur(0.5px);
    animation: scanLine 3.2s ease-in-out infinite;
    pointer-events: none;
  }
  .projection-content { position: relative; z-index: 2; }
  .projection-title {
    font-size: 13px; font-weight: 800; letter-spacing: 0.22em;
    text-transform: uppercase; color: #6D3FB5; margin: 0 0 6px 0;
    opacity: 0;
    animation: projectionFadeIn 0.9s ease-out 0.05s forwards;
  }
  .projection-underline {
    height: 2px;
    background: linear-gradient(90deg, #6D3FB5 0%,
                #8B5CF6 60%, transparent 100%);
    margin: 0 0 16px 0; transform: scaleX(0); transform-origin: left;
    animation: underlineSweep 1.1s ease-out 0.35s forwards;
  }

  /* ── Headline fact blocks ──────────────────────────────────────── */
  .projection-block {
    display: flex; align-items: baseline; flex-wrap: wrap;
    column-gap: 14px; row-gap: 2px;
    padding: 10px 16px; margin: 0 0 10px 0;
    background: rgba(255, 255, 255, 0.62);
    border-left: 3px solid #8B5CF6; border-radius: 8px;
    opacity: 0;
    animation: projectionFadeIn 1.0s ease-out forwards;
  }
  .projection-block.b1 { animation-delay: 0.55s; }
  .projection-block.b2 { animation-delay: 1.15s; }
  .projection-block.b3 { animation-delay: 1.75s; }

  .projection-block .block-label {
    font-size: 11px; font-weight: 800; letter-spacing: 0.16em;
    text-transform: uppercase; color: #6D3FB5;
    white-space: nowrap; flex: 0 0 auto;
  }
  .projection-block .block-number {
    font-size: 26px; font-weight: 800; color: #D64545;
    line-height: 1.1; letter-spacing: -0.02em;
    white-space: nowrap; flex: 0 0 auto;
    animation: numberPulse 2.4s ease-in-out infinite;
  }
  .projection-block .block-number .block-unit {
    font-size: 13px; font-weight: 700; color: #6D3FB5;
    letter-spacing: 0; margin-left: 2px; animation: none;
  }
  .projection-block .block-detail {
    font-size: 15px; color: #3F3D56; line-height: 1.45;
    flex: 1 1 auto; min-width: 0;
  }
  .projection-block .block-detail .hl {
    color: #6D3FB5; font-weight: 700;
  }

  .projection-priority {
    margin-top: 14px; padding: 14px 20px;
    background: linear-gradient(90deg,
                rgba(139,92,246,0.10) 0%,
                rgba(139,92,246,0.04) 100%);
    border-left: 4px solid #8B5CF6; border-radius: 8px;
    opacity: 0; transform: translateX(-24px);
    animation: slideInLeft 1.1s ease-out 2.4s forwards;
  }
  .projection-priority .priority-label {
    display: inline-block; font-weight: 800; color: #4C1D95;
    letter-spacing: 0.05em; margin-right: 6px;
  }
  .projection-priority .priority-text {
    font-size: 15.5px; color: #3F3D56; line-height: 1.7;
  }
  .projection-priority .projection-caveat {
    display: block; margin-top: 10px; color: #6B7280;
    font-style: italic; font-size: 14px; opacity: 0;
    animation: projectionFadeIn 1.0s ease-out 3.1s forwards;
  }

  /* Narrow screens — let the detail wrap under the number cleanly. */
  @media (max-width: 640px) {
    .projection-block { row-gap: 4px; }
    .projection-block .block-number { font-size: 22px; }
    .projection-block .block-detail { flex-basis: 100%; font-size: 14px; }
  }
</style>
"""


def _apply_global_style() -> None:
    """Apply CSS + matplotlib defaults once at startup."""
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


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, list[str]]:
    """Render the filter sidebar and return (year, states) selections."""
    st.sidebar.title("📊 Filters")
    st.sidebar.markdown(
        "Both filters apply to every KPI card and every chart. "
        "Set to *All* to reproduce the unfiltered reference values."
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

    # Normalise: empty → ["All"]; "All" mixed with others → ["All"]
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


# ── Overview tab ──────────────────────────────────────────────────────────

def render_overview_tab(
    conn: sqlite3.Connection, year: str, state_json: str,
    filter_active: bool,
) -> None:
    """Render the Overview tab: KPI row, insights card, headline chart."""
    k = query_kpis(conn, year, state_json)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders",     safe_metric_display(k["total_orders"]))
    c2.metric("Delivered Orders", safe_metric_display(k["delivered_orders"]))

    avg_score = k["avg_review_score"]
    if avg_score is None:
        c3.metric("Avg Review Score", "—")
    else:
        delta = (
            f"{avg_score - CONFIG.reference_national_avg:+.2f} vs reference"
            if filter_active else None
        )
        c3.metric("Avg Review Score", f"{avg_score:.2f}", delta=delta)

    late_val      = k["late_deliveries"]
    delivered_val = k["delivered_orders"]
    late_delta = (
        f"{100 * late_val / delivered_val:.1f}% of delivered"
        if delivered_val and delivered_val > 0 else None
    )
    c4.metric("Late Deliveries", safe_metric_display(late_val),
              delta=late_delta, delta_color="inverse")

    days = k["avg_delivery_days"]
    c5.metric("Avg Delivery Days",
              f"{days:.2f}" if days is not None else "—")

    if filter_active:
        st.info(
            f"Active filter — **Year:** {year}. "
            "Toggle to All to reproduce the sidebar reference values."
        )

    st.divider()
    st.markdown(
        """
        <div class="info-card">
          <h4>Three insights from the data</h4>
          <ol style="margin-bottom:0; color:#3F3D56; line-height:1.75;">
            <li><b>Late deliveries cost 1.73 stars.</b> On-time = 4.29;
                late = 2.57. Only 8.1% of orders, but the effect is the
                strongest driver in the dataset.</li>
            <li><b>RJ is the operational outlier.</b> 13.5% late rate —
                2.3× the national average — with 12,687 orders of
                leverage.</li>
            <li><b>The decline is smooth, not a cliff.</b> Scores drop
                from 4.45 to 2.26 across five delivery-time buckets;
                every bucket past 20 days falls below the national
                average.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    chart_punctuality_vs_satisfaction(conn, year, state_json)


# ── Key Charts tab ────────────────────────────────────────────────────────

def render_charts_tab(
    conn: sqlite3.Connection, year: str, state_json: str
) -> None:
    """Render the five filterable charts."""
    st.markdown(
        "Five curated, filterable charts. Redundant and low-value "
        "visuals from the original notebook were removed — every chart "
        "on this tab earns its place."
    )
    chart_review_distribution(conn)
    st.divider()
    chart_monthly_score(conn, year, state_json)
    st.divider()
    chart_brazil_map(conn, year, state_json)
    st.divider()
    chart_ontime_by_state(conn, year, state_json)
    st.divider()
    chart_satisfaction_by_delivery_bucket(conn, year, state_json)


# ── Limitations tab ───────────────────────────────────────────────────────

def render_limitations_tab() -> None:
    """Render the seven documented limitations."""
    st.markdown("## Limitations, Data Risks and Interpretation Boundaries")
    st.markdown(
        "Every analytical project has boundaries. Naming them honestly "
        "is part of the deliverable — it tells future analysts exactly "
        "where to pick up."
    )

    info_card(
        "L1 — Observational results do not establish causality",
        """
        <p>The code explicitly recognises that the <b>1.73-star
        difference is correlation, not causation</b>. Product category,
        distance, seller quality and other unmeasured variables may
        affect both delivery timing and ratings.</p>
        <p>Before claiming that a specific intervention will increase
        ratings by a precise amount, use a controlled pilot or a
        regression model that controls for relevant factors.</p>
        """,
        variant="warn",
        tags=["Causality", "Confounders"],
    )
    info_card(
        "L2 — Incomplete time coverage",
        """
        <p>2016 contains only <b>September, October and December</b>,
        with <b>no November orders</b>, while 2018 ends on <b>17
        October</b>. <b>2017 is the only complete calendar year</b> in
        this dataset.</p>
        <p>Year-on-year comparisons that treat these periods as complete
        years may mislead. Reports should state the covered dates and
        compare equivalent periods where possible.</p>
        """,
        variant="warn",
        tags=["Time coverage", "YoY risk"],
    )
    info_card(
        "L3 — Review-row fan-out and customer identity",
        """
        <p>The dataset has <b>99,224 review rows across 98,673 distinct
        orders</b> — a 551-row difference. Some orders have multiple
        review records. The dashboard protects order counts by using
        <code>COUNT(DISTINCT order_id)</code>, but raw joins can still
        inflate counts if the grain of each table is not handled
        carefully.</p>
        <p>The customers table also has <b>99,441 rows but only 96,096
        distinct <code>customer_unique_id</code></b> values.
        Person-level retention or lifetime-value analysis should use the
        stable unique identifier.</p>
        """,
        variant="warn",
        tags=["Data quality", "Identity resolution"],
    )
    info_card(
        "L4 — Review coverage and possible selection bias",
        """
        <p>Reviews are voluntary and some orders may not have reviews,
        meaning the scores represent <b>reviewers rather than every
        customer</b>.</p>
        <p>The code's earlier estimate of ~217 orders without reviews
        compared total order rows with review rows; because review rows
        include repeated reviews for some orders, this estimate should
        be reconciled using distinct order IDs. Confirm with a direct
        anti-join before quoting a final count.</p>
        """,
        variant="info",
        tags=["Selection bias", "Coverage"],
    )
    info_card(
        "L5 — No delivery-distance variable",
        """
        <p>The large geolocation CSV is deliberately excluded.
        Consequently, the current analysis <b>cannot distinguish
        long-distance transit from carrier underperformance</b>.</p>
        <p>Adding validated seller and customer postal-code centroids
        and calculating approximate distance would improve the
        diagnosis.</p>
        """,
        variant="info",
        tags=["Scope", "Distance"],
    )
    info_card(
        "L6 — Dataset and generalisability boundaries",
        """
        <p>The findings concern Olist in Brazil during 2016–2018. They
        may not generalise to other countries, marketplaces, logistics
        networks or current operating conditions.</p>
        <p>The choropleth fetches a public GeoJSON file at runtime, so
        map rendering depends on external availability.</p>
        """,
        variant="info",
        tags=["External validity", "Network dependency"],
    )
    info_card(
        "L7 — Metric and implementation checks",
        """
        <p>Before executive use, reconcile the overall and chart-level
        denominators, validate the state late-rate calculation, and
        confirm that year/state filters behave consistently across all
        visuals.</p>
        <p>Some state scores appear with slightly different rounding in
        different places (for example, RJ is shown as 3.87 in one
        caption and 3.88 in a chart). Establish one canonical query and
        rounding rule.</p>
        """,
        variant="warn",
        tags=["Metric consistency", "Rounding"],
    )

    st.divider()
    st.markdown("### What these limitations mean")
    st.markdown(
        """
        The dashboard is a **strong prioritisation tool**, but it should
        not be treated as a complete root-cause investigation. Its
        safest conclusion is that **lateness and lower satisfaction are
        strongly associated** and that **RJ deserves early operational
        attention**. Root cause, financial impact and causal improvement
        estimates require validation and follow-up analysis.
        """
    )


# ── Recommendations tab ───────────────────────────────────────────────────

def render_recommendations_tab() -> None:
    """Render the seven prioritised recommendations."""
    st.markdown("## Recommendations")
    st.markdown(
        "Each recommendation is tagged with its priority, the evidence "
        "that supports it, and the department best positioned to act. "
        "Priority is graded 🔴 High · 🟡 Medium · 🟢 Strategic."
    )

    info_card(
        "🔴 R1 — Investigate RJ delivery failures (highest ROI)",
        """
        <p><b>Evidence.</b> RJ carries 12,687 orders at a 13.5% late
        rate — 2.3× the 5.9% national average. RJ's average score (3.88)
        sits 0.29 stars below PR (4.18).</p>
        <p><b>Action.</b> Audit RJ carriers, warehouse locations,
        last-mile routes, hand-off delays and performance by
        destination. Run a controlled RJ pilot before scaling.</p>
        <p><b>Lead.</b> Logistics / Operations + Regional Management.</p>
        <p><b>Measures.</b> Late-delivery rate; on-time rate; median and
        90th-percentile transit time; average review score.</p>
        """,
        variant="warn",
        tags=["High priority", "Operations", "RJ"],
    )
    info_card(
        "🔴 R2 — Trigger intervention before day 20",
        """
        <p><b>Evidence.</b> Delivery buckets degrade smoothly:
        4.45 → 4.36 → 4.22 → 3.69 → 2.26. The largest step
        (−0.53 stars) is between "11-20 days" and "21-30 days".</p>
        <p><b>Action.</b> Create ageing-order alerts; escalate at-risk
        parcels; provide revised ETAs; consider expedited routing or
        policy-approved compensation.</p>
        <p><b>Lead.</b> Logistics / Operations + Customer Experience.</p>
        <p><b>Measures.</b> Orders over 20 days; alert-to-resolution
        time; complaint rate; score for delayed orders.</p>
        """,
        variant="warn",
        tags=["High priority", "SLA", "Threshold"],
    )
    info_card(
        "🟡 R3 — Recalibrate estimated delivery dates",
        """
        <p><b>Evidence.</b> When orders are late, they run <b>9.55 days
        past estimate</b> on average, with a worst case of 189 days.
        That is systematic optimism, not random slippage.</p>
        <p><b>Action.</b> Use realised transit history segmented by
        carrier and destination state; back-test predicted versus
        actual dates; refresh the model periodically.</p>
        <p><b>Lead.</b> Data Science / Analytics + Logistics.</p>
        <p><b>Measures.</b> ETA error in days; share delivered by
        promised date; calibration by state/carrier.</p>
        """,
        variant="info",
        tags=["Medium priority", "Forecasting", "Customer comms"],
    )
    info_card(
        "🟡 R4 — Pilot improvement in Bahia (BA)",
        """
        <p><b>Evidence.</b> BA scores 3.86 — the lowest of any state
        with 2,000+ orders — across ~3,340 orders.</p>
        <p><b>Action.</b> Apply the RJ diagnostic checklist to BA; review
        carrier and transit distributions; read low-score comments; test
        changes on a limited scale.</p>
        <p><b>Lead.</b> Regional Operations + Logistics + CX.</p>
        <p><b>Measures.</b> BA average score; late rate; order volume;
        customer complaints before/after.</p>
        """,
        variant="info",
        tags=["Medium priority", "BA", "Pilot"],
    )
    info_card(
        "🟡 R5 — Analyse low-star review text",
        """
        <p><b>Evidence.</b> 1★ reviews are ~11.5% of the total (11,424).
        The free-text <code>review_comment_message</code> column is
        unused.</p>
        <p><b>Action.</b> Cluster 1★/2★ comments into delivery, product
        quality, seller communication, refund and other themes; compare
        RJ and BA.</p>
        <p><b>Lead.</b> Customer Experience Analytics + Data Science.</p>
        <p><b>Measures.</b> Share of low reviews by topic; repeat
        complaint themes; time to close root causes.</p>
        """,
        variant="info",
        tags=["Medium priority", "NLP", "Qualitative"],
    )
    info_card(
        "🟢 R6 — Standardise customer identity for future analytics",
        """
        <p><b>Evidence.</b> 99,441 <code>customer_id</code> rows
        collapse to 96,096 <code>customer_unique_id</code> values — 3.4%
        duplication.</p>
        <p><b>Action.</b> Document <code>customer_unique_id</code> as
        the stable person-level key; create a canonical customer
        dimension; validate joins before retention or lifetime-value
        analysis.</p>
        <p><b>Lead.</b> Data Engineering / Data Governance.</p>
        <p><b>Measures.</b> Duplicate rate; unmatched records;
        reconciliation checks; repeat-customer metric consistency.</p>
        """,
        variant="ok",
        tags=["Strategic", "Data model"],
    )
    info_card(
        "🟢 R7 — Promise reliability, not unsupported speed",
        """
        <p><b>Evidence.</b> On-time orders average 4.29 stars; late
        orders average 2.57. The damage is entirely in the <i>miss</i>,
        not the speed.</p>
        <p><b>Action.</b> Align delivery messaging with realistic
        estimates and operational service levels. Avoid promising a
        date the network cannot reliably meet.</p>
        <p><b>Lead.</b> Marketing / Customer Communications +
        Logistics.</p>
        <p><b>Measures.</b> On-time delivery; ETA accuracy; contacts
        about delivery; review score.</p>
        """,
        variant="ok",
        tags=["Strategic", "Brand", "Positioning"],
    )


# ── Departments tab ───────────────────────────────────────────────────────

def render_departments_tab() -> None:
    """Render the department ownership matrix and governance notes."""
    st.markdown("## Departments — Ownership & Dashboard Hand-off")
    st.markdown(
        "Each department sees exactly which issue they own, which "
        "recommendation resolves it, and which dashboard chart they can "
        "open to watch the metric improve."
    )

    st.markdown(
        """
        <div class="report-body">
        <table>
          <thead>
            <tr>
              <th>Department</th>
              <th>Owns</th>
              <th>Recommendation</th>
              <th>Dashboard chart to monitor</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>Logistics / Operations</b></td>
              <td>RJ late rate (13.5%); nationwide 20-day tail</td>
              <td>R1, R2</td>
              <td>On-Time vs Late by State · Satisfaction by Bucket</td>
            </tr>
            <tr>
              <td><b>Logistics + Customer Experience</b></td>
              <td>Proactive SLA on orders nearing day 20</td>
              <td>R2</td>
              <td>Satisfaction by Delivery-Time Bucket</td>
            </tr>
            <tr>
              <td><b>Data Science + Logistics</b></td>
              <td>EDD model recalibration</td>
              <td>R3</td>
              <td>Monthly Average Review Score</td>
            </tr>
            <tr>
              <td><b>Operations / Regional</b></td>
              <td>BA (3.86) satisfaction</td>
              <td>R4</td>
              <td>Brazil Map (top 10 states)</td>
            </tr>
            <tr>
              <td><b>Data Science / CX Analytics</b></td>
              <td>1★/2★ review-text themes</td>
              <td>R5</td>
              <td>On-Time vs Late by State</td>
            </tr>
            <tr>
              <td><b>Data Engineering</b></td>
              <td>Identity resolution (customer_unique_id)</td>
              <td>R6</td>
              <td>Backend; feeds every tab</td>
            </tr>
            <tr>
              <td><b>Marketing + Logistics</b></td>
              <td>Delivery-certainty positioning</td>
              <td>R7</td>
              <td>Punctuality vs Satisfaction</td>
            </tr>
            <tr>
              <td><b>Executive / Steering</b></td>
              <td>Cross-departmental coordination</td>
              <td>All</td>
              <td>Overview — KPI row + headline</td>
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
        - **Executive sponsor:** Head of Operations (or equivalent), to
          resolve cross-functional blockers and review monthly results.
        - **Operational owner:** Logistics / Operations — accountable
          for the delivery improvement plan and regional pilots.
        - **Customer outcome owner:** Customer Experience — accountable
          for proactive updates, complaint categorisation and service
          recovery.
        - **Analytical support:** Data Science / Analytics — reliable
          metrics, ETA evaluation and review-text analysis.
        - **Data controls:** Data Engineering / Data Governance —
          identity keys, data definitions and join-quality checks.
        - **Commercial alignment:** Marketing — customer-facing promises
          match the delivery capability.
        """
    )

    st.markdown("### Suggested delivery sequence")
    st.markdown(
        """
        - **Days 1–30:** Validate dashboard metrics, build an RJ
          root-cause view, define baseline KPIs.
        - **Days 31–60:** Pilot alerts and operational changes in RJ,
          while investigating BA and review comments.
        - **Days 61–90:** Compare results with the baseline, refine the
          ETA model, and decide which interventions to scale.
        """
    )


# ── Future Research tab ───────────────────────────────────────────────────

def render_research_tab() -> None:
    """Render the eight future-research briefs."""
    st.markdown("## Future Research")
    st.markdown(
        "Open questions and ready-to-pick-up project briefs for the next "
        "analyst. Each entry lists the **question**, the **method**, the "
        "**data** required, and the **expected output**."
    )

    info_card(
        "🔬 F1 — Regress review_score on delivery lateness with controls",
        """
        <p><b>Question.</b> Is the 1.73-star gap causal, or does it
        disappear once we control for distance, product category and
        seller?</p>
        <p><b>Method.</b> OLS with review_score as the dependent
        variable. Regressors: <code>days_late</code> (continuous),
        destination-state fixed effects, product-category fixed effects,
        seller fixed effects, order-value decile. Cluster SEs by
        seller.</p>
        <p><b>Data.</b> All 5 loaded tables, plus optional geolocation
        for a computed great-circle distance.</p>
        <p><b>Output.</b> Coefficient on <code>days_late</code> with a
        confidence interval — the "true" per-day star penalty.</p>
        """,
        variant="info", tags=["Econometrics", "Causal inference", "OLS"],
    )
    info_card(
        "🔬 F2 — Split lateness: seller-caused vs carrier-caused",
        """
        <p><b>Question.</b> Of the 7,826 late deliveries, how many were
        late because the seller handed over late vs late because the
        carrier delivered late?</p>
        <p><b>Method.</b> Define
        <code>seller_delay = order_delivered_carrier_date −
        order_approved_at</code> and
        <code>carrier_delay = order_delivered_customer_date −
        order_delivered_carrier_date</code>. Compare each against a
        state-and-category benchmark.</p>
        <p><b>Data.</b> <code>orders</code> only — the
        <code>order_delivered_carrier_date</code> column is the key.</p>
        <p><b>Output.</b> A two-way attribution table by state.</p>
        """,
        variant="info", tags=["Attribution", "Root cause", "Carrier"],
    )
    info_card(
        "🔬 F3 — Topic-model the review_comment_message corpus",
        """
        <p><b>Question.</b> What are the actual themes inside 1★/2★
        reviews?</p>
        <p><b>Method.</b> BERTopic or LDA over Portuguese comments;
        join labels back to delivery_status and customer_state.</p>
        <p><b>Data.</b> <code>order_reviews</code> +
        <code>orders</code>.</p>
        <p><b>Output.</b> Topic × state heatmap showing which themes
        dominate where.</p>
        """,
        variant="info", tags=["NLP", "Topic modeling", "Portuguese"],
    )
    info_card(
        "🔬 F4 — Model seasonality and the Nov 2017 dip",
        """
        <p><b>Question.</b> Is Nov 2017 a Black-Friday effect, a
        logistics failure, or a data artefact?</p>
        <p><b>Method.</b> STL decomposition of the monthly score
        series; cross-reference residual with order volume and carrier
        performance.</p>
        <p><b>Data.</b> <code>orders</code>, <code>order_items</code>,
        <code>reviews</code>.</p>
        <p><b>Output.</b> Normalised seasonal curve + repeatability
        verdict.</p>
        """,
        variant="info", tags=["Time series", "STL", "Seasonality"],
    )
    info_card(
        "🔬 F5 — Cohort retention using customer_unique_id",
        """
        <p><b>Question.</b> Do customers who experience one late
        delivery come back?</p>
        <p><b>Method.</b> Acquisition-month cohorts keyed on
        <code>customer_unique_id</code>; flag first-order delivery
        status; compare 30/60/90-day return rates.</p>
        <p><b>Data.</b> <code>orders</code> + <code>customers</code> +
        <code>reviews</code>.</p>
        <p><b>Output.</b> Survival curve of repeat purchase by
        first-delivery outcome.</p>
        """,
        variant="info", tags=["Retention", "Cohorts", "LTV"],
    )
    info_card(
        "🔬 F6 — Compute real delivery distance and its star cost",
        """
        <p><b>Question.</b> How much of the 1.73-star gap is explained
        by raw distance?</p>
        <p><b>Method.</b> Load geolocation CSV; aggregate to zip-prefix
        centroid; compute great-circle distance; add as regressor in
        F1.</p>
        <p><b>Data.</b> All 5 tables + geolocation.</p>
        <p><b>Output.</b> Distance-adjusted lateness effect.</p>
        """,
        variant="info", tags=["Geospatial", "Haversine", "Distance"],
    )
    info_card(
        "🔬 F7 — Payment instalments vs expectations",
        """
        <p><b>Question.</b> Do instalment payers tolerate delays
        differently?</p>
        <p><b>Method.</b> Interaction terms in the F1 regression:
        <code>payment_installments × days_late</code>,
        <code>payment_type × days_late</code>.</p>
        <p><b>Data.</b> <code>payments</code> + <code>orders</code> +
        <code>reviews</code>.</p>
        <p><b>Output.</b> Moderator analysis.</p>
        """,
        variant="info", tags=["Moderation", "Payments", "Behaviour"],
    )
    info_card(
        "🔬 F8 — External validation on a second marketplace",
        """
        <p><b>Question.</b> Is 1.73 stars a universal e-commerce effect
        or an Olist-specific number?</p>
        <p><b>Method.</b> Replicate the headline analysis on a public
        non-Olist dataset.</p>
        <p><b>Data.</b> External — beyond the Olist corpus.</p>
        <p><b>Output.</b> A generalisable (or bounded) claim.</p>
        """,
        variant="info", tags=["External validity", "Replication"],
    )


# ── Summary Report tab ────────────────────────────────────────────────────

def render_report_tab(year: str, states: list[str]) -> None:
    """Render the embedded five-page executive summary report."""
    st.markdown("# Customer Satisfaction Analysis — Executive Summary Report")
    st.markdown(
        "**PORA Academy Cohort 7 · Data Analysis · Group 1 Capstone**  \n"
        "*Companion dashboard to the Olist Customer Satisfaction Streamlit "
        "application.*"
    )
    st.markdown(
        f"""
        <div class="callout">
        <b>Active filter:</b> Year = <code>{year}</code> ·
        States = <code>{', '.join(states)}</code>.
        KPI cards above reflect the current sidebar selection. The report
        below presents the unfiltered reference results — use the
        Reference table in the sidebar to cross-check.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page 1 ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>1. Executive Summary</h2>

        <p>This report summarises the findings, limitations and
        recommended operational response from the supplied Customer
        Satisfaction Dashboard. The application analyses Olist
        marketplace orders and customer reviews using Python, pandas,
        SQLite, Matplotlib and Plotly. It presents headline indicators,
        review-score distributions, monthly trends, state comparisons,
        delivery punctuality, and delivery-time buckets.</p>

        <p>The central finding is that <b>delivery punctuality is
        strongly associated with customer satisfaction</b>. The
        dashboard reports an average review score of <b>4.29 for
        on-time deliveries</b> and <b>2.57 for late deliveries</b> — a
        difference of <b>1.73 stars</b>. Overall, it lists 99,441 orders,
        96,478 delivered orders, an average review score of 4.09, 7,826
        late deliveries and average delivery time of 12.56 days. The
        dashboard says late deliveries represent 8.1% of delivered
        orders and are, on average, 9.55 days beyond the estimated
        delivery date.</p>

        <p>The problem is not evenly distributed across Brazil. Rio de
        Janeiro (RJ) is highlighted as the main operational outlier,
        with a reported late-delivery rate of <b>13.5%</b>, compared
        with <b>5.9% nationally</b>, across 12,687 orders. Bahia (BA) is
        also flagged for a low average review score of approximately
        <b>3.86</b>. These patterns support prioritising regional
        logistics diagnostics rather than applying an undifferentiated
        nationwide intervention.</p>

        <h3>Key findings at a glance</h3>
        <table>
          <thead>
            <tr>
              <th>Indicator</th><th>Dashboard result</th>
              <th>Implication</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>Total orders</td><td>99,441</td>
                <td>Large enough dataset to identify operational patterns.</td></tr>
            <tr><td>Delivered orders</td><td>96,478</td>
                <td>Delivery performance is the main operational focus.</td></tr>
            <tr><td>Average review score</td><td>4.09 / 5</td>
                <td>Useful baseline for tracking improvement.</td></tr>
            <tr><td>Late deliveries</td><td>7,826 (8.1%)</td>
                <td>A minority of deliveries create a disproportionate satisfaction concern.</td></tr>
            <tr><td>On-time vs late score</td><td>4.29 vs 2.57</td>
                <td>Late delivery is strongly associated with lower ratings.</td></tr>
            <tr><td>RJ late rate</td><td>13.5% vs 5.9% national</td>
                <td>Prioritise a regional root-cause investigation.</td></tr>
            <tr><td>Delivery time &gt;20 days</td>
                <td>Average ratings fall below 4.09</td>
                <td>Create alerts and recovery actions before delays become prolonged.</td></tr>
          </tbody>
        </table>

        <h3>Purpose and scope</h3>
        <p>The purpose is to translate the dashboard outputs into
        practical decisions, identify which departments should own each
        response, and make the evidence boundaries explicit. The report
        uses the supplied code's stated results; it does not
        independently rerun the underlying CSV files. Therefore, numeric
        findings should be treated as dashboard-reported results pending
        data-quality validation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page 2 ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>2. Findings and Business Interpretation</h2>

        <h3>2.1 Delivery punctuality is the priority</h3>
        <p>The dashboard's most important comparison is <b>4.29</b>
        average stars for orders delivered on time versus <b>2.57</b>
        for late orders. This 1.73-star difference is a strong
        <i>association</i> in the observed data, but it is not proof
        that lateness alone causes the entire difference. Product
        quality, seller performance, destination distance and
        communication failures may contribute.</p>

        <p>The delivery-time bucket analysis shows a progressive
        decline in average scores: <b>4.45</b> for 0–5 days,
        <b>4.36</b> for 6–10 days, <b>4.22</b> for 11–20 days,
        <b>3.69</b> for 21–30 days, and <b>2.26</b> for the longest
        bucket. This supports an early-warning threshold before an
        order reaches 20 days in transit.</p>

        <h3>2.2 Regional performance varies</h3>
        <p>The state-level analysis identifies BA and RJ as
        low-scoring states, while Paraná (PR) and São Paulo (SP) are
        among the stronger performers. RJ's late rate of <b>13.5%</b>
        is about 2.3 times the stated national rate of <b>5.9%</b>. BA,
        with a reported score of 3.86 and approximately 3,340 orders,
        is proposed as a second, smaller pilot market.</p>

        <h3>2.3 Trends and review distribution</h3>
        <p>The review distribution is bimodal: <b>57.8%</b> of reviews
        are five-star and <b>11.5%</b> are one-star. The dashboard also
        calls attention to <b>November 2017</b>, when average monthly
        review score dips. November 2016 has no orders — that absence
        should not be treated as a value to impute.</p>

        <h3>2.4 Overall interpretation</h3>
        <p>The best-supported management response is to improve
        <b>reliability</b>, <b>accuracy of delivery promises</b> and
        <b>communication during delays</b>. Customer Experience should
        support Operations by notifying customers early, explaining
        revised delivery estimates and applying a consistent
        service-recovery policy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page 3 ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>3. Recommendations and Departmental Ownership</h2>
        <p>The actions below translate the code's recommendations into
        an implementation plan. The departments are proposed operational
        owners based on the responsibilities implied by the dashboard.</p>

        <table>
          <thead>
            <tr>
              <th>Priority / action</th>
              <th>Lead department(s)</th>
              <th>Specific actions</th>
              <th>Measures to monitor</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1. Investigate RJ delivery failures (High)</td>
              <td>Logistics / Operations; Regional Management</td>
              <td>Audit carriers, warehouse locations, last-mile routes, hand-off delays and performance by destination.</td>
              <td>Late-delivery rate; on-time rate; median and 90th-percentile transit time; average review score.</td>
            </tr>
            <tr>
              <td>2. Trigger intervention before day 20 (High)</td>
              <td>Logistics / Operations + Customer Experience</td>
              <td>Create ageing-order alerts; escalate at-risk parcels; provide revised ETAs.</td>
              <td>Orders over 20 days; alert-to-resolution time; complaint rate.</td>
            </tr>
            <tr>
              <td>3. Recalibrate estimated delivery dates (Medium)</td>
              <td>Data Science / Analytics + Logistics</td>
              <td>Use realised transit history segmented by carrier and destination state.</td>
              <td>ETA error in days; share delivered by promised date.</td>
            </tr>
            <tr>
              <td>4. Pilot improvement in Bahia (Medium)</td>
              <td>Regional Operations + Logistics + CX</td>
              <td>Apply the RJ diagnostic checklist to BA.</td>
              <td>BA average score; late rate; complaint volume.</td>
            </tr>
            <tr>
              <td>5. Analyse low-star review text (Medium)</td>
              <td>Customer Experience Analytics + Data Science</td>
              <td>Cluster 1- and 2-star comments into recurring themes.</td>
              <td>Share of low reviews by topic.</td>
            </tr>
            <tr>
              <td>6. Standardise customer identity (Strategic)</td>
              <td>Data Engineering / Data Governance</td>
              <td>Document customer_unique_id as the stable person-level key.</td>
              <td>Duplicate rate; reconciliation checks.</td>
            </tr>
            <tr>
              <td>7. Promise reliability, not speed (Strategic)</td>
              <td>Marketing / Customer Communications + Logistics</td>
              <td>Align delivery messaging with realistic estimates.</td>
              <td>On-time delivery; ETA accuracy; review score.</td>
            </tr>
          </tbody>
        </table>

        <h3>Governance and accountability</h3>
        <p><b>Executive sponsor:</b> Head of Operations. <b>Operational
        owner:</b> Logistics / Operations. <b>Customer outcome owner:</b>
        Customer Experience. <b>Analytical support:</b> Data Science /
        Analytics. <b>Data controls:</b> Data Engineering / Data
        Governance. <b>Commercial alignment:</b> Marketing.</p>

        <h3>Suggested delivery sequence</h3>
        <p><b>First 30 days:</b> validate dashboard metrics, build an RJ
        root-cause view, define baseline KPIs. <b>Days 31–60:</b> pilot
        alerts and operational changes in RJ. <b>Days 61–90:</b> compare
        results with the baseline, refine the ETA model, decide which
        interventions to scale.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page 4 ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>4. Limitations, Data Risks and Interpretation Boundaries</h2>

        <h3>4.1 Observational results do not establish causality</h3>
        <p>The 1.73-star difference is <b>correlation, not
        causation</b>. Product category, distance, seller quality and
        other unmeasured variables may affect both delivery timing and
        ratings. Use a controlled pilot or a regression model that
        controls for relevant factors before claiming that a specific
        intervention will increase ratings by a precise amount.</p>

        <h3>4.2 Incomplete time coverage</h3>
        <p>2016 contains only September, October and December, with
        <b>no November orders</b>, while 2018 ends on <b>17 October</b>.
        <b>2017 is the only complete calendar year</b> in this
        dataset.</p>

        <h3>4.3 Review-row fan-out and customer identity</h3>
        <p><b>99,224 review rows across 98,673 distinct orders</b> — a
        551-row difference. The dashboard protects order counts by
        using <code>COUNT(DISTINCT order_id)</code>. The customers table
        also has 99,441 rows but <b>96,096 distinct
        customer_unique_id values</b>.</p>

        <h3>4.4 Review coverage and possible selection bias</h3>
        <p>Reviews are voluntary; scores represent <b>reviewers rather
        than every customer</b>. Confirm order-to-review coverage with a
        direct anti-join before quoting a final count.</p>

        <h3>4.5 No delivery-distance variable</h3>
        <p>The geolocation CSV is excluded. The analysis <b>cannot
        distinguish long-distance transit from carrier
        underperformance</b>.</p>

        <h3>4.6 Dataset and generalisability boundaries</h3>
        <p>Findings concern Olist in Brazil during 2016–2018. The
        choropleth fetches a public GeoJSON at runtime, so map rendering
        depends on external availability.</p>

        <h3>4.7 Metric and implementation checks</h3>
        <p>Before executive use, reconcile overall and chart-level
        denominators, validate the state late-rate calculation, and
        confirm year/state filters behave consistently across all
        visuals. Establish one canonical query and rounding rule.</p>

        <h3>What these limitations mean</h3>
        <p>The dashboard is a strong <b>prioritisation tool</b>, but it
        should not be treated as a complete root-cause investigation.
        Its safest conclusion is that <b>lateness and lower
        satisfaction are strongly associated</b> and that <b>RJ
        deserves early operational attention</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page 5 ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>5. Implementation, Monitoring and Conclusion</h2>

        <h3>5.1 Recommended monitoring scorecard</h3>
        <table>
          <thead>
            <tr>
              <th>KPI</th><th>Definition / use</th>
              <th>Suggested owner</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>On-time delivery rate</td>
                <td>Delivered orders delivered on or before estimated date, divided by eligible delivered orders.</td>
                <td>Logistics / Operations</td></tr>
            <tr><td>Late-delivery rate</td>
                <td>Late delivered orders divided by eligible delivered orders; report nationally and by state/carrier.</td>
                <td>Operations Analytics</td></tr>
            <tr><td>Average review score</td>
                <td>Average review score with review-grain and duplicate handling documented.</td>
                <td>CX Analytics</td></tr>
            <tr><td>Score gap by punctuality</td>
                <td>Average score for on-time minus late deliveries; trend over time and by state.</td>
                <td>CX Analytics / Data Science</td></tr>
            <tr><td>Long-delay share</td>
                <td>Share of delivered orders taking more than 20 days from purchase to delivery.</td>
                <td>Logistics</td></tr>
            <tr><td>ETA error</td>
                <td>Actual delivery date minus estimated date; track average and distribution.</td>
                <td>Data Science / Logistics</td></tr>
            <tr><td>Low-star complaint themes</td>
                <td>Proportion of 1–2 star comments assigned to delivery, product, communication, refund or other topics.</td>
                <td>Customer Experience</td></tr>
            <tr><td>Data quality checks</td>
                <td>Distinct orders, duplicate review rows, missing review coverage, date completeness, join reconciliation.</td>
                <td>Data Engineering</td></tr>
          </tbody>
        </table>

        <h3>5.2 Further analysis recommended by the code</h3>
        <ul>
          <li>Estimate the lateness effect with regression controls for
              seller, product category, destination state and order
              value; cluster standard errors by seller.</li>
          <li>Separate seller-side delay from carrier-side delay using
              approval, carrier hand-off and customer-delivery
              timestamps.</li>
          <li>Topic-model Portuguese review comments and compare
              complaint themes by state and delivery status.</li>
          <li>Investigate November 2017 with seasonal decomposition and
              order-volume / carrier-performance comparisons.</li>
          <li>Study repeat purchase using <code>customer_unique_id</code>
              and compare customers whose first delivery was late with
              those whose first delivery was on time.</li>
          <li>Add approximate delivery distance and test whether
              distance explains part of the observed lateness
              association.</li>
          <li>Test whether payment type or installment count changes the
              relationship between lateness and review scores.</li>
          <li>Replicate the analysis on a second marketplace before
              generalising the 1.73-star difference.</li>
        </ul>

        <h3>5.3 Decision requested</h3>
        <p>Approve a <b>90-day, evidence-led delivery reliability
        initiative</b>, starting with an RJ diagnostic and a day-20
        escalation process. Assign <b>Logistics / Operations</b> as the
        accountable lead, with <b>Customer Experience</b>, <b>Data
        Science</b>, <b>Data Engineering</b>, <b>Regional
        Management</b> and <b>Marketing</b> as supporting teams. Use
        <b>BA</b> as a smaller comparison or pilot market where
        appropriate. Do not present projected score improvements as
        guaranteed outcomes; measure changes against a validated
        baseline.</p>

        <h3>Conclusion</h3>
        <p>The supplied dashboard indicates that <b>delivery
        reliability is the clearest actionable correlate of customer
        satisfaction</b> in the Olist data. On-time deliveries average
        4.29 stars, while late deliveries average 2.57. RJ has a
        substantially higher reported late-delivery rate than the
        national rate, and satisfaction declines sharply when delivery
        exceeds 20 days.</p>

        <p>The recommended response combines <b>regional logistics
        audits</b>, <b>proactive customer communication</b>,
        <b>realistic delivery estimates</b>, <b>review-text
        analysis</b> and <b>stronger data governance</b>. The key
        limitation is that the dashboard identifies patterns, not
        definitive causes.</p>

        <p style="font-size: 13px; color: #6B7280; margin-top: 24px;">
        <i>Source basis:</i> User-supplied Streamlit dashboard code
        (Olist Customer Satisfaction, PORA Academy Cohort 7, Group 1).
        All numerical findings are reported as stated in the code and
        should be validated against the source CSV files before external
        publication.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# §8  BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Application entry point."""
    _apply_global_style()

    # ── Data layer ────────────────────────────────────────────────────
    try:
        conn = build_connection()
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.exception("Data load failed")
        st.error(f"❌ **Data load failed.** {exc}")
        st.stop()

    # ── Sidebar + filter state ────────────────────────────────────────
    year, states      = render_sidebar()
    state_json        = safe_state_json(states)
    filter_active     = year != "All" or "All" not in states

    # ── Title + headline ──────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="
            background: linear-gradient(90deg, #6D3FB5 0%, #8B5CF6 50%, #D64545 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 42px; font-weight: 800;
            letter-spacing: -0.02em; text-align: center;
            margin-bottom: 0; padding-bottom: 0;">
          Customer Satisfaction Dashboard
        </h1>
        <p style='color:#6B7280; font-size:14px;
                  margin-top:4px; text-align:center;'>
          PORA Academy Cohort 7 — Group 1 Capstone Project
        </p>
        """,
        unsafe_allow_html=True,
    )
    render_projection_headline()

    # ── Tabs ──────────────────────────────────────────────────────────
    (
        tab_overview, tab_charts, tab_limitations,
        tab_recommendations, tab_departments, tab_research, tab_report,
    ) = st.tabs([
        "🏠 Overview",
        "📊 Key Charts",
        "⚠️ Limitations",
        "✅ Recommendations",
        "🏢 Departments",
        "🔬 Future Research",
        "📄 Summary Report",
    ])

    with tab_overview:
        render_overview_tab(conn, year, state_json, filter_active)
    with tab_charts:
        render_charts_tab(conn, year, state_json)
    with tab_limitations:
        render_limitations_tab()
    with tab_recommendations:
        render_recommendations_tab()
    with tab_departments:
        render_departments_tab()
    with tab_research:
        render_research_tab()
    with tab_report:
        render_report_tab(year, states)

    # ── Footer ────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "**Reference (filters = All):** Total Orders 99,441 · Delivered "
        "96,478 · Avg Review 4.09 · Late 7,826 · Avg Delivery Days "
        "12.56. Both Year and State filters apply to every KPI and every "
        "chart on the **Overview** and **Key Charts** tabs. Chart 1 "
        "(Review Score Distribution) is intentionally unfiltered — a "
        "whole-dataset fact."
    )


if __name__ == "__main__":
    main()