# olist_app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis · Group 1 Capstone

Decision-making dashboard for customer satisfaction.
Every KPI and chart responds to sidebar filters (Year + State).

Tabs:
    Overview          — KPI row · Delivery histogram · Review distribution
    Satisfaction      — Score by state · Brazil map · Monthly trend
    Delivery          — On-time vs late · Histogram · Buckets · Worst states
    Detail & Method   — Data model, definitions, DQ checks, raw records
    Recommendations   — Priority cards with owners + measures
    Departments       — Ownership table
    Executive Report  — Narrative + CSV/TXT download
    Future Research   — Next analytical steps

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

import io
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Palette:
    """Modern, accessible palette for decision dashboards."""
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
    geojson_timeout_seconds: Final[int] = 20

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


@st.cache_resource(
    ttl=CONFIG.data_cache_ttl_seconds, show_spinner="Loading Olist data…"
)
def build_connection() -> sqlite3.Connection:
    """Load CSVs into in-memory SQLite (Cloud-safe, no disk writes)."""
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
    _LOGGER.info("In-memory DB built with %d tables", len(frames))
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
        _LOGGER.warning("safe_state_json fallback: %s", exc)
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
        _LOGGER.warning("%s failed: %s", label, exc)
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
# §4  PLOTLY STYLING
# ════════════════════════════════════════════════════════════════════════════

def plotly_style(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply a consistent look to every Plotly chart."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=16, r=16, t=56, b=16),
        title=dict(
            font=dict(size=15, color=PALETTE.slate, family="Inter, Segoe UI, sans-serif"),
            x=0.02, xanchor="left",
        ),
        font=dict(
            family="Inter, Segoe UI, system-ui, sans-serif",
            size=12, color=PALETTE.text,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0, frameon=False,
        ),
        hoverlabel=dict(font_size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False, linecolor="#E2E8F0")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# §5  KPI COMPUTATION
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
# §6  UI HELPERS
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
    """Render an actionable insight under a chart."""
    md = f"**Selection:** {scope_label(year, state_json)}  \n**Insight.** {text}"
    {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
    }.get(level, st.info)(md)


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


def show_table(df: pd.DataFrame, *, height: int | None = None) -> None:
    st.dataframe(
        df, use_container_width=True, hide_index=True, height=height,
    )


# ════════════════════════════════════════════════════════════════════════════
# §7  PRESENTATION LAYER
# ════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS: Final[str] = """
<style>
  .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
  }
  h1, h2, h3, h4 { letter-spacing: -0.01em; line-height: 1.25; }
  h1 { margin-bottom: 0.25rem !important; }
  h2 { margin-top: 1.5rem !important; margin-bottom: 0.75rem !important; }
  h3 {
    font-size: 1.05rem !important; font-weight: 600 !important;
    margin-top: 1rem !important; margin-bottom: 0.5rem !important;
    color: #1E293B;
  }
  [data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-left: 4px solid #2563EB; border-radius: 8px;
    padding: 16px 20px; box-shadow: 0 1px 3px rgba(15,23,42,0.06);
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.78rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.05em;
    color: #64748B !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 1.75rem !important; font-weight: 700 !important;
    color: #0F172A !important; line-height: 1.1;
  }
  [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
  button[data-baseweb="tab"] {
    font-weight: 600; font-size: 0.95rem; padding: 0.6rem 1rem;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
    border-bottom-width: 3px !important;
  }
  [data-testid="stAlert"] {
    border-radius: 8px; padding: 0.85rem 1.1rem;
    font-size: 0.92rem; line-height: 1.55;
  }
  .scope-pill {
    display: inline-block; background: #E0E7FF; color: #3730A3;
    border-radius: 999px; padding: 0.35rem 0.9rem;
    font-size: 0.82rem; font-weight: 600; margin: 0.5rem 0 1rem 0;
  }
  .info-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-left: 4px solid #2563EB; border-radius: 8px;
    padding: 1.1rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
  }
  .info-card.warn { border-left-color: #EF4444; }
  .info-card.ok   { border-left-color: #10B981; }
  .info-card.info { border-left-color: #0D9488; }
  .info-card h4 {
    color: #0F172A; margin: 0 0 0.6rem 0;
    font-size: 1rem; font-weight: 700; line-height: 1.35;
  }
  .info-card.warn h4 { color: #B91C1C; }
  .info-card.ok h4   { color: #047857; }
  .info-card.info h4 { color: #0F766E; }
  .info-card p, .info-card li {
    color: #334155; line-height: 1.6;
    margin-bottom: 0.5rem; font-size: 0.92rem;
  }
  .info-card p:last-child { margin-bottom: 0; }
  .tag-row { margin-bottom: 0.6rem; }
  .info-card .tag {
    display: inline-block; background: #F1F5F9; color: #475569;
    border-radius: 6px; padding: 0.2rem 0.65rem;
    font-size: 0.72rem; font-weight: 600;
    margin-right: 0.35rem; margin-bottom: 0.25rem;
    letter-spacing: 0.02em;
  }
  hr { margin: 1.5rem 0 !important; }
  section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
  }
</style>
"""


def _apply_global_style() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_sidebar() -> tuple[str, list[str]]:
    st.sidebar.title("Filters")
    st.sidebar.caption(
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
# §8  CHART FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

# ── Delivery timing histogram ─────────────────────────────────────────────

def chart_delivery_histogram(
    conn: sqlite3.Connection, year: str, state_json: str, *, height: int = 400
) -> None:
    df = safe_query(conn, f"""
        SELECT julianday(o.order_delivered_customer_date)
             - julianday(o.order_estimated_delivery_date) AS days_vs_estimate
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {DELIVERED_SQL} AND {FILTER_SQL}
    """, params=year_multi(year, state_json), label="delivery timing")

    if df.empty:
        _no_data_notice()
        return

    fig = px.histogram(
        df, x="days_vs_estimate", nbins=45,
        title="Distribution of delivery timing (days vs estimate)",
        labels={"days_vs_estimate": "Days (negative = early, positive = late)"},
        color_discrete_sequence=[PALETTE.blue],
    )
    fig.add_vline(
        x=0, line_dash="dash", line_color=PALETTE.red,
        annotation_text="Estimated date",
        annotation_position="top right",
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    st.plotly_chart(plotly_style(fig, height), use_container_width=True)

    late = int((df["days_vs_estimate"] > 0).sum())
    total = len(df)
    late_pct = 100 * late / total if total else 0
    median = float(df["days_vs_estimate"].median())
    _insight(
        year, state_json,
        f"{late_pct:.1f}% of delivered orders arrived late. "
        f"Median timing: {median:+.1f} days vs estimate.",
        level="warning" if late_pct > 10 else "info",
    )


# ── Review score distribution ─────────────────────────────────────────────

def chart_review_distribution(
    conn: sqlite3.Connection, year: str, state_json: str, *, height: int = 400
) -> None:
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
    colors = {
        1: PALETTE.red, 2: PALETTE.red_light, 3: PALETTE.grey,
        4: PALETTE.blue_light, 5: PALETTE.green,
    }

    fig = px.bar(
        df, x="score", y="reviews",
        title="Review Score Distribution",
        labels={"score": "Review score (stars)", "reviews": "Number of reviews"},
        color="score", color_discrete_map=colors, text="pct",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside",
        marker_line_width=0,
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(plotly_style(fig, height), use_container_width=True)

    low_share = float(df.loc[df["score"] <= 2, "pct"].sum())
    share5 = float(df.loc[df["score"] == 5, "pct"].iloc[0]) if (df["score"] == 5).any() else 0
    _insight(
        year, state_json,
        f"{share5:.1f}% of {total:,} reviews are 5★. "
        f"Low scores (1–2★) make up {low_share:.1f}%.",
        level="warning" if low_share > 15 else "info",
    )


# ── Score by state ────────────────────────────────────────────────────────

def chart_state_scores(
    conn: sqlite3.Connection, year: str, state_json: str, *, height: int = 400
) -> None:
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
    fig = px.bar(
        plot_df, x="avg_score", y="state", orientation="h",
        title="Average Review Score by State (top 10 by orders)",
        labels={"avg_score": "Avg review score (1–5)", "state": "State",
                "orders": "Orders"},
        color="avg_score",
        color_continuous_scale=["#FCA5A5", "#FCD34D", "#6EE7B7"],
        text="avg_score",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_xaxes(range=[0, 5])
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(plotly_style(fig, height), use_container_width=True)

    low, high = plot_df.iloc[0], plot_df.iloc[-1]
    spread = float(high["avg_score"] - low["avg_score"])
    _insight(
        year, state_json,
        f"{low['state']} ({low['avg_score']:.2f}) is lowest, "
        f"{high['state']} ({high['avg_score']:.2f}) highest — "
        f"a spread of {spread:.2f} stars.",
        level="warning" if low["avg_score"] < 4.0 else "info",
    )


# ── Brazil map ────────────────────────────────────────────────────────────

@st.cache_data(ttl=CONFIG.geojson_cache_ttl_seconds, show_spinner=False)
def load_brazil_geojson() -> dict[str, Any]:
    if os.path.isfile(CONFIG.geojson_local_path):
        try:
            with open(CONFIG.geojson_local_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            _LOGGER.warning("Local GeoJSON failed: %s", exc)

    last_err: Exception | None = None
    for url in CONFIG.geojson_urls:
        try:
            r = requests.get(url, timeout=CONFIG.geojson_timeout_seconds)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            _LOGGER.warning("GeoJSON mirror failed: %s — %s", url, exc)
            last_err = exc
    raise RuntimeError(f"All GeoJSON sources failed. Last error: {last_err}")


@st.cache_data(ttl=CONFIG.geojson_cache_ttl_seconds, show_spinner=False)
def _geojson_state_lookup() -> pd.DataFrame:
    geojson = load_brazil_geojson()
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        sigla = (
            props.get("sigla") or props.get("SIGLA")
            or props.get("abbrev") or props.get("code")
        )
        if sigla:
            rows.append({"state": str(sigla).upper()})
    return pd.DataFrame(rows)


def chart_brazil_map(
    conn: sqlite3.Connection, year: str, state_json: str, *, height: int = 400
) -> None:
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

    try:
        geojson = load_brazil_geojson()
    except RuntimeError as exc:
        _LOGGER.error("GeoJSON unavailable: %s", exc)
        st.warning(
            "⚠️ Brazil map is temporarily unavailable. "
            "Add `data/brazil-states.geojson` to the repo, "
            "or verify internet access."
        )
        return

    selected = selected_states(state_json)
    filter_active = len(selected) > 0

    plot_df = df.copy()
    if filter_active:
        plot_df["in_scope"] = plot_df["state"].isin(selected)
    else:
        plot_df["in_scope"] = True

    fig = px.choropleth(
        plot_df,
        geojson=geojson,
        locations="state",
        featureidkey="properties.sigla",
        color="delivered_orders",
        color_continuous_scale="YlGnBu",
        title=f"Delivered Orders by State — {scope_label(year, state_json)}",
        labels={"delivered_orders": "Delivered orders"},
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="white",
    )
    fig.update_layout(
        geo=dict(scope="south america"),
        coloraxis_colorbar=dict(title="Orders", thickness=12),
    )
    st.plotly_chart(plotly_style(fig, height), use_container_width=True)

    in_scope = plot_df[plot_df["in_scope"]]
    if in_scope.empty:
        _insight(year, state_json, "No states match the current filter.")
        return
    lo = in_scope.loc[in_scope["delivered_orders"].idxmin()]
    hi = in_scope.loc[in_scope["delivered_orders"].idxmax()]
    _insight(
        year, state_json,
        f"{lo['state']} has the fewest ({int(lo['delivered_orders']):,}) "
        f"and {hi['state']} the most ({int(hi['delivered_orders']):,}) "
        "delivered orders in scope.",
    )


# ── Monthly trend ─────────────────────────────────────────────────────────

def chart_monthly_score(
    conn: sqlite3.Connection, year: str, state_json: str, *, height: int = 400
) -> None:
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

    fig = px.line(
        df, x="month", y="avg_score", markers=True,
        title="Monthly Average Review Score",
        labels={"month": "Month", "avg_score": "Avg review score (1–5)",
                "reviews": "Reviews"},
        hover_data=["reviews"],
    )
    fig.update_traces(line_color=PALETTE.teal, line_width=2.5, marker_size=8)
    lo = max(1.0, float(df["avg_score"].min()) - 0.2)
    hi = min(5.0, float(df["avg_score"].max()) + 0.2)
    fig.update_yaxes(range=[lo, hi])
    st.plotly_chart(plotly_style(fig, height), use_container_width=True)

    low_row  = df.loc[df["avg_score"].idxmin()]
    high_row = df.loc[df["avg_score"].idxmax()]
    _insight(
        year, state_json,
        f"Lowest: {low_row['month']} ({low_row['avg_score']:.2f}); "
        f"highest: {high_row['month']} ({high_row['avg_score']:.2f}).",
    )


# ── On-time vs late ───────────────────────────────────────────────────────

def chart_ontime_vs_late(
    conn: sqlite3.Connection, year: str, state_json: str, *, height: int = 400
) -> None:
    df = safe_query(conn, f"""
        SELECT CASE WHEN o.order_delivered_customer_date
                         <= o.order_estimated_delivery_date
                    THEN 'On Time' ELSE 'Late' END AS status,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score,
               ROUND(AVG(julianday(o.order_delivered_customer_date)
                       - julianday(o.order_purchase_timestamp)), 2)
                   AS avg_delivery_days
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE {DELIVERED_SQL} AND {FILTER_SQL}
        GROUP BY 1
    """, params=year_multi(year, state_json), label="on-time vs late")

    if df.empty:
        _no_data_notice()
        return

    color_map = {"On Time": PALETTE.green, "Late": PALETTE.red}
    fig = px.bar(
        df, x="status", y="avg_score",
        title="Delivery Punctuality vs Satisfaction",
        labels={"status": "", "avg_score": "Avg review score (1–5)",
                "orders": "Orders", "avg_delivery_days": "Avg delivery days"},
        color="status", color_discrete_map=color_map,
        text="avg_score", hover_data=["orders", "avg_delivery_days"],
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_yaxes(range=[1, 5.4])
    st.plotly_chart(plotly_style(fig, height), use_container_width=True)

    scores = dict(zip(df["status"], df["avg_score"]))
    if {"On Time", "Late"} <= scores.keys():
        gap = float(scores["On Time"] - scores["Late"])
        _insight(
            year, state_json,
            f"Late deliveries cost **{gap:.2f} stars** — "
            f"{scores['On Time']:.2f} on time vs {scores['Late']:.2f} late.",
            level="error" if gap >= 1.5 else "warning",
        )
    else:
        _insight(year, state_json,
                 "Only one punctuality class exists in this selection.")


# ── Delivery buckets ──────────────────────────────────────────────────────

def chart_delivery_buckets(
    conn: sqlite3.Connection, year: str, state_json: str, *, height: int = 400
) -> None:
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
                 ELSE '30+ days' END AS bucket,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)), 3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE {DELIVERED_SQL} AND {FILTER_SQL}
        GROUP BY 1
        ORDER BY CASE bucket
                   WHEN '0-5 days' THEN 1 WHEN '6-10 days' THEN 2
                   WHEN '11-20 days' THEN 3 WHEN '21-30 days' THEN 4
                   ELSE 5 END
    """, params=year_multi(year, state_json), label="delivery buckets")

    if df.empty:
        _no_data_notice()
        return

    palette_by_bucket = {
        "0-5 days": PALETTE.green_light, "6-10 days": PALETTE.green,
        "11-20 days": PALETTE.yellow, "21-30 days": PALETTE.orange,
        "30+ days": PALETTE.red,
    }
    fig = px.bar(
        df, x="bucket", y="avg_score",
        title="Satisfaction by Delivery-Time Bucket",
        labels={"bucket": "Days from purchase to delivery",
                "avg_score": "Avg review score (1–5)", "orders": "Orders"},
        color="bucket", color_discrete_map=palette_by_bucket,
        text="avg_score", hover_data=["orders"],
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_yaxes(range=[1, 5.4])
    st.plotly_chart(plotly_style(fig, height), use_container_width=True)

    last = df.iloc[-1]
    _insight(
        year, state_json,
        f"{last['bucket']} orders average {last['avg_score']:.2f} stars — "
        f"{'a severe' if last['avg_score'] < 3.0 else 'a notable'} drop "
        "from the fastest bucket.",
        level="warning" if last["avg_score"] < 3.5 else "info",
    )


# ── Worst states by late rate ─────────────────────────────────────────────

def chart_worst_states_late_rate(
    conn: sqlite3.Connection, year: str, state_json: str, *, height: int = 400
) -> None:
    df = safe_query(conn, f"""
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS delivered,
               SUM(CASE WHEN o.order_delivered_customer_date
                            > o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS late
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {DELIVERED_SQL} AND {FILTER_SQL}
        GROUP BY state
        HAVING delivered >= 100
        ORDER BY (1.0 * late / delivered) DESC
        LIMIT 10
    """, params=year_multi(year, state_json), label="worst states")

    if df.empty:
        _no_data_notice("Not enough delivered orders per state.")
        return

    df["late_rate_pct"] = 100.0 * df["late"] / df["delivered"]

    fig = px.bar(
        df, x="late_rate_pct", y="state", orientation="h",
        title="Worst 10 States by Late-Delivery Rate",
        labels={"late_rate_pct": "Late-delivery rate (%)", "state": "State",
                "delivered": "Delivered orders", "late": "Late orders"},
        color="late_rate_pct",
        color_continuous_scale=["#FCD34D", "#FCA5A5", "#EF4444"],
        text="late_rate_pct",
        hover_data=["delivered", "late"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(plotly_style(fig, height), use_container_width=True)

    worst = df.iloc[0]
    _insight(
        year, state_json,
        f"{worst['state']} has the highest late rate at "
        f"{worst['late_rate_pct']:.1f}% "
        f"({int(worst['late']):,} of {int(worst['delivered']):,} orders).",
        level="warning" if worst["late_rate_pct"] > 10 else "info",
    )


# ════════════════════════════════════════════════════════════════════════════
# §9  TABS
# ════════════════════════════════════════════════════════════════════════════

# ── Tab 1: Overview ───────────────────────────────────────────────────────

def render_overview_tab(
    conn: sqlite3.Connection, year: str, state_json: str,
    filter_active: bool,
) -> None:
    st.markdown("## Overview")

    k = query_kpis(conn, year, state_json)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders", fmt_int(k["total_orders"]))
    c2.metric("Delivered Orders", fmt_int(k["delivered_orders"]))

    avg_score = k["avg_review_score"]
    if avg_score is None:
        c3.metric("Avg Review Score", "—")
    else:
        delta = (f"{avg_score - 4.09:+.2f} vs reference"
                 if filter_active else None)
        c3.metric("Avg Review Score", fmt_float(avg_score), delta=delta)

    late_val = k["late_deliveries"]
    delivered_val = k["delivered_orders"]
    late_delta = (f"{100 * late_val / delivered_val:.1f}% of delivered"
                  if delivered_val and delivered_val > 0 else None)
    c4.metric("Late Deliveries", fmt_int(late_val),
              delta=late_delta, delta_color="inverse")
    c5.metric("Avg Delivery Days", fmt_float(k["avg_delivery_days"]))

    if filter_active:
        st.info(
            f"Active filter — **{scope_label(year, state_json)}**. "
            "Set both filters to All to reproduce the sidebar reference values."
        )

    st.divider()

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown("### Delivery Timing Distribution")
        chart_delivery_histogram(conn, year, state_json)
    with col_right:
        st.markdown("### Review Score Distribution")
        chart_review_distribution(conn, year, state_json)


# ── Tab 2: Satisfaction ───────────────────────────────────────────────────

def render_satisfaction_tab(
    conn: sqlite3.Connection, year: str, state_json: str,
) -> None:
    st.markdown("## Customer Satisfaction")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("### Score by State")
        chart_state_scores(conn, year, state_json)
    with col_b:
        st.markdown("### Brazil Map · Delivered Orders")
        chart_brazil_map(conn, year, state_json)

    st.divider()
    st.markdown("### Monthly Trend")
    chart_monthly_score(conn, year, state_json, height=380)


# ── Tab 3: Delivery ───────────────────────────────────────────────────────

def render_delivery_tab(
    conn: sqlite3.Connection, year: str, state_json: str,
) -> None:
    st.markdown("## Delivery Performance")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("### On-Time vs Late · Satisfaction Impact")
        chart_ontime_vs_late(conn, year, state_json)
    with col_b:
        st.markdown("### Delivery Timing Distribution")
        chart_delivery_histogram(conn, year, state_json)

    st.divider()

    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        st.markdown("### Satisfaction by Delivery Bucket")
        chart_delivery_buckets(conn, year, state_json)
    with col_d:
        st.markdown("### Worst 10 States by Late Rate")
        chart_worst_states_late_rate(conn, year, state_json)


# ── Tab 4: Detail & Method ────────────────────────────────────────────────

def render_detail_tab(
    conn: sqlite3.Connection, year: str, state_json: str,
) -> None:
    st.markdown("## Detail & Method")
    st.markdown(
        "**Research question:** How are satisfaction and delivery indicators "
        "measured, and what data-quality issues affect interpretation?"
    )

    st.markdown("### Data model")
    st.markdown(
        "The dashboard uses **orders as the analysis grain**. Reviews, "
        "payments, and order items are aggregated to one row per order before "
        "they are joined to the orders table. This prevents one-to-many joins "
        "from inflating order counts or duplicating review scores."
    )

    st.markdown("### Definitions")
    definitions = pd.DataFrame([
        {"Indicator": "Customer satisfaction",
         "Definition": "Mean review_score, on a 1–5 scale, after aggregating "
                       "review records to order level."},
        {"Indicator": "Late delivery",
         "Definition": "Actual customer delivery date is later than the "
                       "estimated delivery date."},
        {"Indicator": "On time / early",
         "Definition": "Actual customer delivery date is on or before the "
                       "estimated delivery date."},
        {"Indicator": "Late-delivery rate",
         "Definition": "Late delivered orders divided by delivered orders "
                       "with classifiable actual and estimated delivery dates."},
        {"Indicator": "Days past estimate",
         "Definition": "Actual delivery date minus estimated delivery date, "
                       "in days; positive means late."},
        {"Indicator": "Order count",
         "Definition": "Distinct order_id, not raw rows from a one-to-many "
                       "table."},
    ])
    show_table(definitions)

    st.markdown("### Data quality checks")
    dq = safe_query(conn, f"""
        SELECT 'orders' AS tbl, COUNT(*) AS rows,
               COUNT(DISTINCT o.order_id) AS distinct_orders,
               SUM(CASE WHEN o.order_id IS NULL THEN 1 ELSE 0 END) AS null_ids
        FROM orders o JOIN customers c ON o.customer_id = c.customer_id
        WHERE {FILTER_SQL}
        UNION ALL
        SELECT 'reviews', COUNT(*), COUNT(DISTINCT r.order_id),
               SUM(CASE WHEN r.order_id IS NULL THEN 1 ELSE 0 END)
        FROM reviews r JOIN orders o ON r.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE {FILTER_SQL}
    """, params=year_multi(year, state_json) * 2, label="DQ checks")
    if not dq.empty:
        show_table(dq)
    else:
        st.info("No data-quality rows for the current filters.")

    st.warning(
        "**Limitations.** These are observational marketplace data; "
        "associations do not establish causation. Some records may lack review "
        "scores or delivery timestamps. Multiple review rows can exist per "
        "order, so raw review-row counts are not equivalent to order counts. "
        "State-level comparisons may also be affected by different order volumes."
    )

    with st.expander("Inspect order-level records (first 500 in scope)"):
        records = safe_query(conn, f"""
            SELECT o.order_id, o.order_status,
                   strftime('%Y', o.order_purchase_timestamp) AS purchase_year,
                   c.customer_state,
                   r.review_score,
                   CASE WHEN o.order_delivered_customer_date
                             <= o.order_estimated_delivery_date
                        THEN 'On Time' ELSE 'Late' END AS delivery_status,
                   ROUND(julianday(o.order_delivered_customer_date)
                       - julianday(o.order_estimated_delivery_date), 1)
                       AS days_vs_estimate,
                   ROUND(julianday(o.order_delivered_customer_date)
                       - julianday(o.order_purchase_timestamp), 1)
                       AS delivery_days
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            LEFT JOIN reviews r ON o.order_id = r.order_id
            WHERE {DELIVERED_SQL} AND {FILTER_SQL}
            LIMIT 500
        """, params=year_multi(year, state_json), label="order records")
        if not records.empty:
            show_table(records)
            st.caption("Showing at most the first 500 delivered orders in scope.")
        else:
            st.info("No delivered orders for the current filters.")


# ── Tab 5: Recommendations ────────────────────────────────────────────────

def render_recommendations_tab() -> None:
    st.markdown("## Recommendations")
    st.caption(
        "**These actions do not respond to the sidebar filters.** They are "
        "derived from the whole dataset, because a strategic decision should "
        "not flip simply because someone narrowed the view."
    )

    st.divider()
    st.markdown("### Priority actions")

    info_card(
        "🔴 R1 — Improve delivery reliability in worst-performing states",
        """<p><b>Evidence.</b> RJ carries 12,687 orders at a 13.5% late rate
        — 2.3× the national average. BA scores lowest (3.86) of any state
        with 2,000+ orders.</p>
        <p><b>Action.</b> Audit carriers, warehouse locations, last-mile
        routes and hand-off delays in RJ and BA. Run a controlled pilot
        before scaling.</p>
        <p><b>Lead.</b> Logistics / Operations + Regional Management.</p>
        <p><b>Measures.</b> Late-delivery rate; on-time rate; median and 90th
        percentile transit time; average review score.</p>""",
        variant="warn", tags=["High priority", "Operations", "RJ", "BA"],
    )
    info_card(
        "🔴 R2 — Trigger intervention before day 20",
        """<p><b>Evidence.</b> Delivery buckets decline 4.45 → 4.36 → 4.22 →
        3.69 → 2.26. The largest single step (−0.53 stars) is between
        11–20 days and 21–30 days.</p>
        <p><b>Action.</b> Ageing-order alerts; escalate at-risk parcels;
        proactive revised ETAs.</p>
        <p><b>Lead.</b> Logistics / Operations + Customer Experience.</p>
        <p><b>Measures.</b> Orders over 20 days; alert-to-resolution time;
        complaint rate.</p>""",
        variant="warn", tags=["High priority", "SLA", "Day 20"],
    )
    info_card(
        "🟡 R3 — Recalibrate estimated delivery dates",
        """<p><b>Evidence.</b> Late orders run 9.55 days past estimate on
        average — systematic optimism, not random slippage.</p>
        <p><b>Action.</b> Fit ETA from realised transit history by carrier
        and destination state; back-test predicted vs actual.</p>
        <p><b>Lead.</b> Data Science / Analytics + Logistics.</p>
        <p><b>Measures.</b> ETA error (days); % delivered by promised date.</p>""",
        variant="info", tags=["Medium priority", "Forecasting"],
    )
    info_card(
        "🟡 R4 — Proactively communicate delays",
        """<p><b>Evidence.</b> On-time = 4.29; late = 2.57. The damage is in
        the <i>miss</i>, not the speed.</p>
        <p><b>Action.</b> Notify customers when an order is at risk of
        arriving late; provide a revised estimate and clear support route.</p>
        <p><b>Lead.</b> Customer Experience / Support.</p>
        <p><b>Measures.</b> Contacts about delivery; response/resolution
        time; review score.</p>""",
        variant="info", tags=["Medium priority", "CX"],
    )
    info_card(
        "🟢 R5 — Review low-score orders systematically",
        """<p><b>Evidence.</b> 1–2★ reviews are ~14% of all reviews; the
        association with late delivery is strongest in 21+ day orders.</p>
        <p><b>Action.</b> Create a routine for analysing 1–2★ reviews
        alongside delivery delays, product issues, and seller performance.</p>
        <p><b>Lead.</b> Customer Experience / Seller Operations.</p>
        <p><b>Measures.</b> 1–2★ review share; root-cause categories per
        month.</p>""",
        variant="ok", tags=["Ongoing", "CX", "Sellers"],
    )


# ── Tab 6: Departments ────────────────────────────────────────────────────

def render_departments_tab() -> None:
    st.markdown("## Department ownership and performance measures")

    dept = pd.DataFrame([
        {"Department": "Logistics / Delivery Operations",
         "Responsibility": "Reduce late deliveries and shorten delays beyond estimate.",
         "Suggested KPI": "Late-delivery rate; average days past estimate."},
        {"Department": "Seller Operations",
         "Responsibility": "Work with sellers whose fulfilment patterns link to poor delivery or low ratings.",
         "Suggested KPI": "Seller-level late rate; low-score order share."},
        {"Department": "Customer Experience / Support",
         "Responsibility": "Communicate delivery exceptions; close the loop on dissatisfied customers.",
         "Suggested KPI": "1–2★ review share; response and resolution time."},
        {"Department": "Data / Business Intelligence",
         "Responsibility": "Maintain validated order-level data and consistent dashboard definitions.",
         "Suggested KPI": "Data completeness; refresh success; metric reconciliation."},
        {"Department": "Regional Operations",
         "Responsibility": "Investigate geographic differences in punctuality and satisfaction.",
         "Suggested KPI": "State-level late rate and average review score, with order volume."},
        {"Department": "Performance Management",
         "Responsibility": "Track improvement over time and hold owners accountable.",
         "Suggested KPI": "Monthly movement in late rate and on-time/late score gap."},
    ])
    show_table(dept)
    st.caption(
        "Department assignments are recommendations for operational "
        "ownership, not claims about Olist's internal organisational structure."
    )


# ── Tab 7: Executive Report ───────────────────────────────────────────────

def render_executive_tab(
    conn: sqlite3.Connection, year: str, state_json: str,
) -> None:
    st.markdown("## Executive Report")

    stats = safe_query(conn, f"""
        SELECT COUNT(DISTINCT o.order_id) AS total_orders,
               COUNT(DISTINCT CASE WHEN o.order_status='delivered'
                                    THEN o.order_id END) AS delivered_orders,
               SUM(CASE WHEN o.order_delivered_customer_date
                            > o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS late_orders,
               ROUND(AVG(CASE WHEN o.order_delivered_customer_date
                                   <= o.order_estimated_delivery_date
                              THEN CAST(r.review_score AS REAL) END), 2)
                   AS on_time_score,
               ROUND(AVG(CASE WHEN o.order_delivered_customer_date
                                   > o.order_estimated_delivery_date
                              THEN CAST(r.review_score AS REAL) END), 2)
                   AS late_score,
               ROUND(AVG(julianday(o.order_delivered_customer_date)
                       - julianday(o.order_estimated_delivery_date)), 2)
                   AS avg_days_vs_estimate
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        WHERE {FILTER_SQL}
    """, params=year_multi(year, state_json), label="exec report")

    if stats.empty:
        st.info("No data for the current filters.")
        return

    total_orders   = safe_scalar(stats, "total_orders", 0)
    delivered      = safe_scalar(stats, "delivered_orders", 0)
    late_orders    = safe_scalar(stats, "late_orders", 0)
    on_time_score  = safe_scalar(stats, "on_time_score")
    late_score     = safe_scalar(stats, "late_score")
    avg_days_est   = safe_scalar(stats, "avg_days_vs_estimate")
    late_pct = 100 * late_orders / delivered if delivered else None
    gap = (on_time_score - late_score
           if on_time_score is not None and late_score is not None else None)

    st.markdown("### Key finding")
    if gap is not None:
        st.markdown(
            f"Within **{scope_label(year, state_json)}**, delivered orders "
            f"arriving on time or early averaged **{on_time_score:.2f}/5**, "
            f"while late delivered orders averaged **{late_score:.2f}/5**. "
            f"The difference was **{gap:.2f} stars**. "
            f"**{late_pct:.1f}%** of delivered orders arrived late."
        )
    else:
        st.info(
            "The selected scope does not contain enough delivery and review "
            "data to compute the main comparison."
        )

    st.markdown("### Supporting facts")
    st.markdown(
        f"1. Delivered orders in scope: **{fmt_int(delivered)}** "
        f"(of {fmt_int(total_orders)} total).\n"
        f"2. Late delivered orders: **{fmt_int(late_orders)}** — "
        f"average timing vs estimate: **{fmt_float(avg_days_est)} days**.\n"
        f"3. Mean review-score gap between on-time/early and late "
        f"deliveries: **{fmt_float(gap)} stars**."
    )

    st.markdown("### Recommended management response")
    st.markdown(
        "Prioritise delivery-exception prevention and customer communication, "
        "then monitor punctuality and satisfaction together each month. "
        "Segment follow-up by state, seller, and product category to identify "
        "where operational attention is most likely to help."
    )

    st.markdown("### Caveat")
    st.markdown(
        "The analysis identifies an **association, not proof of causation**. "
        "Results depend on the selected filters and on the completeness of "
        "recorded delivery dates and review scores."
    )

    st.divider()

    # Downloadable summary
    summary = pd.DataFrame([
        {"KPI": "Scope", "Value": scope_label(year, state_json)},
        {"KPI": "Total orders", "Value": total_orders},
        {"KPI": "Delivered orders", "Value": delivered},
        {"KPI": "Late orders", "Value": late_orders},
        {"KPI": "Late-delivery rate (%)",
         "Value": round(late_pct, 1) if late_pct is not None else None},
        {"KPI": "Avg on-time review score",
         "Value": round(on_time_score, 2) if on_time_score is not None else None},
        {"KPI": "Avg late review score",
         "Value": round(late_score, 2) if late_score is not None else None},
        {"KPI": "Score gap (stars)",
         "Value": round(gap, 2) if gap is not None else None},
        {"KPI": "Avg days vs estimate",
         "Value": round(avg_days_est, 2) if avg_days_est is not None else None},
    ])
    show_table(summary)

    col1, col2 = st.columns(2)
    with col1:
        csv_buf = io.StringIO()
        summary.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Download summary (.csv)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="olist_executive_summary.csv",
            mime="text/csv",
        )
    with col2:
        report_txt = (
            "Olist Customer Satisfaction Executive Report\n"
            f"Scope: {scope_label(year, state_json)}\n\n"
            f"Total orders: {fmt_int(total_orders)}\n"
            f"Delivered orders: {fmt_int(delivered)}\n"
            f"Late orders: {fmt_int(late_orders)}\n"
            f"Late-delivery rate: "
            f"{fmt_float(late_pct)}%\n"
            f"Avg on-time review score: {fmt_float(on_time_score)}/5\n"
            f"Avg late review score: {fmt_float(late_score)}/5\n"
            f"Score gap: {fmt_float(gap)} stars\n"
            f"Avg days vs estimate: {fmt_float(avg_days_est)} days\n\n"
            "Interpretation: descriptive association only; "
            "this analysis does not establish causation.\n"
        )
        st.download_button(
            "📄 Download narrative (.txt)",
            data=report_txt.encode("utf-8"),
            file_name="olist_executive_summary.txt",
            mime="text/plain",
        )


# ── Tab 8: Future Research ────────────────────────────────────────────────

def render_future_tab() -> None:
    st.markdown("## Future research")
    st.markdown(
        "To move beyond descriptive association, future work could:"
    )
    st.markdown("""
- **Model predictors of low satisfaction:** use ordinal or logistic
  regression with delivery timing, freight cost, product category, seller,
  payment type, and geography.
- **Control for confounding:** compare late and on-time orders with similar
  characteristics, and report uncertainty intervals.
- **Analyse text reviews:** classify review comments into delivery, product
  quality, customer service, and other themes.
- **Study seller and regional variation:** account for differing order
  volumes and avoid over-interpreting small groups.
- **Evaluate interventions:** use a pre/post or controlled design to assess
  whether proactive delay notifications improve ratings.
- **Check robustness:** compare mean review scores with median scores and
  the proportion of 1–2★ reviews.
""")


# ════════════════════════════════════════════════════════════════════════════
# §10  BOOTSTRAP
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

    st.markdown(
        "<h1 style='margin-bottom:0.15rem;'>"
        "Customer Satisfaction Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.caption("PORA Academy Cohort 7 — Group 1 Capstone Project")
    st.markdown(
        f'<span class="scope-pill">Scope: {scope_label(year, state_json)}</span>',
        unsafe_allow_html=True,
    )

    (
        tab_overview, tab_satisfaction, tab_delivery, tab_detail,
        tab_recs, tab_dept, tab_exec, tab_future,
    ) = st.tabs([
        "Overview",
        "Satisfaction",
        "Delivery",
        "Detail & Method",
        "Recommendations",
        "Departments",
        "Executive Report",
        "Future Research",
    ])

    with tab_overview:
        render_overview_tab(conn, year, state_json, filter_active)
    with tab_satisfaction:
        render_satisfaction_tab(conn, year, state_json)
    with tab_delivery:
        render_delivery_tab(conn, year, state_json)
    with tab_detail:
        render_detail_tab(conn, year, state_json)
    with tab_recs:
        render_recommendations_tab()
    with tab_dept:
        render_departments_tab()
    with tab_exec:
        render_executive_tab(conn, year, state_json)
    with tab_future:
        render_future_tab()

    st.divider()
    st.caption(
        "**Reference (filters = All):** Total Orders 99,441 · Delivered "
        "96,478 · Avg Review 4.09 · Late 7,826 · Avg Delivery Days 12.56. "
        "Every KPI, chart and insight responds to the Year and State "
        "filters and reconciles to the verified README tables when both "
        "are set to All."
    )


if __name__ == "__main__":
    main()