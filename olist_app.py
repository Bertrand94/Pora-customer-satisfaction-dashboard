# olist_app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis · Group 1 Capstone

Professional, decision-first dashboard.

The 5 README-required charts (4 required + 1 advanced), each on the tab
the README specifies:
    ✅ Q2  Review score distribution (bar) — WHOLE DATASET (not filterable)
    ✅ Q3  Monthly average review score (line)
    ✅ Q4  Average review score by state (horiz bar, top 10)
    ✅ Q5  On-time vs late delivery by state (grouped bar, top 5)
    ✅ Q6  Delivery punctuality vs satisfaction (ADVANCED — required)

Tabs (README "shape of your app"):
    Overview               — KPIs + headline + Q2 (headline chart)
    Customer Satisfaction  — Q3, Q4, Q5 (+ supporting map)
    Detail & Method        — Q6 (advanced) + limitations + diagnostics
    Recommendations        — decision actions (whole dataset)

Reference (filters = All):
    Total Orders 99,441 · Delivered 96,478 · Avg Review 4.09
    Late 7,826 · Avg Delivery Days 12.56 · Headline gap 1.72★

Run:      streamlit run olist_app.py
Cloud:    Streamlit Community Cloud — commit `data/` folder or set OLIST_DATA_DIR.
Requires: streamlit, pandas, matplotlib, requests, scipy, plotly
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
import tempfile
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

CHART_FIGSIZE:      Final[tuple[float, float]] = (8.6, 4.8)
CHART_FIGSIZE_WIDE: Final[tuple[float, float]] = (13.5, 4.4)
PARTIAL_YEARS:      Final[frozenset[str]] = frozenset({"2016", "2018"})
MIN_REVIEWS_MONTH:  Final[int] = 30
PUNCT_MIN_N_TIER:   Final[int] = 30
PUNCT_SCENARIOS:    Final[tuple[int, ...]] = (5, 10, 25, 50)

REF: Final[dict[str, float]] = {
    "total_orders":      99_441,
    "delivered_orders":  96_478,
    "avg_review_score":   4.09,
    "late_deliveries":    7_826,
    "avg_delivery_days": 12.56,
    "on_time_score":      4.29,
    "late_score":         2.57,
    "star_gap":           1.72,
}


@dataclass(frozen=True)
class Palette:
    primary:       str = "#4F46E5"
    primary_light: str = "#A5B4FC"
    primary_dark:  str = "#3730A3"
    accent:        str = "#F97316"
    accent_light:  str = "#FDBA74"
    accent_dark:   str = "#C2410C"

    success:       str = "#10B981"
    success_light: str = "#6EE7B7"
    warn:          str = "#F59E0B"
    warn_light:    str = "#FCD34D"
    danger:        str = "#E11D48"
    danger_light:  str = "#FDA4AF"

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
    required_columns: Final[dict[str, tuple[str, ...]]] = field(
        default_factory=lambda: {
            "orders": (
                "order_id", "customer_id", "order_status",
                "order_purchase_timestamp", "order_approved_at",
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
    date_columns: Final[tuple[str, ...]] = (
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "review_creation_date",
        "review_answer_timestamp", "shipping_limit_date",
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
# §3  DATA LAYER (with data_cleaning.ipynb pipeline + tempfile fallback)
# ════════════════════════════════════════════════════════════════════════════

_HERE: Final[str] = os.path.dirname(os.path.abspath(__file__))


def _data_dir() -> str:
    return os.environ.get("OLIST_DATA_DIR", os.path.join(_HERE, "data"))


def _resolve_db_path() -> str:
    custom = os.environ.get("OLIST_DB_PATH")
    if custom:
        return custom
    for candidate in (
        os.path.join(_HERE, "olist.db"),
        os.path.join(tempfile.gettempdir(), "olist.db"),
    ):
        parent = os.path.dirname(candidate) or "."
        if os.access(parent, os.W_OK):
            return candidate
    return os.path.join(tempfile.gettempdir(), "olist.db")


def _clean_dataframe(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of data_cleaning.ipynb — strip, parse dates, uppercase states."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        df[col] = df[col].replace("", pd.NA)
    for col in CONFIG.date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "customer_state" in df.columns:
        df["customer_state"] = df["customer_state"].str.upper()
    if "seller_state" in df.columns:
        df["seller_state"] = df["seller_state"].str.upper()
    if table == "reviews" and "review_score" in df.columns:
        df["review_score"] = pd.to_numeric(
            df["review_score"], errors="coerce"
        )
    _LOGGER.info("Cleaned %s — %d rows", table, len(df))
    return df


@st.cache_resource(ttl=CONFIG.data_cache_ttl_seconds,
                   show_spinner="Loading Olist data…")
def _build_connection() -> sqlite3.Connection:
    data_dir = _data_dir()
    db_path = _resolve_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    for table, filename in CONFIG.tables.items():
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing CSV for {table!r} at {path}. "
                "Set OLIST_DATA_DIR or commit data/ to the repo."
            )
        df = _clean_dataframe(table, pd.read_csv(path))
        df.to_sql(table, conn, if_exists="replace", index=False)
    _LOGGER.info("DB ready at %s", db_path)
    return conn


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds,
               show_spinner="Building analysis model…")
def _build_model(_db_path_token: str) -> pd.DataFrame:
    """One row per order with joins for customers + averaged review score.

    Passing `_db_path_token` (underscore prefix) tells Streamlit's cache
    not to hash the argument, but it still invalidates on explicit token
    change if we need to.
    """
    conn = _build_connection()
    orders = pd.read_sql(
        "SELECT order_id, customer_id, order_status, "
        "       order_purchase_timestamp, "
        "       order_delivered_customer_date, "
        "       order_estimated_delivery_date "
        "FROM orders", conn,
    )
    customers = pd.read_sql(
        "SELECT customer_id, customer_state FROM customers", conn,
    )
    # Avg review_score per order — fan-out safe
    reviews = pd.read_sql(
        "SELECT order_id, AVG(review_score) AS review_score "
        "FROM reviews GROUP BY order_id", conn,
    )

    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(reviews, on="order_id", how="left")

    for c in (
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["year"]  = df["order_purchase_timestamp"].dt.year.astype("Int64").astype(str)
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    df["is_delivered"] = (
        df["order_status"].eq("delivered")
        & df["order_delivered_customer_date"].notna()
    )
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["days_late"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days
    df["is_late"] = df["is_delivered"] & (df["days_late"] > 0)

    return df


def get_model() -> pd.DataFrame:
    return _build_model(_resolve_db_path())


def apply_filters(df: pd.DataFrame, year: str,
                  states: Sequence[str]) -> pd.DataFrame:
    out = df
    if year != "All":
        out = out[out["year"] == year]
    if states:
        out = out[out["customer_state"].isin(states)]
    return out


def scope_label(year: str, states: Sequence[str]) -> str:
    y = "All years" if year == "All" else year
    s = ", ".join(states) if states else "All states"
    return f"{y} · {s}"


def is_partial_year(year: str) -> bool:
    return year in PARTIAL_YEARS


# ════════════════════════════════════════════════════════════════════════════
# §4  KPI COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds, show_spinner=False)
def query_kpis(year: str, states: tuple[str, ...]) -> dict[str, Any]:
    df = apply_filters(get_model(), year, list(states))
    delivered = df[df["is_delivered"]]
    avg_review = (df["review_score"].mean()
                  if df["review_score"].notna().any() else None)
    avg_days = (delivered["delivery_days"].mean()
                if delivered["delivery_days"].notna().any() else None)
    return {
        "total_orders":      int(df["order_id"].nunique()),
        "delivered_orders":  int(delivered["order_id"].nunique()),
        "avg_review_score":  float(avg_review) if avg_review is not None else None,
        "late_deliveries":   int(delivered["is_late"].sum()),
        "avg_delivery_days": float(avg_days) if avg_days is not None else None,
    }


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds, show_spinner=False)
def query_headline_gap(year: str, states: tuple[str, ...]) -> dict[str, Any]:
    df = apply_filters(get_model(), year, list(states))
    d = df[df["is_delivered"] & df["review_score"].notna()].copy()
    d["status"] = d["is_late"].map({True: "Late", False: "On Time"})
    grp = d.groupby("status")["review_score"].mean()
    on_time = float(grp.get("On Time")) if "On Time" in grp else None
    late    = float(grp.get("Late"))    if "Late"    in grp else None
    gap     = (on_time - late) if (on_time is not None and late is not None) else None
    return {"on_time": on_time, "late": late, "gap": gap}


def fmt_int(v: Any) -> str:
    return "—" if v is None or pd.isna(v) else f"{int(v):,}"


def fmt_float(v: Any, p: int = 2) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.{p}f}"


# ════════════════════════════════════════════════════════════════════════════
# §5  STREAMLIT API SHIMS (width deprecation safe)
# ════════════════════════════════════════════════════════════════════════════

def _st_pyplot(fig: plt.Figure) -> None:
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)


def _st_dataframe(df: pd.DataFrame, *, hide_index: bool = True) -> None:
    try:
        st.dataframe(df, hide_index=hide_index, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=hide_index, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# §6  LOGO
# ════════════════════════════════════════════════════════════════════════════

def _get_logo_base64() -> str:
    local_path = os.path.join(_HERE, "pora_logo.png")
    if os.path.isfile(local_path):
        try:
            with open(local_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
                return f"data:image/png;base64,{data}"
        except OSError as e:
            _LOGGER.warning("Logo failed: %s", e)
    # Fallback: minimal inline SVG
    return (
        "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5v"
        "cmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIg"
        "c3Ryb2tlPSIjNEY0NkU1IiBzdHJva2Utd2lkdGg9IjIiPjxjaXJjbGUgY3g9"
        "IjEyIiBjeT0iMTIiIHI9IjEwIi8+PHBhdGggZD0iTTEyIDh2OE04IDEyaDgi"
        "Lz48L3N2Zz4="
    )


# ════════════════════════════════════════════════════════════════════════════
# §7  CHART HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _no_data(msg: str = "No data for the current filters.") -> None:
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-icon">📭</div>
          <div class="empty-title">Nothing to show</div>
          <div class="empty-body">{msg}</div>
          <div class="empty-hint">💡 Try widening the Year or State filters.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _insight(year: str, states: Sequence[str], text: str) -> None:
    st.markdown(
        f"""
        <div class="insight">
          <span class="insight-chip">💡 Insight</span>
          <div class="insight-body">
            <span class="insight-scope">{scope_label(year, states)}</span>
            <span class="insight-text">{text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color=PALETTE.slate_200, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=PALETTE.slate_500, labelsize=10)


def _finish(fig: plt.Figure) -> None:
    try:
        fig.tight_layout()
    except Exception:
        pass
    _st_pyplot(fig)


def _partial_note(year: str) -> None:
    if not is_partial_year(year):
        return
    detail = {
        "2016": "2016 has only 3 active months (Sep, Oct, Dec) — "
                "November has zero orders.",
        "2018": "2018 ends 17 Oct — not a full calendar year.",
    }[year]
    st.caption(f"⚠️ **Partial year — {year}.** {detail}")


def _safe_chart(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        _LOGGER.exception("Chart %s failed", fn.__name__)
        st.warning(f"⚠️ This chart could not render: {exc}")


def _chart_card(fn, *args, **kwargs) -> None:
    with st.container(border=True):
        _safe_chart(fn, *args, **kwargs)


# ════════════════════════════════════════════════════════════════════════════
# §8  CHARTS — Q2 (whole dataset per README)
# ════════════════════════════════════════════════════════════════════════════

def chart_review_distribution(year: str, states: tuple[str, ...]) -> None:
    """Q2 — README: 'not filterable — whole-dataset fact'.

    Reproduces: 1: 11,424 (11.5%) · 2: 3,151 (3.2%) · 3: 8,179 (8.2%)
                4: 19,142 (19.3%) · 5: 57,328 (57.8%)
    """
    st.markdown("### Review Score Distribution")
    st.caption(
        "📌 **Whole-dataset fact (README Q2).** This chart does not respond "
        "to filters — it reproduces the verified benchmark exactly."
    )

    conn = _build_connection()
    df = pd.read_sql(
        "SELECT CAST(review_score AS INTEGER) AS score, "
        "       COUNT(*) AS reviews, "
        "       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM reviews), 1) AS pct "
        "FROM reviews GROUP BY 1 ORDER BY 1",
        conn,
    )
    if df.empty:
        _no_data()
        return

    total = int(df["reviews"].sum())
    shades = [
        PALETTE.success if s == 5
        else PALETTE.danger if s == 1
        else PALETTE.slate_400
        for s in df["score"]
    ]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    bars = ax.bar(df["score"].astype(str), df["reviews"], color=shades,
                  edgecolor=PALETTE.white, linewidth=1.4)
    ax.set_title("Review Score Distribution", pad=14, loc="left")
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Number of reviews")
    ax.set_ylim(0, df["reviews"].max() * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    _style_axes(ax)
    for b, n, pct in zip(bars, df["reviews"], df["pct"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{int(n):,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10,
                color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    p5 = float(df.loc[df["score"] == 5, "pct"].iloc[0])
    p1 = float(df.loc[df["score"] == 1, "pct"].iloc[0])
    _insight(year, states,
             f"{p5:.1f}% of {total:,} reviews are 5★; {p1:.1f}% are 1★.")


# ════════════════════════════════════════════════════════════════════════════
# §9  CHARTS — Q3 (monthly trend)
# ════════════════════════════════════════════════════════════════════════════

def chart_monthly_score(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Monthly Average Review Score")
    st.caption(
        "📌 Watch for the **Nov 2017 dip** — the largest single monthly drop "
        "in the dataset. Gap in Nov 2016 is real (zero orders)."
    )
    df = apply_filters(get_model(), year, list(states))
    df = df[df["review_score"].notna() & df["month"].notna()]
    if df.empty:
        _no_data()
        return

    grp = (
        df.groupby("month")
          .agg(reviews=("order_id", "nunique"),
               avg_score=("review_score", "mean"))
          .reset_index().sort_values("month")
    )
    if grp.empty:
        _no_data()
        return

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    ax.plot(grp["month"], grp["avg_score"], marker="o",
            color=PALETTE.primary, linewidth=2.4, markersize=6,
            markerfacecolor=PALETTE.white, markeredgewidth=1.8)
    ax.set_title("Monthly Average Review Score", pad=14, loc="left")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg review score (1–5)")

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

    low_i = grp["avg_score"].idxmin()
    high_i = grp["avg_score"].idxmax()
    ax.scatter([grp.loc[low_i, "month"]], [grp.loc[low_i, "avg_score"]],
               s=170, color=PALETTE.danger, zorder=5,
               edgecolor=PALETTE.white, linewidth=1.8)
    ax.annotate(
        f"Lowest\n{grp.loc[low_i,'month']}\n{grp.loc[low_i,'avg_score']:.2f}",
        xy=(grp.loc[low_i, "month"], grp.loc[low_i, "avg_score"]),
        xytext=(0, -46), textcoords="offset points", ha="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE.slate_500, lw=1.2),
        fontsize=10, color=PALETTE.danger, fontweight="bold",
    )
    _finish(fig)

    noisy = int((grp["reviews"] < MIN_REVIEWS_MONTH).sum())
    note = (
        f" {noisy} month(s) have < {MIN_REVIEWS_MONTH} reviews — noisy averages."
    ) if noisy else ""
    _insight(year, states,
             f"Lowest {grp.loc[low_i,'month']} ({grp.loc[low_i,'avg_score']:.2f}); "
             f"highest {grp.loc[high_i,'month']} ({grp.loc[high_i,'avg_score']:.2f}).{note}")
    _partial_note(year)


# ════════════════════════════════════════════════════════════════════════════
# §10  CHARTS — Q4 (state scores)
# ════════════════════════════════════════════════════════════════════════════

def chart_state_scores(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Average Review Score by State *(top 10)*")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["review_score"].notna() & df["customer_state"].notna()]
    if df.empty:
        _no_data()
        return

    grp = (
        df.groupby("customer_state")
          .agg(orders=("order_id", "nunique"),
               avg_score=("review_score", "mean"))
          .reset_index()
          .sort_values("orders", ascending=False)
          .head(10)
    )
    plot_df = grp.sort_values("avg_score", ascending=True)
    colors = [
        PALETTE.danger if s < 4.00
        else PALETTE.warn if s < 4.10
        else PALETTE.success
        for s in plot_df["avg_score"]
    ]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.barh(plot_df["customer_state"], plot_df["avg_score"],
                   color=colors, edgecolor=PALETTE.white, linewidth=1.4)
    ax.set_title("Average Review Score by State (top 10 by orders)",
                 pad=14, loc="left")
    ax.set_xlabel("Average review score (1–5)")
    ax.set_xlim(0, 5.4)
    _style_axes(ax)
    ax.grid(axis="x", color=PALETTE.slate_200, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    for b, s, n in zip(bars, plot_df["avg_score"], plot_df["orders"]):
        ax.text(s + 0.04, b.get_y() + b.get_height() / 2,
                f"{s:.2f}  (n={int(n):,})", va="center", fontsize=10,
                color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    low, high = plot_df.iloc[0], plot_df.iloc[-1]
    if len(plot_df) == 1:
        text = f"{low['customer_state']} averages {low['avg_score']:.2f}★."
    else:
        text = (f"{low['customer_state']} ({low['avg_score']:.2f}) anchors "
                f"the low end; {high['customer_state']} "
                f"({high['avg_score']:.2f}) the high end — spread "
                f"{high['avg_score'] - low['avg_score']:.2f}★.")
    _insight(year, states, text)


# ════════════════════════════════════════════════════════════════════════════
# §11  CHARTS — Q5 (on-time vs late by state)
# ════════════════════════════════════════════════════════════════════════════

def chart_ontime_by_state(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### On-Time vs Late by State *(top 5)*")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["is_delivered"] & df["customer_state"].notna()]
    if df.empty:
        _no_data()
        return

    grp = (
        df.groupby("customer_state")
          .agg(delivered=("order_id", "nunique"),
               late=("is_late", "sum"))
          .reset_index()
          .sort_values("delivered", ascending=False)
          .head(5)
    )
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
    ax.set_xticklabels(grp["customer_state"])
    ax.set_title("On-Time vs Late Deliveries by State (top 5)",
                 pad=14, loc="left")
    ax.set_xlabel("State")
    ax.set_ylabel("Delivered orders")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    _finish(fig)

    rate = 100.0 * grp["late"] / grp["delivered"]
    worst = grp.loc[rate.idxmax()]
    best  = grp.loc[rate.idxmin()]
    overall = 100.0 * grp["late"].sum() / grp["delivered"].sum()
    _insight(year, states,
             f"Across top-5 states, {overall:.1f}% delivered late. "
             f"{worst['customer_state']} highest "
             f"({100 * worst['late'] / worst['delivered']:.1f}%); "
             f"{best['customer_state']} lowest "
             f"({100 * best['late'] / best['delivered']:.1f}%).")


# ════════════════════════════════════════════════════════════════════════════
# §12  CHART — Q6 ADVANCED (binary on-time vs late — README verified)
# ════════════════════════════════════════════════════════════════════════════

def _welch_ci(a: pd.Series, b: pd.Series, alpha: float = 0.05
              ) -> tuple[float, float, float, float, float]:
    """Return diff, ci_low, ci_high, t_stat, cohens_d."""
    import numpy as np
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (float(a.mean() - b.mean()) if na and nb else float("nan"),
                float("nan"), float("nan"), float("nan"), float("nan"))
    diff = float(a.mean() - b.mean())
    try:
        from scipy import stats
        va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
        se = float(np.sqrt(va / na + vb / nb))
        denom = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        df_w = (va / na + vb / nb) ** 2 / denom if denom > 0 else float("nan")
        tcrit = float(stats.t.ppf(1 - alpha / 2, df_w)) if df_w == df_w else float("nan")
        ci_lo, ci_hi = diff - tcrit * se, diff + tcrit * se
        t_stat, _ = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        pooled = float(np.sqrt(
            ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
        )) if (na + nb - 2) > 0 else float("nan")
        d = diff / pooled if pooled and pooled > 0 else float("nan")
        return diff, ci_lo, ci_hi, float(t_stat), float(d)
    except ImportError:
        return diff, float("nan"), float("nan"), float("nan"), float("nan")


def _cohens_verdict(d: float) -> str:
    if d != d:
        return "n/a"
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def chart_punctuality_vs_satisfaction(year: str,
                                      states: tuple[str, ...]) -> None:
    """Q6 — Advanced feature (README required).

    Implements the verified README Q6 query (On Time vs Late), extended
    with:
      · Statistical verdict (Welch CI + Cohen's d)
      · Severity tier sub-chart (nuance)
      · What-if slider + sensitivity table

    At 'All' this reproduces the verified benchmark:
        On Time: 88,171 orders · 4.29 avg · 10.88 avg days
        Late   :  7,661 orders · 2.57 avg · 31.39 avg days
    """
    st.markdown("### ⚡ Delivery Punctuality vs Satisfaction")
    st.caption(
        "**Advanced feature (Q6 — required).** The single strongest "
        "satisfaction driver. The gap below is what the whole dashboard "
        "should be built around."
    )

    df = apply_filters(get_model(), year, list(states))
    df = df[df["is_delivered"] & df["review_score"].notna()
            & df["days_late"].notna()].copy()
    if df.empty:
        _no_data("No delivered orders with both a delivery date and a review.")
        return

    df["status"] = df["is_late"].map({True: "Late", False: "On Time"})

    # ── Primary binary view — reproduces README Q6 ────────────────────
    grp = (
        df.groupby("status")
          .agg(orders=("order_id", "nunique"),
               avg_score=("review_score", "mean"),
               avg_days=("delivery_days", "mean"))
          .reindex(["On Time", "Late"]).dropna(subset=["orders"])
    )
    if grp.empty:
        _no_data()
        return

    scores = grp["avg_score"].to_dict()
    if "On Time" in scores and "Late" in scores:
        on_score, late_score = scores["On Time"], scores["Late"]
        gap = on_score - late_score
    else:
        on_score = late_score = gap = None

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    colors = [PALETTE.success if s == "On Time" else PALETTE.danger
              for s in grp.index]
    bars = ax.bar(grp.index.astype(str), grp["avg_score"], color=colors,
                  edgecolor=PALETTE.white, linewidth=1.6, width=0.55)
    ax.set_title("Delivery Punctuality vs Satisfaction", pad=14, loc="left")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5.6)
    _style_axes(ax)
    for b, s, n, days in zip(bars, grp["avg_score"], grp["orders"], grp["avg_days"]):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.10,
                f"{s:.2f}★\n(n={int(n):,}; {days:.1f}d)",
                ha="center", va="bottom", fontsize=11,
                color=PALETTE.slate_700, fontweight="bold")

    if gap is not None:
        ax.annotate(
            "", xy=(1, late_score + 0.15), xytext=(1, on_score - 0.15),
            arrowprops=dict(arrowstyle="<->", color=PALETTE.primary_dark, lw=2.4),
        )
        ax.text(1.20, (on_score + late_score) / 2,
                f"Δ {gap:.2f}★", ha="left", va="center",
                fontsize=15, color=PALETTE.primary_dark, fontweight="bold")
    _finish(fig)

    # ── Statistical verdict ────────────────────────────────────────────
    if "On Time" in scores and "Late" in scores:
        a = df.loc[df["status"] == "On Time", "review_score"]
        b = df.loc[df["status"] == "Late", "review_score"]
        na, nb = len(a), len(b)
        if na >= PUNCT_MIN_N_TIER and nb >= PUNCT_MIN_N_TIER:
            diff, ci_lo, ci_hi, t_stat, d = _welch_ci(a, b)
            robust = (ci_lo == ci_lo and ci_hi == ci_hi and ci_lo > 0)
            sig = ("✅ statistically robust" if robust
                   else "⚠️ CI includes 0 — inconclusive")
            d_txt = f"{d:.2f} ({_cohens_verdict(d)})" if d == d else "n/a"
            ci_txt = (f"[{ci_lo:+.2f}, {ci_hi:+.2f}]"
                      if ci_lo == ci_lo else "n/a (scipy unavailable)")
            verdict = (
                f"<b>On-Time − Late gap: {diff:+.2f}★</b> "
                f"(95% CI {ci_txt}) · Welch t = {t_stat:.2f} · "
                f"Cohen's d = {d_txt} · n_on={na:,}, n_late={nb:,} · {sig}"
            )
        else:
            verdict = (
                f"<b>Insufficient sample for a statistical test.</b> "
                f"On-time n={na:,}, late n={nb:,}; need ≥ "
                f"{PUNCT_MIN_N_TIER} each. Widen the Year or State filters."
            )
        st.markdown(
            f"""
            <div class="verdict-card">
              <span class="verdict-label">📊 Statistical verdict</span>
              <span class="verdict-text">{verdict}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Severity tiers (nuance) ────────────────────────────────────────
    st.markdown("#### 🔍 Nuance — how the penalty grows with lateness")
    st.caption(
        "The binary split hides the dose-response. The sub-chart below "
        "shows how review scores fall as lateness worsens."
    )
    df["severity"] = "On Time"
    df.loc[df["days_late"].between(1, 3), "severity"] = "1–3d late"
    df.loc[df["days_late"].between(4, 7), "severity"] = "4–7d late"
    df.loc[df["days_late"] >= 8, "severity"] = "8+d late"
    tier_order = ["On Time", "1–3d late", "4–7d late", "8+d late"]
    tgrp = (
        df.groupby("severity")
          .agg(orders=("order_id", "nunique"),
               avg_score=("review_score", "mean"))
          .reindex(tier_order).dropna(subset=["orders"])
    )
    if not tgrp.empty:
        colors_t = [PALETTE.success, PALETTE.warn_light,
                    PALETTE.warn, PALETTE.danger][:len(tgrp)]
        fig2, ax2 = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
        b2 = ax2.bar(tgrp.index.astype(str), tgrp["avg_score"],
                     color=colors_t, edgecolor=PALETTE.white, linewidth=1.4,
                     width=0.62)
        ax2.set_title("Avg score by lateness severity", pad=14, loc="left")
        ax2.set_ylabel("Avg review score (1–5)")
        ax2.set_ylim(1, 5.4)
        _style_axes(ax2)
        for b, s, n in zip(b2, tgrp["avg_score"], tgrp["orders"]):
            flag = " ⚠️" if n < PUNCT_MIN_N_TIER else ""
            ax2.text(b.get_x() + b.get_width() / 2, s + 0.08,
                     f"{s:.2f}\nn={int(n):,}{flag}",
                     ha="center", va="bottom", fontsize=10,
                     color=PALETTE.slate_700, fontweight="bold")
        _finish(fig2)

    # ── What-if + sensitivity (whole selection) ────────────────────────
    if "On Time" in scores and "Late" in scores:
        a = df.loc[df["status"] == "On Time", "review_score"]
        b = df.loc[df["status"] == "Late", "review_score"]
        if len(a) >= PUNCT_MIN_N_TIER and len(b) >= PUNCT_MIN_N_TIER:
            st.markdown("#### 🎛️ What-if: convert late orders to on-time")
            pct = st.slider(
                "Percentage of late orders made on-time",
                min_value=0, max_value=100, value=10, step=5,
                key=f"whatif_{year}_{len(states)}",
                help="Projects the effect on the selection average. "
                     "Assumes converted orders land at the on-time mean.",
            )
            on_mean, late_mean = float(a.mean()), float(b.mean())
            n_on, n_late = len(a), len(b)
            sel_avg = float(df["review_score"].mean())
            new_total = n_on + n_late
            converted = n_late * pct / 100.0
            new_mean = (n_on * on_mean
                        + (n_late - converted) * late_mean
                        + converted * on_mean) / new_total
            c1, c2, c3 = st.columns(3, gap="medium")
            c1.metric("Late orders fixed", f"{converted:,.0f}",
                      delta=f"{pct}% of {n_late:,}")
            c2.metric("Expected avg score", f"{new_mean:.3f}★")
            c3.metric("Δ vs current", f"{new_mean - sel_avg:+.3f}★")

            with st.expander("📋 Full sensitivity table (5%, 10%, 25%, 50%)"):
                rows = []
                for p in PUNCT_SCENARIOS:
                    conv = n_late * p / 100.0
                    nm = (n_on * on_mean
                          + (n_late - conv) * late_mean
                          + conv * on_mean) / new_total
                    rows.append({
                        "Scenario": f"{p}% of late → on-time",
                        "Late orders fixed": f"{conv:,.0f}",
                        "Expected avg score": f"{nm:.3f}",
                        "Δ vs current": f"{nm - sel_avg:+.3f}★",
                    })
                _st_dataframe(pd.DataFrame(rows))
                lift_1pct = (n_late * 0.01 * (on_mean - late_mean)) / new_total
                st.caption(f"Stars gained per 1% of late orders converted = "
                           f"**{lift_1pct:+.4f}★**.")

    if gap is not None and "On Time" in scores and "Late" in scores:
        det_rate = (df.loc[df["status"] == "Late", "review_score"]
                    .le(2).mean() * 100)
        _insight(
            year, states,
            f"Late orders average <b>{late_score:.2f}★</b> vs "
            f"<b>{on_score:.2f}★</b> on time — a gap of <b>{gap:.2f}★</b>. "
            f"{det_rate:.0f}% of late orders become 1–2★ detractors. "
            "This dwarfs every geographic difference on the dashboard."
        )
    _partial_note(year)


# ════════════════════════════════════════════════════════════════════════════
# §13  Brazil map (supporting, in Customer Satisfaction)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CONFIG.geojson_cache_ttl_seconds, show_spinner=False)
def _load_geojson() -> dict[str, Any] | None:
    for url in CONFIG.geojson_urls:
        try:
            r = requests.get(url, timeout=CONFIG.geojson_timeout_seconds)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            _LOGGER.warning("GeoJSON failed: %s — %s", url, exc)
    return None


def _geojson_to_gdf(geojson: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        sigla = (props.get("sigla") or props.get("SIGLA")
                 or props.get("abbrev") or props.get("code")
                 or props.get("name"))
        if sigla:
            rows.append({"state": str(sigla).upper(),
                         "geometry": feat.get("geometry", {})})
    return pd.DataFrame(rows)


def _plot_geo(ax, geometry, facecolor):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not gtype or not coords:
        return
    polys = [coords] if gtype == "Polygon" else (
        coords if gtype == "MultiPolygon" else []
    )
    for poly in polys:
        for ring_idx, ring in enumerate(poly):
            if not ring:
                continue
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            ax.fill(xs, ys, facecolor=facecolor, edgecolor=PALETTE.white,
                    linewidth=0.6, zorder=2 if ring_idx == 0 else 3)


def chart_brazil_map(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Brazil Map · Delivered Orders")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["is_delivered"] & df["customer_state"].notna()]
    counts = df.groupby("customer_state")["order_id"].nunique()
    if counts.empty:
        _no_data("No state-level data for this year.")
        return
    geojson = _load_geojson()
    if not geojson:
        st.warning("⚠️ Brazil map unavailable (GeoJSON fetch failed).")
        return
    gdf = _geojson_to_gdf(geojson)
    if gdf.empty:
        st.warning("⚠️ Could not parse state geometries.")
        return

    filter_active = bool(states)
    orders_by_state = counts.to_dict()
    hi = float(max(orders_by_state.values())) * 1.05
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(hi, 1.0))
    cmap = plt.get_cmap("YlGnBu")

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    ax.set_axis_off()
    ax.set_title(f"Delivered Orders by State — {scope_label(year, states)}",
                 fontsize=12, fontweight="bold",
                 color=PALETTE.ink_soft, pad=12, loc="left")

    for _, row in gdf.iterrows():
        st_code = row["state"]
        n = orders_by_state.get(st_code)
        in_scope = (not filter_active) or (st_code in states)
        face = cmap(norm(n)) if (n is not None and in_scope) else PALETTE.slate_200
        _plot_geo(ax, row["geometry"], facecolor=face)

    for _, row in gdf.iterrows():
        st_code = row["state"]
        geom = row["geometry"]
        n = orders_by_state.get(st_code)
        in_scope = (not filter_active) or (st_code in states)
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
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        if n is not None and in_scope:
            ax.text(cx, cy, f"{st_code}\n{int(n):,}", ha="center", va="center",
                    fontsize=6, color=PALETTE.ink, fontweight="bold", zorder=5)
        else:
            ax.text(cx, cy, st_code, ha="center", va="center",
                    fontsize=6, color=PALETTE.slate_400, zorder=5)

    ax.set_xlim(-74.0, -34.0)
    ax.set_ylim(-34.0, 6.0)
    ax.set_aspect("equal")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.04, pad=0.04, shrink=0.7)
    cbar.set_label("Delivered orders", fontsize=10, color=PALETTE.slate_500)
    cbar.ax.tick_params(labelsize=9)
    _finish(fig)


# ════════════════════════════════════════════════════════════════════════════
# §14  UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

def info_card(title: str, body: str, *, variant: str = "info",
              tags: Sequence[str] | None = None,
              priority: str = "") -> None:
    pills = ""
    if tags:
        pills = ("<div style='margin-bottom:10px;'>"
                 + "".join(f'<span class="tag">{t}</span>' for t in tags)
                 + "</div>")
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
        body = "<span>Only one delivery status in this selection.</span>"
    else:
        body = (f'<strong>Late deliveries cost {g["gap"]:.2f} stars.</strong> '
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


def render_headline_strip(year: str, states: tuple[str, ...]) -> None:
    g = query_headline_gap(year, states)
    if g["gap"] is None:
        return
    st.markdown(
        f"""
        <div class="headline-strip">
          <span class="headline-strip-icon">⚡</span>
          <div class="headline-strip-text">
            <span class="headline-strip-label">
              Headline finding · {scope_label(year, states)}
            </span>
            <p style="margin:0;">
              Late deliveries cost <b>{g['gap']:.2f}★</b> — the strongest
              satisfaction driver. See the Customer Satisfaction tab for evidence.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_snapshot(year: str, states: tuple[str, ...]) -> None:
    g = query_headline_gap(year, states)
    if g["gap"] is None:
        st.caption("Widen filters to compute the on-time vs late gap.")
        return
    d1, d2, d3 = st.columns(3, gap="medium")
    d1.metric("On-Time Avg Score", f"{g['on_time']:.2f}★")
    d2.metric("Late Avg Score", f"{g['late']:.2f}★")
    d3.metric("Star Gap (Driver)", f"{g['gap']:.2f}★",
              delta="Priority #1" if g["gap"] > 1.0 else None,
              delta_color="off")


def render_executive_summary(year: str, states: tuple[str, ...]) -> None:
    k = query_kpis(year, states)
    g = query_headline_gap(year, states)
    if g["gap"] is None:
        verdict, color = "Inconclusive", PALETTE.slate_500
    elif g["gap"] >= 1.5:
        verdict, color = "Critical priority", PALETTE.danger
    elif g["gap"] >= 1.0:
        verdict, color = "High priority", PALETTE.warn
    elif g["gap"] >= 0.5:
        verdict, color = "Moderate", PALETTE.accent
    else:
        verdict, color = "Healthy", PALETTE.success

    gap_txt = f"{g['gap']:.2f}★" if g["gap"] is not None else "—"
    avg_txt = f"{k['avg_review_score']:.2f}★" if k["avg_review_score"] else "—"

    st.markdown(
        f"""
        <div class="exec-summary">
          <div class="exec-summary-main">
            <div class="exec-summary-eyebrow">EXECUTIVE SUMMARY</div>
            <div class="exec-summary-title">
              Delivery punctuality is the <b>#1 satisfaction lever</b>
            </div>
            <div class="exec-summary-detail">
              Late orders lose <b style="color:{PALETTE.danger};">{gap_txt}</b>
              on average versus on-time orders — a gap far larger than any
              geographic difference. The current selection averages
              <b>{avg_txt}</b> across <b>{k['total_orders']:,}</b> orders.
            </div>
          </div>
          <div class="exec-summary-badge"
               style="border-color:{color}; background:{color}1A; color:{color};">
            <span class="exec-summary-badge-label">Priority</span>
            <span class="exec-summary-badge-value">{verdict}</span>
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
        import plotly.express as px
        fig = px.scatter(
            items, x="effort", y="impact", text="id",
            size=[16] * len(items), color="impact",
            color_continuous_scale=["#6EE7B7", "#FCD34D", "#FDA4AF"],
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
        try:
            st.plotly_chart(fig, width="stretch")
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
        for _, r in items.iterrows():
            c = (PALETTE.danger if r["impact"] >= 5
                 else PALETTE.warn if r["impact"] >= 4
                 else PALETTE.accent)
            ax.scatter(r["effort"], r["impact"], s=400, color=c,
                       edgecolor="white", linewidth=2.5, zorder=3)
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
# §15  GLOBAL CSS (safe selectors, projector-friendly)
# ════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS: Final[str] = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">

<style>
  :root {
    --brand:        #4F46E5;
    --brand-light:  #A5B4FC;
    --brand-dark:   #3730A3;
    --accent:       #F97316;
    --success:      #10B981;
    --warn:         #F59E0B;
    --danger:       #E11D48;
    --ink:          #0F172A;
    --slate-700:    #334155;
    --slate-600:    #475569;
    --slate-500:    #64748B;
    --slate-400:    #94A3B8;
    --slate-300:    #CBD5E1;
    --slate-200:    #E2E8F0;
    --slate-100:    #F1F5F9;
    --slate-50:     #F8FAFC;
  }
  html, body, [class*="css"], .stApp {
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif !important;
    font-size: 15px; line-height: 1.6; color: var(--ink);
    -webkit-font-smoothing: antialiased;
  }
  .stApp { background: var(--slate-50); }
  h1, h2, h3, .app-hero-title, .hero-headline,
  .exec-summary-title, .hero-label, .exec-summary-eyebrow {
    font-family: 'Sora', 'Inter', sans-serif !important;
    letter-spacing: -0.015em;
  }
  code { font-family: 'JetBrains Mono', monospace !important; }

  .app-hero {
    display: flex; align-items: center; gap: 32px;
    background: linear-gradient(135deg, #EEF2FF 0%, #FFFFFF 60%, #FFF7ED 100%);
    border: 1px solid var(--slate-200);
    border-radius: 20px; padding: 32px 40px; margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(79,70,229,0.08);
  }
  .app-hero-logo { flex-shrink: 0; width: 110px; height: auto; object-fit: contain; }
  .app-hero-content { flex: 1; min-width: 0; }
  .app-hero-eyebrow {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--brand); margin-bottom: 12px;
  }
  .app-hero-title {
    font-size: clamp(2.2rem, 1.8rem + 1.6vw, 3rem);
    font-weight: 800 !important; color: var(--ink) !important;
    line-height: 1.08 !important; margin: 0 0 12px 0 !important;
    letter-spacing: -0.035em !important;
  }
  .app-hero-title b { color: var(--brand-dark); }
  .app-hero-sub {
    font-size: 1.05rem; color: var(--slate-600);
    margin: 0 0 16px 0; line-height: 1.55; max-width: 780px;
  }
  .app-hero-meta { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }

  h1 { color: var(--ink); font-weight: 800;
       font-size: clamp(1.4rem, 1.1rem + 1vw, 1.9rem);
       line-height: 1.2; margin-bottom: 8px; }
  h2 { color: var(--ink); font-weight: 700;
       font-size: clamp(1.2rem, 1.05rem + 0.6vw, 1.45rem);
       line-height: 1.25; margin-top: 12px !important;
       margin-bottom: 16px !important; }
  h3 { color: var(--ink); font-weight: 700;
       font-size: clamp(1.05rem, 1rem + 0.3vw, 1.15rem);
       line-height: 1.3; margin-top: 0 !important;
       margin-bottom: 12px !important; }
  h4 { color: var(--brand); font-size: 1rem; font-weight: 700;
       margin-bottom: 12px !important; }
  p, li, span, label { color: var(--slate-700); font-size: 0.95rem; }

  .block-container {
    padding-top: 24px !important; padding-bottom: 64px !important;
    max-width: 1500px;
    padding-left: 24px !important; padding-right: 24px !important;
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
    border-right: 1px solid #312E81;
    padding-top: 16px;
  }
  section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important; font-weight: 700;
    font-family: 'Sora', sans-serif !important;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label {
    color: #A5B4FC !important; font-weight: 600;
    text-transform: uppercase; font-size: 0.7rem;
    letter-spacing: 0.08em;
  }
  section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
  }
  section[data-testid="stSidebar"] table { color: #E2E8F0 !important; }
  section[data-testid="stSidebar"] th,
  section[data-testid="stSidebar"] td {
    color: #E2E8F0 !important; border-color: #312E81 !important;
    font-size: 0.8rem !important; padding: 8px 12px !important;
  }

  /* Sidebar nav — pill buttons via st.radio */
  section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex; flex-direction: column; gap: 6px; padding-top: 6px;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(165,180,252,0.20) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    transition: background 0.15s ease, border-color 0.15s ease;
    cursor: pointer;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(165,180,252,0.14) !important;
    border-color: rgba(165,180,252,0.4) !important;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #E2E8F0 !important; font-weight: 600 !important;
    font-size: 0.92rem !important; margin: 0 !important;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--brand) 0%, var(--brand-dark) 100%) !important;
    border-color: var(--brand-light) !important;
    box-shadow: 0 2px 10px rgba(79,70,229,0.35);
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #FFFFFF !important; font-weight: 700 !important;
  }

  .sidebar-warning {
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.45);
    border-left: 3px solid var(--warn);
    border-radius: 8px; padding: 12px; margin-top: 12px;
    font-size: 0.83rem; line-height: 1.5;
  }
  .sidebar-warning strong { color: #FCD34D !important; }
  .sidebar-warning span   { color: #E2E8F0 !important; }

  .sidebar-headline {
    background: rgba(79, 70, 229, 0.20);
    border: 1px solid rgba(165, 180, 252, 0.45);
    border-left: 3px solid var(--brand-light);
    border-radius: 8px; padding: 12px 14px; margin-top: 16px;
    line-height: 1.55;
  }
  .sidebar-headline-label {
    font-size: 0.65rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--brand-light) !important;
    margin-bottom: 8px;
  }
  .sidebar-headline strong {
    color: #FFFFFF !important; font-size: 0.83rem;
    display: block; margin-bottom: 4px;
  }
  .sidebar-headline span { color: #CBD5E1 !important; font-size: 0.75rem; }
  .sidebar-headline span b { color: #FFFFFF !important; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid var(--slate-200);
    border-left: 5px solid var(--brand);
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 12px rgba(15,23,42,0.04);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(15,23,42,0.05),
                0 12px 24px rgba(79,70,229,0.10);
  }
  [data-testid="stMetricLabel"] {
    color: var(--slate-500) !important;
    font-size: 0.72rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.06em;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  [data-testid="stMetricValue"] {
    color: var(--ink) !important; font-weight: 800 !important;
    font-size: clamp(1.4rem, 1.2rem + 0.6vw, 1.85rem) !important;
    letter-spacing: -0.02em; line-height: 1.15;
    font-family: 'Sora', 'Inter', sans-serif !important;
    font-variant-numeric: tabular-nums;
  }
  [data-testid="stMetricDelta"] {
    font-weight: 600 !important; font-size: 0.78rem !important;
  }

  /* Chart card container */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid var(--slate-200) !important;
    border-radius: 14px !important;
    padding: 20px 24px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03), 0 4px 14px rgba(15,23,42,0.04);
    margin-bottom: 16px;
  }
  [data-testid="stVerticalBlockBorderWrapper"] h3 { margin-top: 0 !important; }

  .stButton > button {
    background: var(--brand); color: #FFFFFF;
    border: none; border-radius: 8px;
    padding: 8px 20px; font-weight: 600;
    transition: background 0.15s ease, transform 0.1s ease;
  }
  .stButton > button:hover {
    background: var(--brand-dark); transform: translateY(-1px);
  }

  [data-testid="stAlert"] {
    background-color: #FEF3C7;
    border-left: 4px solid var(--warn);
    color: #78350F; border-radius: 10px;
  }
  [data-testid="stExpander"] {
    border: 1px solid var(--slate-200);
    border-radius: 12px; background: #FFFFFF; margin-bottom: 12px;
  }
  [data-testid="stExpander"] summary {
    font-weight: 600; color: var(--brand);
    font-family: 'Sora', sans-serif !important;
  }
  hr { border-color: var(--slate-200); margin: 24px 0; }

  /* Info cards */
  .info-card {
    position: relative;
    background: #FFFFFF;
    border: 1px solid var(--slate-200);
    border-left: 5px solid var(--brand);
    border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
  }
  .info-card.warn { border-left-color: var(--danger); }
  .info-card.ok   { border-left-color: var(--success); }
  .info-card.info { border-left-color: var(--brand); }
  .info-card.warn h4 { color: var(--danger); }
  .info-card.ok h4   { color: var(--success); }
  .info-card.info h4 { color: var(--brand); }
  .info-card p, .info-card li {
    color: var(--slate-700); line-height: 1.65; font-size: 0.95rem;
  }
  .info-card .tag {
    display: inline-block; background: var(--slate-100);
    color: var(--slate-600); border-radius: 999px;
    padding: 4px 12px; font-size: 0.72rem; font-weight: 600;
    margin-right: 6px; margin-bottom: 6px;
  }
  .priority-bar {
    position: absolute; top: 0; right: 0; width: 5px; height: 100%;
    border-radius: 0 12px 12px 0;
  }
  .priority-high   { background: var(--danger); }
  .priority-medium { background: var(--warn); }
  .priority-low    { background: var(--success); }

  /* Headline strip / hero */
  .hero-banner, .headline-strip {
    display: flex; align-items: center; gap: 12px;
    border-radius: 12px; padding: 14px 18px; margin: 16px 0;
  }
  .hero-banner {
    background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 55%, #C7D2FE 100%);
    border: 1px solid #A5B4FC; border-left: 4px solid var(--brand);
    box-shadow: 0 2px 10px rgba(79,70,229,0.10);
  }
  .headline-strip {
    background: linear-gradient(90deg, #EEF2FF 0%, #FFFFFF 100%);
    border: 1px solid #C7D2FE; border-left: 4px solid var(--brand);
  }
  .hero-banner .hero-icon, .headline-strip-icon {
    font-size: 20px; line-height: 1; flex-shrink: 0;
  }
  .hero-banner .hero-text, .headline-strip-text { flex: 1; min-width: 0; }
  .hero-banner .hero-label, .headline-strip-label {
    display: block; font-size: 0.68rem; font-weight: 800;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--brand); margin-bottom: 4px;
  }
  .hero-banner .hero-headline, .headline-strip-text p {
    font-size: 0.95rem; font-weight: 500; color: var(--slate-700);
    line-height: 1.55; margin: 0;
  }
  .hero-banner .hero-headline b, .headline-strip-text b {
    color: var(--brand-dark); font-weight: 700;
  }

  .exec-summary {
    display: flex; align-items: stretch; gap: 16px;
    background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 100%);
    border: 1px solid var(--slate-200); border-radius: 14px;
    padding: 24px; margin: 16px 0;
    box-shadow: 0 2px 12px rgba(79,70,229,0.06);
  }
  .exec-summary-main { flex: 1; min-width: 0; }
  .exec-summary-eyebrow {
    font-size: 0.68rem; font-weight: 800; letter-spacing: 0.16em;
    color: var(--brand); margin-bottom: 8px;
  }
  .exec-summary-title {
    font-size: clamp(1.2rem, 1.05rem + 0.6vw, 1.45rem);
    font-weight: 800; color: var(--ink); line-height: 1.25;
    margin-bottom: 8px;
  }
  .exec-summary-title b { color: var(--brand-dark); }
  .exec-summary-detail {
    color: var(--slate-600); font-size: 0.95rem; line-height: 1.6;
  }
  .exec-summary-badge {
    display: flex; flex-direction: column; justify-content: center;
    align-items: center; text-align: center;
    min-width: 130px; padding: 12px 16px;
    border: 2px solid; border-radius: 12px;
  }
  .exec-summary-badge-label {
    font-size: 0.65rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; opacity: 0.85; margin-bottom: 8px;
  }
  .exec-summary-badge-value {
    font-family: 'Sora', sans-serif; font-weight: 800;
    font-size: 1rem; line-height: 1.2;
  }

  .overview-banner {
    background: #FEF3C7; border-left: 4px solid var(--warn);
    border-radius: 12px; padding: 12px 20px; margin-bottom: 16px;
    font-size: 0.95rem; color: #78350F; line-height: 1.55;
  }
  .overview-banner b { color: #92400E; }

  .scope-pill {
    display: inline-block;
    background: linear-gradient(135deg, var(--brand) 0%, var(--accent) 100%);
    color: #FFFFFF; border-radius: 999px;
    padding: 8px 16px; font-size: 0.83rem; font-weight: 600;
    box-shadow: 0 2px 10px rgba(79,70,229,0.30);
  }
  .scope-pill-subtle {
    display: inline-block; background: var(--slate-100);
    color: var(--slate-700); border: 1px solid var(--slate-200);
    border-radius: 999px; padding: 8px 16px;
    font-size: 0.83rem; font-weight: 600;
  }

  .insight {
    display: flex; gap: 12px; align-items: flex-start;
    background: #F5F3FF; border-left: 3px solid var(--brand-light);
    border-radius: 8px; padding: 12px 16px; margin-top: 12px;
    font-size: 0.83rem;
  }
  .insight-chip {
    flex-shrink: 0; background: var(--brand); color: #FFFFFF;
    padding: 2px 10px; border-radius: 999px;
    font-size: 0.7rem; font-weight: 700; white-space: nowrap;
  }
  .insight-body { display: flex; flex-direction: column; gap: 2px; }
  .insight-scope {
    font-size: 0.68rem; font-weight: 700; color: var(--brand);
    text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.9;
  }
  .insight-text { color: var(--slate-600); line-height: 1.55; }
  .insight-text b { color: var(--ink); font-weight: 700; }

  .verdict-card {
    display: flex; flex-direction: column; gap: 4px;
    background: linear-gradient(135deg, #ECFDF5 0%, #F0FDFA 100%);
    border: 1px solid #6EE7B7; border-left: 4px solid var(--success);
    border-radius: 10px; padding: 12px 16px; margin: 12px 0;
  }
  .verdict-label {
    font-size: 0.7rem; font-weight: 800; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--success);
  }
  .verdict-text {
    font-size: 0.83rem; color: var(--slate-700); line-height: 1.55;
  }
  .verdict-text b { color: var(--ink); }

  .empty-state {
    text-align: center; padding: 56px 24px;
    background: var(--slate-50); border: 1px dashed var(--slate-300);
    border-radius: 12px;
  }
  .empty-icon { font-size: 36px; }
  .empty-title {
    font-size: 1.15rem; font-weight: 700; color: var(--slate-700);
    margin: 8px 0; font-family: 'Sora', sans-serif;
  }
  .empty-body { color: var(--slate-500); font-size: 0.95rem; }
  .empty-hint {
    margin-top: 12px; color: var(--brand);
    font-size: 0.83rem; font-weight: 500;
  }

  [data-testid="stDataFrame"] {
    border-radius: 12px; overflow: hidden;
    border: 1px solid var(--slate-200);
  }
  .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--slate-500); font-size: 0.83rem;
  }

  @media (max-width: 1100px) {
    .app-hero { flex-direction: column; align-items: flex-start;
                padding: 24px; gap: 16px; }
    .app-hero-logo { width: 80px; }
    .exec-summary { flex-direction: column; }
  }
  @media (max-width: 768px) {
    .block-container { padding: 16px !important; }
    .app-hero { padding: 20px 16px; }
    .app-hero-title { font-size: 1.8rem !important; }
    .hero-banner, .headline-strip { flex-direction: column; align-items: flex-start; }
  }
</style>
"""


def _apply_global_style() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    mpl.rcParams.update({
        "figure.facecolor":   "#FFFFFF",
        "axes.facecolor":     "#FFFFFF",
        "axes.grid":          False,
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
        "legend.frameon":     False,
        "legend.fontsize":    10,
        "figure.autolayout":  False,
    })


# ════════════════════════════════════════════════════════════════════════════
# §16  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

NAV_TABS: Final[tuple[str, ...]] = (
    "📈  Overview",
    "⭐  Customer Satisfaction",
    "🔬  Detail & Method",
    "💡  Recommendations",
)


def render_sidebar() -> tuple[str, str, tuple[str, ...]]:
    # 1. Nav tabs at the top
    st.sidebar.markdown("### 🧭 Navigation")
    active_tab = st.sidebar.radio(
        "Navigation", NAV_TABS, index=0, key="active_tab",
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    # 2. Filters below
    st.sidebar.subheader("🎛️ Filters")
    st.sidebar.caption(
        "Both filters apply to every KPI and chart. "
        "Set both to All to reproduce the reference values."
    )

    def _reset() -> None:
        st.session_state["filter_year"] = "All"
        st.session_state["filter_states"] = []

    year = st.sidebar.selectbox(
        "Reporting Year", CONFIG.years, index=0, key="filter_year",
    )
    if is_partial_year(year):
        st.sidebar.markdown(
            f"""
            <div class="sidebar-warning">
              <strong>⚠️ {year} is a partial year</strong>
              <span>Year-over-year comparisons are misleading.
              See Detail &amp; Method.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    states = st.sidebar.multiselect(
        "State", options=CONFIG.all_states, default=[],
        placeholder="All states",
        help="Pick one or more states. Leaving empty shows all states.",
        key="filter_states",
    )
    states_tuple = tuple(states)

    if states or year != "All":
        st.sidebar.button("↺  Reset filters", on_click=_reset,
                          use_container_width=True)

    st.sidebar.divider()
    render_headline_sidebar(year, states_tuple)
    st.sidebar.divider()
    st.sidebar.markdown("**📊 Reference — filters = All**")
    st.sidebar.markdown(
        "| KPI | Value |\n"
        "|---|---|\n"
        f"| Total Orders | {int(REF['total_orders']):,} |\n"
        f"| Delivered | {int(REF['delivered_orders']):,} |\n"
        f"| Avg Review | {REF['avg_review_score']:.2f} |\n"
        f"| Late | {int(REF['late_deliveries']):,} |\n"
        f"| Avg Delivery Days | {REF['avg_delivery_days']:.2f} |"
    )
    return active_tab, year, states_tuple


# ════════════════════════════════════════════════════════════════════════════
# §17  TABS
# ════════════════════════════════════════════════════════════════════════════

def render_overview(year: str, states: tuple[str, ...],
                    filter_active: bool) -> None:
    st.markdown("## 📈 Overview")
    st.markdown("Volume, sentiment, and year contribution at a glance.")

    if is_partial_year(year):
        detail = {
            "2016": "2016 contains only September, October and December — "
                    "November 2016 has zero orders.",
            "2018": "2018 ends on 17 October — it is not a full calendar year.",
        }[year]
        st.markdown(
            f"""
            <div class="overview-banner">
              <b>⚠️ Partial year — {year}.</b> {detail}
              Year-over-year comparisons against a full year are misleading.
              Set the Year filter to <b>All</b> or <b>2017</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    k = query_kpis(year, states)
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    c1.metric("Total Orders", fmt_int(k["total_orders"]))
    c2.metric("Delivered Orders", fmt_int(k["delivered_orders"]))
    if k["avg_review_score"] is None:
        c3.metric("Avg Review Score", "—")
    else:
        delta = (f"{k['avg_review_score'] - REF['avg_review_score']:+.2f} vs baseline"
                 if filter_active else None)
        c3.metric("Avg Review Score", fmt_float(k["avg_review_score"]),
                  delta=delta)
    late_val, delivered_val = k["late_deliveries"], k["delivered_orders"]
    late_delta = (f"{100 * late_val / delivered_val:.1f}% of delivered"
                  if delivered_val else None)
    c4.metric("Late Deliveries", fmt_int(late_val),
              delta=late_delta, delta_color="inverse")
    c5.metric("Avg Delivery Days", fmt_float(k["avg_delivery_days"]))

    if filter_active:
        st.info(f"Active filter — **{scope_label(year, states)}**. "
                "Set both filters to All to reproduce sidebar reference values.")

    render_headline_strip(year, states)
    _chart_card(chart_review_distribution, year, states)


def render_theme(year: str, states: tuple[str, ...]) -> None:
    st.markdown("## ⭐ Customer Satisfaction")
    render_decision_snapshot(year, states)
    st.divider()
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        _chart_card(chart_state_scores, year, states)
    with col_b:
        _chart_card(chart_ontime_by_state, year, states)
    _chart_card(chart_monthly_score, year, states)
    _chart_card(chart_brazil_map, year, states)


def render_detail(year: str, states: tuple[str, ...]) -> None:
    st.markdown("## 🔬 Detail & Method")
    st.markdown("The advanced feature, its evidence, and the caveats that "
                "govern how far you can take the findings.")

    # ── Q6 ADVANCED — belongs here per README ──────────────────────────
    _chart_card(chart_punctuality_vs_satisfaction, year, states)

    st.divider()
    st.markdown("### Limitations")

    info_card(
        "L1 — Association, not causation",
        """<p>The on-time vs late star gap is a strong <i>association</i>,
        not proof that lateness alone causes lower ratings. Product
        quality, seller performance, distance and communication failures
        all contribute.</p>
        <p>Before claiming a specific intervention will increase ratings
        by a precise amount, use a controlled pilot or a regression that
        controls for confounders.</p>""",
        variant="warn", tags=["Causality", "Confounders"],
    )
    info_card(
        "L2 — Incomplete time coverage",
        """<p>2016 contains only September, October and December, with
        <b>no November orders</b>. 2018 ends on <b>17 October</b>.
        <b>2017 is the only complete calendar year</b>.</p>""",
        variant="warn", tags=["Time coverage", "YoY risk"],
    )
    info_card(
        "L3 — Review fan-out",
        """<p>The dataset has <b>99,224 review rows across 98,673
        distinct orders</b> — 551 rows of fan-out. The dashboard
        <b>averages review scores per order</b> before analysis.</p>""",
        variant="warn", tags=["Data quality", "Fan-out"],
    )
    info_card(
        "L4 — Customer identity",
        """<p>The <code>customers</code> table has <b>99,441 rows</b> but
        only <b>96,096 distinct <code>customer_unique_id</code></b>
        values. Any customer count here is order-grain.</p>""",
        variant="info", tags=["Identity", "Grain"],
    )
    info_card(
        "L5 — No delivery-distance variable",
        """<p>The geolocation CSV is deliberately excluded. The analysis
        cannot distinguish long-distance transit from carrier
        underperformance.</p>""",
        variant="info", tags=["Scope", "Distance"],
    )
    info_card(
        "L6 — Dataset and generalisability boundaries",
        """<p>Findings concern Olist in Brazil during 2016–2018. They may
        not generalise to other markets or current conditions.</p>""",
        variant="info", tags=["External validity"],
    )
    info_card(
        "L7 — Small selections are noisy",
        """<p>Narrow selections (one state × one year) can contain only a
        few hundred reviews. Averages from small samples swing widely —
        read the <code>n=</code> labels before drawing conclusions. The
        punctuality chart flags tiers below <b>n=30</b> with ⚠️.</p>""",
        variant="info", tags=["Sample size", "Filters"],
    )

    st.divider()
    st.markdown("### Whole-dataset diagnostics")
    conn = _build_connection()
    reviews_qa = pd.read_sql(
        "SELECT COUNT(*) AS review_rows, "
        "       COUNT(DISTINCT order_id) AS distinct_orders, "
        "       COUNT(*) - COUNT(DISTINCT order_id) AS duplicate_rows "
        "FROM reviews", conn,
    )
    customers_qa = pd.read_sql(
        "SELECT (SELECT COUNT(*) FROM customers) AS customer_rows, "
        "       (SELECT COUNT(DISTINCT customer_unique_id) FROM customers) "
        "           AS unique_customers",
        conn,
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Reviews fan-out**")
        _st_dataframe(reviews_qa)
    with c2:
        st.markdown("**Customer identity**")
        _st_dataframe(customers_qa)


def render_recommendations() -> None:
    st.markdown("## 💡 Recommendations")
    st.markdown(
        "Prioritised actions following from the headline finding. "
        "They are based on the **whole dataset**, so they do not change "
        "when the sidebar filters move."
    )
    render_priority_matrix()
    st.divider()
    st.markdown("### Prioritised recommendations")
    info_card(
        "🔴 R1 — Investigate RJ delivery failures (highest ROI)",
        """<p><b>Evidence.</b> RJ carries 12,687 orders at a 13.5% late
        rate — 2.3× the 5.9% national average.</p>
        <p><b>Action.</b> Audit RJ carriers, warehouse locations, last-mile
        routes and hand-off delays. Controlled pilot first.</p>
        <p><b>Lead.</b> Logistics / Operations + Regional Management.</p>""",
        variant="warn",
        tags=["Impact: High", "Effort: Medium", "RJ"],
        priority="high",
    )
    info_card(
        "🔴 R2 — Trigger intervention before day 20",
        """<p><b>Evidence.</b> Delivery buckets decline 4.45 → 4.36 → 4.22
        → 3.69 → 2.26. The largest single step (−0.53★) is between
        11–20 and 21–30 days.</p>
        <p><b>Action.</b> Ageing-order alerts; escalate at-risk parcels;
        revised ETAs.</p>
        <p><b>Lead.</b> Logistics / Operations + Customer Experience.</p>""",
        variant="warn",
        tags=["Impact: High", "Effort: Low", "SLA"],
        priority="high",
    )
    info_card(
        "🟡 R3 — Recalibrate estimated delivery dates",
        """<p><b>Evidence.</b> Late orders run 9.55 days past estimate on
        average — systematic optimism, not random slippage.</p>
        <p><b>Action.</b> Fit ETA from realised transit history by carrier
        and destination; back-test predicted vs actual.</p>
        <p><b>Lead.</b> Data Science / Analytics + Logistics.</p>""",
        variant="info",
        tags=["Impact: Medium", "Effort: Medium", "Forecasting"],
        priority="medium",
    )
    info_card(
        "🟡 R4 — Pilot improvement in Bahia (BA)",
        """<p><b>Evidence.</b> BA scores 3.86 — the lowest of any state
        with 2,000+ orders — across ~3,340 orders.</p>
        <p><b>Action.</b> Apply the RJ diagnostic checklist to BA; test at
        limited scale.</p>
        <p><b>Lead.</b> Regional Operations + Logistics + CX.</p>""",
        variant="info",
        tags=["Impact: Medium", "Effort: Medium", "BA"],
        priority="medium",
    )
    info_card(
        "🟢 R5 — Promise reliability, not unsupported speed",
        """<p><b>Evidence.</b> On-time = 4.29; late = 2.57. The damage is
        in the <i>miss</i>, not the speed.</p>
        <p><b>Action.</b> Align delivery messaging with realistic estimates
        the network can meet.</p>
        <p><b>Lead.</b> Marketing / Customer Communications + Logistics.</p>""",
        variant="ok",
        tags=["Impact: High", "Effort: Low", "Brand"],
        priority="low",
    )


# ════════════════════════════════════════════════════════════════════════════
# §18  BOOTSTRAP
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
                "Built with Streamlit · Matplotlib · pandas"
            ),
        },
    )
    _apply_global_style()

    st.session_state.setdefault("filter_year", "All")
    st.session_state.setdefault("filter_states", [])
    st.session_state.setdefault("active_tab", NAV_TABS[0])

    # Warm-load the model once with a spinner
    try:
        with st.spinner("Preparing analysis model…"):
            get_model()
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.exception("Data load failed")
        st.error(
            f"❌ **Data load failed.** {exc}\n\n"
            "On Streamlit Cloud, ensure `OLIST_DATA_DIR` points to the "
            "folder with the CSVs (or commit a `data/` folder to the repo)."
        )
        st.stop()

    active_tab, year, states = render_sidebar()
    k_live = query_kpis(year, states)

    logo_src = _get_logo_base64()
    avg_live = (f"{k_live['avg_review_score']:.2f}★"
                if k_live["avg_review_score"] else f"{REF['avg_review_score']:.2f}★")
    st.markdown(
        f"""
        <div class="app-hero">
          <img src="{logo_src}" class="app-hero-logo" alt="PORA Academy Logo">
          <div class="app-hero-content">
            <div class="app-hero-eyebrow">
              PORA Academy · Cohort 7 · Capstone
            </div>
            <div class="app-hero-title">
              Customer <b>Satisfaction</b> Dashboard
            </div>
            <p class="app-hero-sub">
              A decision-first view of 99,000+ Olist orders.
              Delivery punctuality is the strongest satisfaction driver —
              late orders lose <b>{REF['star_gap']:.2f}★</b> on average.
            </p>
            <div class="app-hero-meta">
              <span class="scope-pill">Scope: {scope_label(year, states)}</span>
              <span class="scope-pill-subtle">📦 {k_live['total_orders']:,} orders</span>
              <span class="scope-pill-subtle">⭐ {avg_live} avg</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_executive_summary(year, states)
    st.write("")

    filter_active = year != "All" or bool(states)

    if active_tab == NAV_TABS[0]:
        render_overview(year, states, filter_active=filter_active)
    elif active_tab == NAV_TABS[1]:
        render_theme(year, states)
    elif active_tab == NAV_TABS[2]:
        render_detail(year, states)
    else:
        render_recommendations()

    st.divider()
    st.caption(
        f"**Reference (filters = All):** Total Orders "
        f"{int(REF['total_orders']):,} · Delivered "
        f"{int(REF['delivered_orders']):,} · Avg Review "
        f"{REF['avg_review_score']:.2f} · Late "
        f"{int(REF['late_deliveries']):,} · Avg Delivery Days "
        f"{REF['avg_delivery_days']:.2f} · Headline gap "
        f"{REF['star_gap']:.2f}★ (on-time {REF['on_time_score']:.2f} vs "
        f"late {REF['late_score']:.2f}). "
        "Every KPI, chart and insight line responds to the Year and State "
        "filters and reconciles to the verified README tables when both "
        "are set to All."
    )


if __name__ == "__main__":
    main()