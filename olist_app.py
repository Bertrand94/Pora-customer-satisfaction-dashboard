# olist_app.py
"""
════════════════════════════════════════════════════════════════════════════
OLIST CUSTOMER SATISFACTION DASHBOARD
PORA Academy · Cohort 7 · Group 1 Capstone

A decision-first dashboard built on 99,441 Olist orders (2016–2018).
Primary finding: delivery punctuality is the #1 satisfaction lever —
late orders cost ~1.73★ (40% of the on-time score).

DEPLOY
  Local:    streamlit run olist_app.py
  Cloud:    Streamlit Community Cloud → set OLIST_DATA_DIR if CSVs
            are not in ./data
REQUIREMENTS
  streamlit>=1.30 · pandas>=2.0 · matplotlib>=3.7 · requests
  Optional: scipy (CI), plotly (interactive matrix)
════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import base64
import hashlib
import logging
import math
import os
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
    # Brand
    primary:       str = "#6366F1"   # Indigo 500
    primary_light: str = "#A5B4FC"
    primary_dark:  str = "#4338CA"
    accent:        str = "#F97316"   # Orange 500
    accent_light:  str = "#FDBA74"
    accent_dark:   str = "#C2410C"

    # Semantic
    success:       str = "#10B981"
    success_light: str = "#6EE7B7"
    success_dark:  str = "#047857"
    warn:          str = "#F59E0B"
    warn_light:    str = "#FCD34D"
    warn_dark:     str = "#B45309"
    danger:        str = "#EF4444"
    danger_light:  str = "#FCA5A5"
    danger_dark:   str = "#B91C1C"

    # Neutrals
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
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ),
            "customers": ("customer_id", "customer_unique_id",
                          "customer_state"),
            "order_items": ("order_id", "price", "freight_value"),
            "reviews":   ("order_id", "review_score"),
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
    n = _safe_num(num)
    d = _safe_num(den)
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


def _safe_sum(series: pd.Series | None) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(series.fillna(0).sum())


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
# §2.y  STREAMLIT API SHIMS (version drift)
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
# §3  DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

def _data_dir() -> str:
    return os.environ.get(
        "OLIST_DATA_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    )


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds,
               show_spinner="Loading Olist data…")
def load_data() -> dict[str, pd.DataFrame]:
    data_dir = _data_dir()
    frames: dict[str, pd.DataFrame] = {}

    for table, filename in CONFIG.tables.items():
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing CSV for table {table!r} at {path}. "
                "Set OLIST_DATA_DIR to the folder containing the CSVs."
            )
        try:
            frames[table] = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Could not read {filename}: {exc}") from exc

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
        "Data loaded: %d tables, %d rows",
        len(frames), sum(len(df) for df in frames.values()),
    )
    return frames


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds,
               show_spinner="Preparing analysis model…")
def build_model(frames_fingerprint: str) -> pd.DataFrame:
    frames = load_data()
    orders    = frames["orders"].copy()
    customers = frames["customers"][["customer_id", "customer_state"]].copy()
    reviews   = frames["reviews"][["order_id", "review_score"]].copy()

    reviews["review_score"] = pd.to_numeric(
        reviews["review_score"], errors="coerce"
    )

    reviews_order = (
        reviews.dropna(subset=["review_score"])
               .groupby("order_id", as_index=False)["review_score"].mean()
               .rename(columns={"review_score": "review_score_order"})
    )

    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(reviews_order, on="order_id", how="left")

    for col in (
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["year"]  = df["order_purchase_timestamp"].dt.year.astype("Int64").astype(str)
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    df["is_delivered"] = (
        df["order_status"].eq("delivered")
        & df["order_delivered_customer_date"].notna()
    )

    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    df["is_late"] = (
        df["order_status"].eq("delivered")
        & (df["order_delivered_customer_date"]
           > df["order_estimated_delivery_date"])
    )

    df["days_late"] = (
        df["order_delivered_customer_date"]
        - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0

    return df


def get_model() -> pd.DataFrame:
    frames = load_data()
    fingerprint = hashlib.md5(
        "|".join(f"{k}:{len(v)}" for k, v in sorted(frames.items())).encode()
    ).hexdigest()
    return build_model(fingerprint)


def apply_filters(df: pd.DataFrame, year: str,
                  states: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    if year and year != "All" and "year" in out.columns:
        out = out[out["year"] == year]
    if states and "customer_state" in out.columns:
        out = out[out["customer_state"].isin(list(states))]
    return out


def scope_label(year: str, states: Sequence[str]) -> str:
    y = "All years" if (not year or year == "All") else year
    s = ", ".join(states) if states else "All states"
    return f"{y} · {s}"


def is_partial_year(year: str | None) -> bool:
    return bool(year) and year in PARTIAL_YEARS


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds, show_spinner=False)
def _reviews_with_order_scope(
    year: str, states_tuple: tuple[str, ...], *, delivered_only: bool = False
) -> pd.DataFrame:
    frames = load_data()
    model = apply_filters(get_model(), year, list(states_tuple))
    if model is None or model.empty:
        return pd.DataFrame(
            columns=["order_id", "review_score", "year", "month",
                     "customer_state"]
        )
    if delivered_only:
        model = model[model["is_delivered"]]
        if model.empty:
            return pd.DataFrame(
                columns=["order_id", "review_score", "year", "month",
                         "customer_state"]
            )

    reviews = frames["reviews"][["order_id", "review_score"]].copy()
    reviews["review_score"] = pd.to_numeric(
        reviews["review_score"], errors="coerce"
    )
    scope = model[["order_id", "year", "month", "customer_state"]].copy()
    return reviews.merge(scope, on="order_id", how="inner")


# ════════════════════════════════════════════════════════════════════════════
# §4  KPI COMPUTATION  (Q1, Q6)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds, show_spinner=False)
def query_kpis(year: str, states_tuple: tuple[str, ...]) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "total_orders": 0, "delivered_orders": 0, "delivered_pct": None,
        "avg_review_score": None, "late_deliveries": 0, "late_pct": None,
        "avg_delivery_days": None, "review_rows": 0,
        "delivered_with_date_orders": 0,
    }

    df = apply_filters(get_model(), year, list(states_tuple))
    if df is None or df.empty:
        return empty

    delivered_status = df[df["order_status"].eq("delivered")]
    delivered_with_date = delivered_status[
        delivered_status["order_delivered_customer_date"].notna()
    ]

    reviews = _reviews_with_order_scope(year, states_tuple, delivered_only=False)
    avg_review = _safe_mean(reviews["review_score"]) if not reviews.empty else None
    avg_days = (
        _safe_mean(delivered_with_date["delivery_days"])
        if not delivered_with_date.empty else None
    )

    total_orders     = int(df["order_id"].nunique())
    delivered_orders = int(delivered_status["order_id"].nunique())
    late_deliveries  = (
        int(delivered_with_date["is_late"].sum())
        if not delivered_with_date.empty else 0
    )

    delivered_pct = _safe_div(delivered_orders, total_orders)
    late_pct      = _safe_div(late_deliveries, delivered_orders)

    return {
        "total_orders":      total_orders,
        "delivered_orders":  delivered_orders,
        "delivered_pct":     (delivered_pct * 100.0)
                             if delivered_pct is not None else None,
        "avg_review_score":  float(avg_review) if avg_review is not None else None,
        "late_deliveries":   late_deliveries,
        "late_pct":          (late_pct * 100.0)
                             if late_pct is not None else None,
        "avg_delivery_days": float(avg_days) if avg_days is not None else None,
        "review_rows":       int(len(reviews)),
        "delivered_with_date_orders":
            int(delivered_with_date["order_id"].nunique())
            if not delivered_with_date.empty else 0,
    }


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds, show_spinner=False)
def query_headline_gap(year: str,
                       states_tuple: tuple[str, ...]) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "on_time": None, "late": None, "gap": None,
        "gap_pct": None, "n_on": 0, "n_late": 0,
    }

    reviews = _reviews_with_order_scope(year, states_tuple, delivered_only=True)
    if reviews is None or reviews.empty:
        return empty

    model = apply_filters(get_model(), year, list(states_tuple))
    if model is None or model.empty:
        return empty
    model = model[model["is_delivered"]][["order_id", "is_late"]]
    if model.empty:
        return empty

    reviews = reviews.merge(model, on="order_id", how="inner")
    reviews = reviews[reviews["review_score"].notna()]
    if reviews.empty:
        return empty

    reviews["status"] = reviews["is_late"].map(
        {True: "Late", False: "On Time"}
    )
    means  = reviews.groupby("status")["review_score"].mean()
    counts = reviews.groupby("status")["review_score"].count()

    on_mean  = float(means["On Time"]) if "On Time" in means.index else None
    late_mean = float(means["Late"])   if "Late"    in means.index else None
    n_on   = int(counts["On Time"]) if "On Time" in counts.index else 0
    n_late = int(counts["Late"])    if "Late"    in counts.index else 0

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
    if not is_partial_year(year):
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

def chart_review_distribution(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Review Score Distribution")
    df = _reviews_with_order_scope(year, states)
    df = df[df["review_score"].notna()] if not df.empty else df
    if df is None or df.empty:
        _no_data(); return

    counts = df["review_score"].round().astype(int).value_counts().sort_index()
    if counts.empty:
        _no_data(); return
    total = int(counts.sum())
    if total == 0:
        _no_data(); return
    pct = 100.0 * counts / total

    shades = [
        PALETTE.success if s == 5
        else PALETTE.danger if s == 1
        else PALETTE.slate_400
        for s in counts.index
    ]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    bars = ax.bar(counts.index.astype(str), counts.values, color=shades,
                  edgecolor=PALETTE.white, linewidth=1.4)
    ax.set_title("Review Score Distribution", pad=14, loc="left")
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Number of reviews")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    _style_axes(ax)
    for b, p, n in zip(bars, pct.values, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{p:.1f}%\n({int(n):,})", ha="center", va="bottom",
                fontsize=10, color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    s5 = float(pct.get(5, 0.0)); s1 = float(pct.get(1, 0.0))
    low_share  = float(pct.loc[pct.index.isin([1, 2])].sum())
    high_share = float(pct.loc[pct.index.isin([4, 5])].sum())
    _insight(
        year, states,
        f"<b>{s5:.1f}%</b> of {total:,} reviews are 5★ while only "
        f"<b>{s1:.1f}%</b> are 1★ — promoters (4–5★) hold "
        f"<b>{high_share:.1f}%</b> vs detractors (1–2★) at "
        f"<b>{low_share:.1f}%</b>."
    )


def chart_review_volume_by_year(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Review Volume Share by Year")
    df = _reviews_with_order_scope(year, states)
    if df is not None and not df.empty:
        df = df[df["review_score"].notna() & df["year"].notna()]
    if df is None or df.empty:
        _no_data(); return

    counts = df.groupby("year").size().sort_index()
    if counts.empty:
        _no_data(); return
    total = int(counts.sum())
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
            "months (Sep–Dec) and 2018 ends on 17 October. These are "
            "**not full calendar years**, so their review volume appears "
            "artificially low."
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
# §6.x  PUNCTUALITY  (Q6, Q9)
# ════════════════════════════════════════════════════════════════════════════

def _late_severity(days_late: pd.Series) -> pd.Series:
    out = pd.Series("On Time", index=days_late.index, dtype="object")
    out = out.mask(days_late.between(1, 3), "Slightly Late")
    out = out.mask(days_late.between(4, 7), "Moderately Late")
    out = out.mask(days_late >= 8, "Severely Late")
    return out


def _welch_ci(a: pd.Series, b: pd.Series, alpha: float = 0.05
              ) -> tuple[float, float, float, float, float]:
    nan = float("nan")
    try:
        import numpy as np
    except Exception:
        return nan, nan, nan, nan, nan
    try:
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return nan, nan, nan, nan, nan
        diff = float(a.mean() - b.mean())
        try:
            from scipy import stats  # type: ignore
        except Exception:
            return diff, nan, nan, nan, nan
        t, _p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
        se = float(np.sqrt(va / na + vb / nb))
        denom = ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        df_w = (va / na + vb / nb) ** 2 / denom if denom > 0 else nan
        tcrit = float(stats.t.ppf(1 - alpha / 2, df_w)) if df_w == df_w else nan
        ci_low  = diff - tcrit * se
        ci_high = diff + tcrit * se
        pooled = (float(np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)))
                  if (na + nb - 2) > 0 else nan)
        d = diff / pooled if pooled and pooled > 0 else nan
        return diff, ci_low, ci_high, float(t), float(d)
    except Exception:
        return nan, nan, nan, nan, nan


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
    st.markdown("### Delivery Punctuality vs Satisfaction")
    st.caption(
        "Row-level on-time vs late anchors from Q6 · severity tiers use "
        "order-grain means for the dose–response drill-down."
    )

    q6 = query_headline_gap(year, states)
    if q6["gap"] is None:
        _no_data("No delivered orders with both a delivery date and a review.")
        return

    model = apply_filters(get_model(), year, list(states))
    if model is None or model.empty:
        _no_data(); return
    model = model[
        model["is_delivered"]
        & model["review_score_order"].notna()
        & model["days_late"].notna()
    ].copy()
    if model.empty:
        _no_data(); return
    model["severity"] = _late_severity(model["days_late"])
    model["is_detractor"] = model["review_score_order"].le(2)

    tier_order = ["On Time", "Slightly Late", "Moderately Late", "Severely Late"]
    grp = (
        model.groupby("severity", observed=False)
             .agg(
                 orders=("order_id", "nunique"),
                 avg_score=("review_score_order", "mean"),
                 detractor_rate=("is_detractor", "mean"),
                 avg_days_late=("days_late", "mean"),
             )
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
    ax1.set_title("Mean score by lateness severity (order-grain)",
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

    sel_avg = _safe_mean(model["review_score_order"]) or 0.0
    ax1.axhline(sel_avg, color=PALETTE.slate_500, linestyle="--",
                linewidth=1.1, label=f"Selection avg {sel_avg:.2f}★")
    ax1.legend(loc="lower left", frameon=False, fontsize=9)

    dr_max = float((grp["detractor_rate"] * 100).max()) if len(grp) else 0.0
    ymax = max(5.0, dr_max * 1.25)
    bars2 = ax2.bar(grp.index.astype(str), grp["detractor_rate"] * 100,
                    color=colors, edgecolor=PALETTE.white, linewidth=1.4,
                    width=0.62)
    ax2.set_title("1–2★ detractor share by severity", pad=10, loc="left",
                  fontsize=12)
    ax2.set_ylabel("% of reviews that are 1–2★")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylim(0, ymax)
    _style_axes(ax2)
    for b, p, reliable in zip(bars2, grp["detractor_rate"] * 100,
                              grp["reliable"]):
        label = f"{p:.1f}%"
        if not reliable: label += " ⚠️"
        ax2.text(b.get_x() + b.get_width() / 2, p + ymax * 0.02,
                 label, ha="center", va="bottom", fontsize=9,
                 color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    on_scores   = model.loc[model["severity"] == "On Time", "review_score_order"]
    late_scores = model.loc[model["severity"] != "On Time", "review_score_order"]
    n_on, n_late = len(on_scores), len(late_scores)

    if n_on >= PUNCT_MIN_N_STATUS and n_late >= PUNCT_MIN_N_STATUS:
        diff, ci_lo, ci_hi, t_stat, d = _welch_ci(on_scores, late_scores)
        robust = (ci_lo == ci_lo and ci_hi == ci_hi and ci_lo > 0)
        sig = ("✅ statistically robust" if robust
               else "⚠️ CI includes 0 — inconclusive")
        d_txt = (f"{d:.2f} ({_cohens_verdict(d)})"
                 if d == d else "n/a (scipy unavailable)")
        ci_txt = (f"[{ci_lo:+.2f}, {ci_hi:+.2f}]"
                  if ci_lo == ci_lo else "n/a")
        verdict = (
            f"<b>On-Time − Late gap: {diff:+.2f}★</b> "
            f"(95% CI {ci_txt}) · Welch t = {t_stat:.2f} · "
            f"Cohen's d = {d_txt} · n_on={n_on:,}, n_late={n_late:,} · {sig}"
        )
    else:
        verdict = (
            f"<b>Insufficient sample for a statistical test.</b> "
            f"On-time n={n_on:,}, late n={n_late:,}; need ≥ "
            f"{PUNCT_MIN_N_STATUS} each. Widen the Year or State filters."
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

    st.markdown("#### 🎛️ What-if: convert late orders to on-time")
    if n_late >= PUNCT_MIN_N_STATUS and n_on >= PUNCT_MIN_N_STATUS:
        pct = st.slider(
            "Percentage of late orders made on-time",
            min_value=0, max_value=100, value=10, step=5,
            key=f"whatif_{year}_{len(states)}",
            help="Projects the effect on the selection average. "
                 "Assumes converted orders land at the on-time mean.",
        )
        on_mean   = _safe_mean(on_scores)   or 0.0
        late_mean = _safe_mean(late_scores) or 0.0
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

    if n_late >= PUNCT_MIN_N_STATUS and n_on >= PUNCT_MIN_N_STATUS:
        on_mean   = _safe_mean(on_scores)   or 0.0
        late_mean = _safe_mean(late_scores) or 0.0
        new_total = n_on + n_late
        scen_rows = []
        for p in PUNCT_SCENARIOS:
            conv = n_late * p / 100.0
            nm = ((n_on * on_mean
                   + (n_late - conv) * late_mean
                   + conv * on_mean) / new_total
                  if new_total > 0 else sel_avg)
            scen_rows.append({
                "Scenario":           f"{p}% of late → on-time",
                "Late orders fixed":  f"{conv:,.0f}",
                "Expected avg score": f"{nm:.3f}",
                "Δ vs current":       f"{nm - sel_avg:+.3f}★",
            })
        with st.expander("📋 Full sensitivity table (5%, 10%, 25%, 50%)",
                         expanded=False):
            _st_dataframe(pd.DataFrame(scen_rows))
            lift_1pct = ((n_late * 0.01 * (on_mean - late_mean)) / new_total
                         if new_total else 0.0)
            st.caption(
                f"Stars gained per 1% of late orders converted = "
                f"**{lift_1pct:+.4f}★**."
            )

    q6_gap_txt = f"{q6['gap']:.2f}★"
    q6_gap_pct = (f"{q6['gap_pct']:.1f}%"
                  if q6["gap_pct"] is not None else "—")
    late_share = float(grp.loc[grp.index != "On Time", "tier_share"].sum()) \
        if len(grp) else 0.0
    noisy = int((~grp["reliable"]).sum()) if len(grp) else 0
    noisy_txt = (f" {noisy} tier(s) flagged ⚠️ (n < {PUNCT_MIN_N_TIER})."
                 if noisy else "")
    top_det = grp["detractor_rate"].idxmax() if len(grp) else grp.index[0]
    on_time_dr = (float(grp.loc["On Time", "detractor_rate"] * 100)
                  if "On Time" in grp.index else 0.0)
    top_dr = float(grp.loc[top_det, "detractor_rate"] * 100) if len(grp) else 0.0

    _insight(
        year, states,
        f"<b>Q6 anchor:</b> on-time <b>{q6['on_time']:.2f}★</b> "
        f"(n={q6['n_on']:,}) vs late <b>{q6['late']:.2f}★</b> "
        f"(n={q6['n_late']:,}) — gap <b>{q6_gap_txt}</b> "
        f"(<b>{q6_gap_pct}</b>) across <b>{late_share:.1f}%</b> of orders "
        f"that arrived late. 1–2★ share jumps from "
        f"<b>{on_time_dr:.1f}%</b> to <b>{top_dr:.1f}%</b> at peak "
        f"severity.{noisy_txt}"
    )
    _partial_note(year)


def chart_satisfaction_by_delivery_bucket(year: str,
                                          states: tuple[str, ...]) -> None:
    st.markdown("### Satisfaction by Delivery-Time Bucket")

    df = _reviews_with_order_scope(year, states, delivered_only=True)
    if df is None or df.empty:
        _no_data(); return

    model = apply_filters(get_model(), year, list(states))
    if model is None or model.empty:
        _no_data(); return
    model = model[model["is_delivered"]][["order_id", "delivery_days"]]
    if model.empty:
        _no_data(); return

    df = df.merge(model, on="order_id", how="inner")
    df = df[df["review_score"].notna() & df["delivery_days"].notna()]
    if df.empty:
        _no_data(); return

    bins = [0, 5, 10, 20, 30, float("inf")]
    labels = ["0-5 days", "6-10 days", "11-20 days", "21-30 days", "30+ days"]
    df = df.copy()
    try:
        df["bucket"] = pd.cut(df["delivery_days"], bins=bins,
                              labels=labels, include_lowest=True)
    except Exception:
        _no_data(); return
    df = df[df["bucket"].notna()]
    if df.empty:
        _no_data(); return

    grp = df.groupby("bucket", observed=False).agg(
        orders=("order_id", "nunique"),
        reviews=("review_score", "count"),
        avg_score=("review_score", "mean"),
    ).reset_index()
    grp = grp[grp["orders"] > 0]
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
              for b in grp["bucket"]]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    bars = ax.bar(grp["bucket"].astype(str), grp["avg_score"],
                  color=colors, edgecolor=PALETTE.white, linewidth=1.4)
    ax.set_title("Satisfaction by Delivery-Time Bucket (row-level AVG)",
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

    if len(grp) == 1:
        text = (f"All delivered orders sit in the "
                f"<b>{grp.iloc[0]['bucket']}</b> bucket "
                f"(<b>100.0%</b>, {grp.iloc[0]['avg_score']:.2f}★).")
    else:
        first, last = grp.iloc[0], grp.iloc[-1]
        below = grp.loc[grp["avg_score"] < sel_avg]
        below_share = float(below["share"].sum())
        below_txt = (
            f"<b>{below_share:.1f}%</b> of orders sit in buckets below avg "
            f"({', '.join(map(str, below['bucket']))})."
            if not below.empty else
            "No bucket falls below the selection average."
        )
        fast_share = float(grp.loc[grp["bucket"].astype(str).isin(
            ["0-5 days", "6-10 days"]), "share"].sum())
        slow_share = float(grp.loc[grp["bucket"].astype(str).isin(
            ["21-30 days", "30+ days"]), "share"].sum())
        text = (f"Scores drop <b>{first['avg_score']:.2f}★</b> "
                f"({first['bucket']}, <b>{fast_share:.1f}%</b> of orders) → "
                f"<b>{last['avg_score']:.2f}★</b> ({last['bucket']}, "
                f"<b>{slow_share:.1f}%</b>); satisfaction breaks sharply "
                f"after 20 days. {below_txt}")
    _insight(year, states, text)


# ════════════════════════════════════════════════════════════════════════════
# §6.y  GEOGRAPHY  (Q4, Q5)
# ════════════════════════════════════════════════════════════════════════════

def chart_state_scores(year: str, states: tuple[str, ...]) -> None:
    df = _reviews_with_order_scope(year, states)
    if df is not None and not df.empty:
        df = df[df["review_score"].notna() & df["customer_state"].notna()]
    if df is None or df.empty:
        st.markdown("### Average Review Score by State *(top 10)*")
        _no_data(); return

    grp = (df.groupby("customer_state")
             .agg(
                 orders=("order_id", "nunique"),
                 reviews=("review_score", "count"),
                 avg_score=("review_score", "mean"),
             )
             .reset_index()
             .sort_values("orders", ascending=False)
             .head(10))
    if grp.empty:
        _no_data(); return
    total_orders = float(grp["orders"].sum())
    if total_orders <= 0:
        _no_data(); return
    grp["share"] = 100.0 * grp["orders"] / total_orders

    n_states  = len(grp)
    title_cap = ("top 10" if n_states >= 10
                 else f"{n_states} state{'s' if n_states != 1 else ''}")
    scope_txt = "in current selection" if states else "by orders"
    st.markdown(f"### Average Review Score by State *({title_cap} {scope_txt})*")

    plot_df = grp.sort_values("avg_score", ascending=True)
    if plot_df.empty:
        _no_data(); return

    colors = [
        PALETTE.danger if s < 4.00
        else PALETTE.warn if s < 4.10
        else PALETTE.success
        for s in plot_df["avg_score"]
    ]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.barh(plot_df["customer_state"], plot_df["avg_score"],
                   color=colors, edgecolor=PALETTE.white, linewidth=1.4)
    ax.set_title(
        f"Average Review Score by State ({title_cap} {scope_txt})",
        pad=14, loc="left",
    )
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

    if len(plot_df) == 1:
        text = (f"<b>{plot_df.iloc[0]['customer_state']}</b> averages "
                f"{plot_df.iloc[0]['avg_score']:.2f}★ "
                f"(<b>100.0%</b> of orders).")
    else:
        low, high = plot_df.iloc[0], plot_df.iloc[-1]
        spread = high["avg_score"] - low["avg_score"]
        spread_pct = (100.0 * _safe_div(spread, high["avg_score"])
                      if _safe_div(spread, high["avg_score"]) is not None
                      else 0.0)
        weak = plot_df.loc[plot_df["avg_score"] < 4.10]
        weak_share = float(weak["share"].sum())
        text = (f"<b>{low['customer_state']}</b> "
                f"({low['avg_score']:.2f}★, {low['share']:.1f}% of orders) "
                f"anchors the low end and <b>{high['customer_state']}</b> "
                f"({high['avg_score']:.2f}★, {high['share']:.1f}%) the high "
                f"end — a {spread:.2f}★ spread (<b>{spread_pct:.1f}%</b>) "
                f"dwarfed by the on-time↔late gap. "
                f"<b>{weak_share:.1f}%</b> of orders sit in states scoring "
                f"&lt; 4.10★.")
    _insight(year, states, text)


def chart_ontime_by_state(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### On-Time vs Late by State *(top 5)*")
    df = apply_filters(get_model(), year, list(states))
    if df is None or df.empty:
        _no_data(); return
    df = df[df["is_delivered"] & df["customer_state"].notna()]
    if df.empty:
        _no_data(); return

    grp = (df.groupby("customer_state")
             .agg(
                 delivered=("order_id", "nunique"),
                 late=("is_late", "sum"),
             )
             .reset_index()
             .sort_values("delivered", ascending=False)
             .head(5))
    grp = grp[grp["delivered"] > 0]
    if grp.empty:
        _no_data(); return
    grp["on_time"]  = grp["delivered"] - grp["late"]
    grp["late_pct"] = 100.0 * grp["late"] / grp["delivered"]

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
    worst_mx = (worst["late_pct"] / overall_pct
                if overall_pct else float("nan"))
    _insight(
        year, states,
        f"Top-5 states average <b>{overall_pct:.1f}%</b> late; "
        f"<b>{worst['customer_state']}</b> is worst at "
        f"<b>{worst['late_pct']:.1f}%</b> (<b>{worst_mx:.1f}×</b> group avg) "
        f"and <b>{best['customer_state']}</b> best at "
        f"<b>{best['late_pct']:.1f}%</b> — a "
        f"<b>{worst['late_pct'] - best['late_pct']:.1f} pp</b> spread."
    )


# ════════════════════════════════════════════════════════════════════════════
# §6.z  MONTHLY TREND  (Q3)
# ════════════════════════════════════════════════════════════════════════════

def chart_monthly_score(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Monthly Average Review Score")
    df = _reviews_with_order_scope(year, states)
    if df is not None and not df.empty:
        df = df[df["review_score"].notna() & df["month"].notna()]
    if df is None or df.empty:
        _no_data(); return

    grp = (df.groupby("month")
             .agg(
                 reviews=("review_score", "count"),
                 orders=("order_id", "nunique"),
                 avg_score=("review_score", "mean"),
             )
             .reset_index()
             .sort_values("month"))
    if grp.empty:
        _no_data(); return

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE_WIDE)
    ax.plot(grp["month"], grp["avg_score"], marker="o",
            color=PALETTE.primary, linewidth=2.4, markersize=6,
            markerfacecolor=PALETTE.white, markeredgewidth=1.8)
    ax.set_title("Monthly Average Review Score (row-level AVG)",
                 pad=14, loc="left")
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
    note = (
        f" <b>{noisy_pct:.1f}%</b> of months have < "
        f"{CONFIG.min_reviews_per_month} reviews — noisy averages."
    ) if noisy else ""
    swing = grp.loc[high_i, "avg_score"] - grp.loc[low_i, "avg_score"]
    swing_pct = (100.0 * _safe_div(swing, grp.loc[high_i, "avg_score"])
                 if _safe_div(swing, grp.loc[high_i, "avg_score"]) is not None
                 else 0.0)
    _insight(
        year, states,
        f"Lowest <b>{grp.loc[low_i,'month']}</b> "
        f"({grp.loc[low_i,'avg_score']:.2f}★) vs highest "
        f"<b>{grp.loc[high_i,'month']}</b> "
        f"({grp.loc[high_i,'avg_score']:.2f}★) — a "
        f"<b>{swing:.2f}★</b> (<b>{swing_pct:.1f}%</b>) seasonal swing "
        f"that stays small next to the late-delivery gap.{note}"
    )
    _partial_note(year)


# ════════════════════════════════════════════════════════════════════════════
# §6.zz  MAP
# ════════════════════════════════════════════════════════════════════════════

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
        sigla = (
            props.get("sigla") or props.get("SIGLA")
            or props.get("abbrev") or props.get("code")
            or props.get("name")
        )
        if not sigla:
            continue
        geom = feat.get("geometry", {}) or {}
        if not isinstance(geom, dict):
            continue
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
    df = get_model()
    if df is None or df.empty:
        _no_data(); return
    df = df[df["is_delivered"] & df["customer_state"].notna()]
    if year != "All":
        df = df[df["year"] == year]
    if df.empty:
        _no_data("No state-level data for this year."); return
    counts = df.groupby("customer_state")["order_id"].nunique()
    if counts.empty:
        _no_data("No state-level data for this year."); return

    geojson = load_brazil_geojson()
    if not geojson:
        st.warning("⚠️ Brazil map unavailable (GeoJSON fetch failed).")
        return

    gdf = _geojson_to_gdf(geojson)
    if gdf.empty:
        st.warning("⚠️ Could not parse state geometries.")
        return

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
    ax.set_title(f"Delivered Orders by State — {scope_label(year, states)}",
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
        share = (100.0 * n / total_orders
                 if (n is not None and total_orders) else None)
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
    if not in_scope_pairs:
        _insight(year, states, "No states match the current filter.")
        return
    lo_s, lo_v = min(in_scope_pairs, key=lambda x: x[1])
    hi_s, hi_v = max(in_scope_pairs, key=lambda x: x[1])
    hi_pct = (100.0 * _safe_div(hi_v, total_orders)
              if _safe_div(hi_v, total_orders) is not None else 0.0)
    lo_pct = (100.0 * _safe_div(lo_v, total_orders)
              if _safe_div(lo_v, total_orders) is not None else 0.0)
    top3 = sorted(in_scope_pairs, key=lambda x: x[1], reverse=True)[:3]
    top3_pct = (100.0 * _safe_div(sum(v for _, v in top3), total_orders)
                if _safe_div(sum(v for _, v in top3), total_orders) is not None
                else 0.0)
    mode = ("Selected states coloured; rest greyed."
            if filter_active else
            "All 27 states coloured by delivered count.")
    _insight(
        year, states,
        f"<b>{hi_s}</b> carries the largest volume ({int(hi_v):,}, "
        f"<b>{hi_pct:.1f}%</b>); <b>{lo_s}</b> the smallest "
        f"({int(lo_v):,}, <b>{lo_pct:.1f}%</b>). Top-3 states concentrate "
        f"<b>{top3_pct:.1f}%</b> of deliveries. {mode}"
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
    scope = scope_label(year, states)
    if g["gap"] is None:
        body = ("Delivery punctuality is the strongest satisfaction driver "
                "in this dataset. Only one delivery status appears in the "
                "current selection — widen the filters to compare on-time "
                "and late orders.")
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
              ({gap_pct_txt}) — the strongest satisfaction driver. See the
              Customer Satisfaction tab for the evidence.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
# §8  GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS: Final[str] = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">

<style>
  :root {
    --brand:        #6366F1;
    --brand-light:  #A5B4FC;
    --brand-dark:   #4338CA;
    --accent:       #F97316;
    --accent-light: #FDBA74;
    --success:      #10B981;
    --success-dk:   #047857;
    --warn:         #F59E0B;
    --warn-dk:      #B45309;
    --danger:       #EF4444;
    --danger-dk:    #B91C1C;
    --ink:          #0F172A;
    --ink-soft:     #1E293B;
    --slate-700:    #334155;
    --slate-600:    #475569;
    --slate-500:    #64748B;
    --slate-400:    #94A3B8;
    --slate-300:    #CBD5E1;
    --slate-200:    #E2E8F0;
    --slate-100:    #F1F5F9;
    --slate-50:     #F8FAFC;
    --s1: 4px;  --s2: 8px;  --s3: 12px; --s4: 16px;
    --s5: 24px; --s6: 32px; --s7: 48px; --s8: 64px;
    --fs-hero:    clamp(2rem, 1.7rem + 1.5vw, 2.75rem);
    --fs-h1:      clamp(1.35rem, 1.1rem + 1vw, 1.85rem);
    --fs-h2:      clamp(1.15rem, 1.05rem + 0.5vw, 1.4rem);
    --fs-h3:      clamp(1rem, 0.95rem + 0.3vw, 1.15rem);
    --fs-body:    0.95rem;
    --fs-small:   0.82rem;
    --fs-micro:   0.72rem;
    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  14px;
    --radius-xl:  20px;
    --shadow-sm:  0 1px 2px rgba(15,23,42,0.04),
                  0 1px 3px rgba(15,23,42,0.06);
    --shadow-md:  0 4px 6px rgba(15,23,42,0.04),
                  0 10px 24px rgba(15,23,42,0.06);
    --shadow-brand: 0 4px 12px rgba(99,102,241,0.22);
  }

  html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: 15px;
    line-height: 1.65;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  .stApp { background: var(--slate-50); }

  h1, h2, h3, .app-hero-title, .hero-headline, .exec-summary-title,
  .hero-label, .exec-summary-eyebrow {
    font-family: 'Sora', 'Inter', sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--ink);
  }
  code, .stCodeBlock { font-family: 'JetBrains Mono', monospace !important; }

  .app-hero {
    display: flex;
    align-items: center;
    gap: var(--s6);
    background: linear-gradient(135deg, #EEF2FF 0%, #FFFFFF 60%, #FFF7ED 100%);
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-xl);
    padding: var(--s6) var(--s7);
    margin-bottom: var(--s5);
    box-shadow: 0 8px 32px rgba(99,102,241,0.10);
  }
  .app-hero-logo {
    flex-shrink: 0; width: 110px; height: auto; object-fit: contain;
  }
  .app-hero-content { flex: 1; min-width: 0; }
  .app-hero-eyebrow {
    font-size: 0.75rem; font-weight: 800;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--brand); margin-bottom: var(--s3);
    font-family: 'Sora', sans-serif !important;
  }
  .app-hero-title {
    font-size: var(--fs-hero) !important; font-weight: 800 !important;
    color: var(--ink) !important; line-height: 1.1 !important;
    margin: 0 0 var(--s3) 0 !important; padding: 0 !important;
    letter-spacing: -0.03em !important; display: block;
  }
  .app-hero-title b {
    color: var(--brand-dark);
    background: linear-gradient(120deg, var(--brand) 0%, var(--accent) 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  @supports not ((-webkit-background-clip: text) or (background-clip: text)) {
    .app-hero-title b {
      color: var(--brand) !important; background: none !important;
      -webkit-text-fill-color: var(--brand) !important;
    }
  }
  .app-hero-sub {
    font-size: 1rem; color: var(--slate-600);
    margin: 0 0 var(--s4) 0; line-height: 1.6; max-width: 760px;
  }
  .app-hero-meta {
    display: flex; gap: var(--s2); flex-wrap: wrap; align-items: center;
  }

  h1 { color: var(--ink); font-weight: 800; font-size: var(--fs-h1);
       line-height: 1.2; margin-bottom: var(--s3); }
  h2 { color: var(--ink); font-weight: 700; font-size: var(--fs-h2);
       line-height: 1.3; margin-top: var(--s5) !important;
       margin-bottom: var(--s4) !important; }
  h3 { color: var(--ink-soft); font-weight: 700; font-size: var(--fs-h3);
       line-height: 1.35; margin-top: 0 !important;
       margin-bottom: var(--s3) !important; }
  h4 { color: var(--brand); font-size: 1rem; font-weight: 700;
       line-height: 1.4; margin-bottom: var(--s3) !important; }
  p, li, span, label { color: var(--slate-700); font-size: var(--fs-body); }
  .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--slate-500) !important;
    font-size: var(--fs-small) !important;
    line-height: 1.55;
  }

  .block-container {
    padding-top: var(--s5) !important;
    padding-bottom: var(--s8) !important;
    max-width: 1480px;
    padding-left: var(--s5) !important;
    padding-right: var(--s5) !important;
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
    border-right: 1px solid #312E81;
    padding-top: var(--s4);
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
    border-color: rgba(255,255,255,0.08) !important;
    margin: var(--s4) 0;
  }
  section[data-testid="stSidebar"] table { color: #E2E8F0 !important; }
  section[data-testid="stSidebar"] th,
  section[data-testid="stSidebar"] td {
    color: #E2E8F0 !important;
    border-color: #312E81 !important;
    font-size: 0.78rem !important;
    padding: 6px 10px !important;
  }
  section[data-testid="stSidebar"] .stButton > button {
    background: var(--brand) !important; color: #FFFFFF !important;
    border-radius: var(--radius-md); font-weight: 600;
    border: none; padding: 8px 14px;
    transition: background 0.15s ease, transform 0.1s ease;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--brand-dark) !important;
    transform: translateY(-1px);
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex; flex-direction: column; gap: var(--s1);
    padding-top: var(--s1);
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(165,180,252,0.16) !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 14px !important;
    transition: background 0.15s ease, border-color 0.15s ease;
    cursor: pointer;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(165,180,252,0.12) !important;
    border-color: rgba(165,180,252,0.38) !important;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #E2E8F0 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--brand) 0%, var(--brand-dark) 100%) !important;
    border-color: var(--brand-light) !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.45);
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #FFFFFF !important; font-weight: 700 !important;
  }

  .sidebar-warning {
    background: rgba(245,158,11,0.14);
    border: 1px solid rgba(245,158,11,0.42);
    border-left: 3px solid #F59E0B;
    border-radius: var(--radius-md);
    padding: 10px 12px;
    margin-top: var(--s3);
    font-size: var(--fs-small);
    line-height: 1.5;
  }
  .sidebar-warning strong { color: #FCD34D !important; display: block; margin-bottom: 4px; }
  .sidebar-warning span   { color: #E2E8F0 !important; }

  .sidebar-headline {
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(165,180,252,0.42);
    border-left: 3px solid var(--brand-light);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    margin-top: var(--s4);
    line-height: 1.55;
  }
  .sidebar-headline-label {
    font-size: 0.65rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--brand-light) !important;
    margin-bottom: var(--s2);
  }
  .sidebar-headline strong {
    color: #FFFFFF !important; font-size: var(--fs-small); display: block;
    margin-bottom: 4px;
  }
  .sidebar-headline span { color: #CBD5E1 !important; font-size: 0.74rem; }
  .sidebar-headline span b { color: #FFFFFF !important; }

  [data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--slate-200);
    border-left: 5px solid var(--brand);
    border-radius: var(--radius-lg);
    padding: var(--s5);
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease,
                border-color 0.2s ease;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-left-color: var(--accent);
  }
  [data-testid="stMetricLabel"] {
    color: var(--slate-500) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  [data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-weight: 800 !important;
    font-size: clamp(1.4rem, 1.2rem + 0.6vw, 1.85rem) !important;
    letter-spacing: -0.025em;
    line-height: 1.15;
    font-family: 'Sora', 'Inter', sans-serif !important;
    font-variant-numeric: tabular-nums;
    margin-top: 4px;
  }
  [data-testid="stMetricDelta"] {
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    margin-top: 4px;
  }

  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-lg) !important;
    padding: var(--s5) var(--s6) !important;
    box-shadow: var(--shadow-sm);
    margin-bottom: var(--s4);
    transition: box-shadow 0.2s ease;
  }
  [data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: var(--shadow-md);
  }
  [data-testid="stVerticalBlockBorderWrapper"] h3 {
    margin-top: 0 !important;
    margin-bottom: var(--s3) !important;
  }

  .stButton > button {
    background: var(--brand); color: #FFFFFF;
    border: none; border-radius: var(--radius-md);
    padding: 10px 20px; font-weight: 600;
    font-size: var(--fs-body);
    transition: background 0.15s ease, transform 0.1s ease,
                box-shadow 0.15s ease;
  }
  .stButton > button:hover {
    background: var(--brand-dark);
    transform: translateY(-1px);
    box-shadow: var(--shadow-brand);
  }

  [data-testid="stAlert"] {
    background-color: #FEF3C7;
    border-left: 4px solid var(--warn);
    color: #78350F;
    border-radius: var(--radius-md);
    padding: var(--s3) var(--s4);
  }
  [data-testid="stExpander"] {
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-md);
    background: #FFFFFF;
    margin-bottom: var(--s3);
  }
  [data-testid="stExpander"] summary {
    font-weight: 600; color: var(--brand);
    font-family: 'Sora', sans-serif !important;
    font-size: var(--fs-body);
  }
  .stSlider label {
    font-weight: 600 !important;
    color: var(--slate-700) !important;
    font-size: var(--fs-small) !important;
  }

  .info-card {
    position: relative; background: #FFFFFF;
    border: 1px solid var(--slate-200);
    border-left: 5px solid var(--brand);
    border-radius: var(--radius-lg);
    padding: var(--s5);
    margin-bottom: var(--s4);
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  .info-card:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
  }
  .info-card.warn { border-left-color: var(--danger); }
  .info-card.ok   { border-left-color: var(--success); }
  .info-card.info { border-left-color: var(--brand); }
  .info-card.warn h4 { color: var(--danger); }
  .info-card.ok h4   { color: var(--success); }
  .info-card.info h4 { color: var(--brand); }
  .info-card p, .info-card li {
    color: var(--slate-700); line-height: 1.7;
    font-size: var(--fs-body);
  }
  .info-card .tag {
    display: inline-block; background: var(--slate-100);
    color: var(--slate-600); border-radius: 999px;
    padding: 4px 12px; font-size: 0.7rem;
    font-weight: 600; margin-right: var(--s2); margin-bottom: var(--s2);
    letter-spacing: 0.03em;
  }
  .priority-bar {
    position: absolute; top: 0; right: 0;
    width: 5px; height: 100%;
    border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
  }
  .priority-high   { background: var(--danger); }
  .priority-medium { background: var(--warn); }
  .priority-low    { background: var(--success); }

  .hero-banner, .headline-strip {
    display: flex; align-items: center; gap: var(--s3);
    border-radius: var(--radius-lg);
    padding: var(--s4) var(--s5);
    margin: var(--s4) 0;
  }
  .hero-banner {
    background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 55%, #C7D2FE 100%);
    border: 1px solid #A5B4FC;
    border-left: 4px solid var(--brand);
    box-shadow: 0 2px 12px rgba(99,102,241,0.14);
  }
  .headline-strip {
    background: linear-gradient(90deg, #EEF2FF 0%, #FFFFFF 100%);
    border: 1px solid #C7D2FE;
    border-left: 4px solid var(--brand);
  }
  .hero-banner .hero-icon, .headline-strip-icon {
    font-size: 22px; line-height: 1; flex-shrink: 0;
  }
  .hero-banner .hero-text,
  .headline-strip .headline-strip-text { flex: 1; min-width: 0; }
  .hero-banner .hero-label,
  .headline-strip .headline-strip-label {
    display: block; font-size: 0.68rem; font-weight: 800;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--brand); margin-bottom: 4px;
  }
  .hero-banner .hero-headline,
  .headline-strip-text p {
    font-size: var(--fs-body); font-weight: 500;
    color: var(--slate-700); line-height: 1.6;
    margin: 0; letter-spacing: normal;
  }
  .hero-banner .hero-headline b,
  .headline-strip-text b {
    color: var(--brand-dark); font-weight: 700;
  }

  .exec-summary {
    display: flex; align-items: stretch; gap: var(--s5);
    background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 100%);
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: var(--s5) var(--s6);
    margin: var(--s4) 0;
    box-shadow: 0 4px 16px rgba(99,102,241,0.08);
  }
  .exec-summary-main { flex: 1; min-width: 0; }
  .exec-summary-eyebrow {
    font-size: 0.7rem; font-weight: 800;
    letter-spacing: 0.16em; color: var(--brand);
    margin-bottom: var(--s2);
  }
  .exec-summary-title {
    font-size: var(--fs-h2); font-weight: 800;
    color: var(--ink); line-height: 1.3;
    margin-bottom: var(--s3);
  }
  .exec-summary-title b { color: var(--brand-dark); }
  .exec-summary-detail {
    color: var(--slate-600); font-size: var(--fs-body);
    line-height: 1.65;
  }
  .exec-summary-badge {
    display: flex; flex-direction: column; justify-content: center;
    align-items: center; text-align: center;
    min-width: 140px; padding: var(--s4) var(--s5);
    border: 2px solid; border-radius: var(--radius-lg);
    flex-shrink: 0;
  }
  .exec-summary-badge-label {
    font-size: 0.65rem; font-weight: 800;
    letter-spacing: 0.14em; text-transform: uppercase;
    opacity: 0.85; margin-bottom: var(--s2); color: inherit;
  }
  .exec-summary-badge-value {
    font-family: 'Sora', sans-serif; font-weight: 800;
    font-size: 1.05rem; line-height: 1.2; color: inherit;
  }

  .overview-banner {
    background: #FEF3C7; border-left: 4px solid var(--warn);
    border-radius: var(--radius-lg); padding: var(--s4) var(--s5);
    margin-bottom: var(--s4);
    font-size: var(--fs-body); color: #78350F; line-height: 1.6;
  }
  .overview-banner b { color: #92400E; }

  .scope-pill {
    display: inline-block;
    background: linear-gradient(135deg, var(--brand) 0%, var(--accent) 100%);
    color: #FFFFFF; border-radius: 999px;
    padding: 6px 14px; font-size: var(--fs-small); font-weight: 600;
    box-shadow: 0 2px 10px rgba(99,102,241,0.28);
  }
  .scope-pill-subtle {
    display: inline-block; background: var(--slate-100);
    color: var(--slate-700); border: 1px solid var(--slate-200);
    border-radius: 999px; padding: 6px 14px;
    font-size: var(--fs-small); font-weight: 600;
  }

  .insight {
    display: flex; gap: var(--s3); align-items: flex-start;
    background: #F5F3FF; border-left: 3px solid var(--brand-light);
    border-radius: var(--radius-md);
    padding: var(--s3) var(--s4);
    margin-top: var(--s3);
    font-size: var(--fs-small);
  }
  .insight-chip {
    flex-shrink: 0;
    background: var(--brand); color: #FFFFFF;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.68rem; font-weight: 700; white-space: nowrap;
  }
  .insight-body { display: flex; flex-direction: column; gap: 2px; }
  .insight-scope {
    font-size: 0.66rem; font-weight: 700; color: var(--brand);
    text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.9;
  }
  .insight-text { color: var(--slate-600); line-height: 1.6; }
  .insight-text b { color: var(--ink); font-weight: 700; }

  .verdict-card {
    display: flex; flex-direction: column; gap: 4px;
    background: linear-gradient(135deg, #ECFDF5 0%, #F0FDFA 100%);
    border: 1px solid #6EE7B7; border-left: 4px solid var(--success);
    border-radius: var(--radius-md);
    padding: var(--s3) var(--s4);
    margin: var(--s3) 0;
  }
  .verdict-label {
    font-size: 0.68rem; font-weight: 800;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--success-dk);
  }
  .verdict-text {
    font-size: var(--fs-small); color: var(--slate-700); line-height: 1.6;
  }
  .verdict-text b { color: var(--ink); }

  .empty-state {
    text-align: center; padding: var(--s7) var(--s5);
    background: var(--slate-50);
    border: 1px dashed var(--slate-300);
    border-radius: var(--radius-lg);
  }
  .empty-icon { font-size: 40px; }
  .empty-title {
    font-size: 1.1rem; font-weight: 700; color: var(--slate-700);
    margin: var(--s2) 0; font-family: 'Sora', sans-serif;
  }
  .empty-body { color: var(--slate-500); font-size: var(--fs-body); }
  .empty-hint {
    margin-top: var(--s3); color: var(--brand);
    font-size: var(--fs-small); font-weight: 500;
  }

  [data-testid="stDataFrame"] {
    border-radius: var(--radius-md); overflow: hidden;
    border: 1px solid var(--slate-200);
  }
  .stPyplot { margin-top: var(--s1); }

  hr { border-color: var(--slate-200); margin: var(--s5) 0; }

  @media (max-width: 1100px) {
    .app-hero { flex-direction: column; align-items: flex-start;
                padding: var(--s5); gap: var(--s4); }
    .app-hero-logo { width: 84px; }
    .exec-summary { flex-direction: column; }
    .exec-summary-badge { min-width: auto; }
    [data-testid="stMetric"] { padding: var(--s4); }
  }
  @media (max-width: 768px) {
    .block-container { padding: var(--s3) !important; }
    .app-hero { padding: var(--s5) var(--s4); }
    .app-hero-title { font-size: 1.75rem !important; }
    .hero-banner, .headline-strip {
      flex-direction: column; align-items: flex-start;
    }
    .exec-summary { flex-direction: column; }
    .exec-summary-badge { min-width: auto; }
  }
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
    st.sidebar.markdown(
        '<div class="sidebar-nav-label">Navigation</div>',
        unsafe_allow_html=True,
    )
    active_tab = st.sidebar.radio(
        "Navigation",
        options=NAV_TABS,
        index=0,
        key="active_tab",
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.subheader("🎛️ Filters")
    st.sidebar.caption(
        "Both filters apply to **every KPI and chart**. "
        "Set both to *All* to reproduce the reference values."
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

    if is_partial_year(year):
        st.sidebar.markdown(
            f"""
            <div class="sidebar-warning">
              <strong>⚠️ {year} is a partial year</strong>
              <span>Year-over-year comparisons are misleading. See Detail &amp; Method.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    try:
        states = st.sidebar.multiselect(
            "State",
            options=CONFIG.all_states,
            default=[],
            placeholder="All states",
            help="Pick one or more states. Leaving empty shows all states.",
            key="filter_states",
        ) or []
    except Exception:
        states = []

    states_tuple = tuple(s for s in states if isinstance(s, str))

    if states_tuple or year != "All":
        try:
            st.sidebar.button(
                "↺  Reset filters",
                on_click=_reset_filters,
                use_container_width=True,
            )
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
        f"| Avg Delivery Days | {REF['avg_delivery_days']:.2f} |\n"
        f"| Q6 On Time n | {REF['q6_on_time_orders']:,} |\n"
        f"| Q6 Late n | {REF['q6_late_orders']:,} |"
    )
    return active_tab, year, states_tuple


# ════════════════════════════════════════════════════════════════════════════
# §10  OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

def render_overview(year: str, states: tuple[str, ...],
                    filter_active: bool) -> None:
    st.markdown("## 📈 Overview")
    st.markdown(
        "The shape of the business across the current selection — volume, "
        "sentiment, and year contribution at a glance."
    )

    if is_partial_year(year):
        detail = {
            "2016": "2016 contains only September, October and December — "
                    "November 2016 has zero orders.",
            "2018": "2018 ends on 17 October — it is not a full calendar year.",
        }.get(year, "")
        if detail:
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
    c2.metric(
        "Delivered Orders",
        fmt_int(k["delivered_orders"]),
        delta=(fmt_pct(k["delivered_pct"], 1)
               if k["delivered_pct"] is not None else None),
        delta_color="off",
    )
    if k["avg_review_score"] is None:
        c3.metric("Avg Review Score", "—")
    else:
        delta = (
            f"{k['avg_review_score'] - REF['avg_review_score']:+.2f} vs baseline"
            if filter_active else None
        )
        c3.metric("Avg Review Score", fmt_float(k["avg_review_score"]),
                  delta=delta)
    late_delta = (
        f"{k['late_pct']:.1f}% of delivered"
        if k["late_pct"] is not None else None
    )
    c4.metric("Late Deliveries", fmt_int(k["late_deliveries"]),
              delta=late_delta, delta_color="inverse")
    c5.metric("Avg Delivery Days", fmt_float(k["avg_delivery_days"]))

    if filter_active:
        st.info(
            f"Active filter — **{scope_label(year, states)}**. "
            "Set both filters to All to reproduce the sidebar reference values."
        )

    render_headline_strip(year, states)

    _chart_card(chart_review_distribution, year, states)
    _chart_card(chart_review_volume_by_year, year, states)


# ════════════════════════════════════════════════════════════════════════════
# §11  CUSTOMER SATISFACTION
# ════════════════════════════════════════════════════════════════════════════

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

    # Geographic view — Q4 then Q5 (stacked)
    _chart_card(chart_state_scores, year, states)             # Q4
    _chart_card(chart_ontime_by_state, year, states)          # Q5

    _chart_card(chart_monthly_score, year, states)
    _chart_card(chart_brazil_map, year, states)


# ════════════════════════════════════════════════════════════════════════════
# §12  DETAIL & METHOD
# ════════════════════════════════════════════════════════════════════════════

def render_detail(year: str, states: tuple[str, ...]) -> None:
    st.markdown("## 🔬 Detail & Method")
    st.markdown(
        "The limitations that govern how far you can take these findings — "
        "read before quoting the numbers in a decision."
    )

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
        <b>2017 is the only complete calendar year</b>.</p>
        <p>Year-over-year comparisons treating these as full years
        mislead — the gap is real, do not interpolate.</p>""",
        variant="warn", tags=["Time coverage", "YoY risk"],
    )
    info_card(
        "L3 — Review fan-out (Q10)",
        """<p>The dataset has <b>99,224 review rows across 98,673
        distinct orders</b> — <b>551</b> rows of fan-out (<b>0.6%</b> of
        rows). Every AVG(review_score) in this app uses <b>row-level
        reviews</b> (matching Q2/Q3/Q4/Q6/Q9). Only the severity
        drill-down and the what-if use per-order means — and those are
        clearly labelled as order-grain.</p>""",
        variant="warn", tags=["Data quality", "Fan-out", "Q10"],
    )
    info_card(
        "L4 — Customer identity (Q11)",
        """<p>The <code>customers</code> table has <b>99,441 rows</b> but
        only <b>96,096 distinct <code>customer_unique_id</code></b>
        values (<b>96.6%</b> unique). Any customer count here is
        order-grain; a person-grain figure would use the unique
        identifier.</p>""",
        variant="info", tags=["Identity", "Grain", "Q11"],
    )
    info_card(
        "L5 — Delivered ≠ delivered-with-a-date",
        """<p><b>Q1</b> reports 96,478 delivered orders (any
        <code>status='delivered'</code> row), but <b>Q5</b>, <b>Q6</b>
        and <b>Q9</b> add an
        <code>order_delivered_customer_date IS NOT NULL</code> guard that
        shrinks the base to <b>95,832</b> delivered orders and
        <b>7,661</b> late ones. Q1's "late = 7,826" therefore includes
        <b>165</b> late orders with a NULL delivered date. Both numbers
        are correct for their queries; the app labels which is
        which.</p>""",
        variant="info", tags=["Q1 vs Q5/Q6", "NULL semantics"],
    )
    info_card(
        "L6 — No delivery-distance variable",
        """<p>The geolocation CSV is deliberately excluded. The analysis
        <b>cannot distinguish long-distance transit from carrier
        underperformance</b>. Adding postal-code centroids and a
        great-circle distance would sharpen the RJ and BA
        diagnostics.</p>""",
        variant="info", tags=["Scope", "Distance"],
    )
    info_card(
        "L7 — Dataset and generalisability boundaries",
        """<p>Findings concern Olist in Brazil during 2016–2018. They
        may not generalise to other countries, marketplaces, logistics
        networks or current operating conditions. The choropleth fetches
        a public GeoJSON at runtime, so map rendering depends on external
        availability.</p>""",
        variant="info", tags=["External validity", "Map proxy"],
    )
    info_card(
        "L8 — Small selections are noisy",
        """<p>Narrow selections (one state in one year) can contain only
        a few hundred reviews. Averages from small samples swing widely —
        read the <code>n=</code> labels before drawing conclusions. The
        punctuality chart flags tiers below <b>n=30</b> with ⚠️.</p>""",
        variant="info", tags=["Sample size", "Filters"],
    )
    info_card(
        "L9 — Reconciliation contract",
        """<p><b>Q1</b> = <code>99,441 · 96,478 · 4.09 · 7,826 · 12.56</code>.
        <b>Q2</b> = <code>11.5 / 3.2 / 8.2 / 19.3 / 57.8%</code>.
        <b>Q6</b> = on-time <code>88,171 @ 4.29</code>, late
        <code>7,661 @ 2.57</code>.
        <b>Q7</b> = <code>7,826 · 9.55 · 189</code>.
        <b>Q10</b> = <code>99,224 / 98,673 / 551</code>.
        <b>Q11</b> = <code>99,441 / 96,096</code>.
        All verified in the current code; see the <i>Whole-dataset
        diagnostics</i> table below.</p>""",
        variant="ok", tags=["Reconciliation", "KPI contract"],
    )

    st.divider()
    st.markdown("### Whole-dataset diagnostics (Q7, Q10, Q11)")
    st.caption(
        "Counts computed across the entire dataset — these do not respond "
        "to the filters because they describe the source data itself."
    )

    try:
        frames = load_data()
        reviews = frames["reviews"]
        customers = frames["customers"]

        review_rows = len(reviews)
        distinct_orders = reviews["order_id"].nunique()
        duplicate_rows = review_rows - distinct_orders
        duplicate_pct = (100.0 * _safe_div(duplicate_rows, review_rows)
                         if _safe_div(duplicate_rows, review_rows) is not None
                         else 0.0)
        customer_rows = len(customers)
        unique_customers = customers["customer_unique_id"].nunique()
        unique_pct = (100.0 * _safe_div(unique_customers, customer_rows)
                      if _safe_div(unique_customers, customer_rows) is not None
                      else 0.0)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("**Reviews fan-out (Q10)**")
            _st_dataframe(pd.DataFrame([{
                "review_rows": review_rows,
                "distinct_orders": distinct_orders,
                "duplicate_rows": duplicate_rows,
                "duplicate_pct": f"{duplicate_pct:.2f}%",
            }]))
        with c2:
            st.markdown("**Customer identity (Q11)**")
            _st_dataframe(pd.DataFrame([{
                "customer_rows": customer_rows,
                "unique_customers": unique_customers,
                "unique_pct": f"{unique_pct:.2f}%",
            }]))
    except Exception as exc:
        _LOGGER.warning("Diagnostics block failed: %s", exc)
        st.caption("Diagnostics unavailable for the current dataset.")

    try:
        model = get_model()
        late = model[model["is_late"]]
        if not late.empty:
            dl = late["days_late"]
            all_delivered = int(model["is_delivered"].sum())
            late_share = (100.0 * _safe_div(len(late), all_delivered)
                          if _safe_div(len(late), all_delivered) is not None
                          else 0.0)
            st.markdown("**Days late when late (Q7, whole dataset)**")
            _st_dataframe(pd.DataFrame([{
                "late_orders": len(late),
                "late_share_of_delivered": f"{late_share:.1f}%",
                "avg_days_late": round(float(dl.mean()), 2),
                "worst_days_late": round(float(dl.max()), 2),
            }]))
            st.caption(
                f"When late, orders run **{dl.mean():.2f} days past "
                f"estimate** on average; the worst case is "
                f"**{dl.max():.1f} days**. Late orders are "
                f"**{late_share:.1f}%** of all delivered orders."
            )
    except Exception as exc:
        _LOGGER.warning("Late diagnostics failed: %s", exc)


# ════════════════════════════════════════════════════════════════════════════
# §13  RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════

def render_recommendations() -> None:
    st.markdown("## 💡 Recommendations")
    st.markdown(
        "Prioritised actions following from the headline finding. "
        "They are based on the **whole dataset**, so they do not change "
        "when the sidebar filters move — a decision should not flip "
        "because someone narrowed the view."
    )

    try:
        render_priority_matrix()
    except Exception as exc:
        _LOGGER.warning("Priority matrix failed: %s", exc)

    st.divider()
    st.markdown("### Prioritised recommendations")
    st.caption(
        "Each card carries evidence, action, lead and measures — ready to "
        "hand to the team that owns it."
    )

    info_card(
        "🔴 R1 — Investigate RJ delivery failures (highest ROI)",
        """<p><b>Evidence.</b> RJ carries 12,687 orders at a 13.5% late
        rate — <b>2.3×</b> the 5.9% national average. RJ's average score
        (3.87) sits 0.31 stars (≈7%) below PR (4.18).</p>
        <p><b>Action.</b> Audit RJ carriers, warehouse locations,
        last-mile routes and hand-off delays. Run a controlled pilot
        before scaling.</p>
        <p><b>Lead.</b> Logistics / Operations + Regional Management.</p>
        <p><b>Measures.</b> Late-delivery rate; on-time rate; median
        and 90th-percentile transit time; average review score.</p>""",
        variant="warn",
        tags=["Impact: High", "Effort: Medium", "RJ"],
        priority="high",
    )
    info_card(
        "🔴 R2 — Trigger intervention before day 20",
        """<p><b>Evidence.</b> Delivery buckets decline 4.45 → 4.36 →
        4.22 → 3.69 → 2.26. The largest single step (−0.53 stars, ≈12%)
        is between 11–20 days and 21–30 days — a <b>49%</b> drop from the
        fastest bucket to the slowest.</p>
        <p><b>Action.</b> Ageing-order alerts; escalate at-risk
        parcels; revised ETAs.</p>
        <p><b>Lead.</b> Logistics / Operations + Customer Experience.</p>
        <p><b>Measures.</b> Orders over 20 days; alert-to-resolution
        time; complaint rate.</p>""",
        variant="warn",
        tags=["Impact: High", "Effort: Low", "SLA"],
        priority="high",
    )
    info_card(
        "🟡 R3 — Recalibrate estimated delivery dates",
        """<p><b>Evidence.</b> Late orders run 9.55 days past estimate
        on average — systematic optimism, not random slippage
        (≈76% of the mean on-time delivery time of 12.56 days).</p>
        <p><b>Action.</b> Fit ETA from realised transit history by
        carrier and destination state; back-test predicted vs actual.</p>
        <p><b>Lead.</b> Data Science / Analytics + Logistics.</p>
        <p><b>Measures.</b> ETA error (days); % delivered by promised
        date.</p>""",
        variant="info",
        tags=["Impact: Medium", "Effort: Medium", "Forecasting"],
        priority="medium",
    )
    info_card(
        "🟡 R4 — Pilot improvement in Bahia (BA)",
        """<p><b>Evidence.</b> BA scores 3.86 — the lowest of any state
        with 2,000+ orders — across ~3,340 orders, <b>3.4%</b> of
        national volume.</p>
        <p><b>Action.</b> Apply the RJ diagnostic checklist to BA; test
        changes at limited scale.</p>
        <p><b>Lead.</b> Regional Operations + Logistics + CX.</p>
        <p><b>Measures.</b> BA avg score; late rate; complaints.</p>""",
        variant="info",
        tags=["Impact: Medium", "Effort: Medium", "BA"],
        priority="medium",
    )
    info_card(
        "🟢 R5 — Promise reliability, not unsupported speed",
        """<p><b>Evidence.</b> On-time = 4.29; late = 2.57 — a
        <b>40.1%</b> drop. The damage is in the <i>miss</i>, not the
        speed.</p>
        <p><b>Action.</b> Align delivery messaging with realistic
        estimates the network can meet.</p>
        <p><b>Lead.</b> Marketing / Customer Communications + Logistics.</p>
        <p><b>Measures.</b> On-time delivery; ETA accuracy; contacts
        about delivery; review score.</p>""",
        variant="ok",
        tags=["Impact: High", "Effort: Low", "Brand"],
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
                "Built with Streamlit · Matplotlib · pandas"
            ),
        },
    )
    _apply_global_style()

    st.session_state.setdefault("filter_year", "All")
    st.session_state.setdefault("filter_states", [])
    st.session_state.setdefault("active_tab", NAV_TABS[0])

    try:
        load_data()
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.exception("Data load failed")
        st.error(
            f"❌ **Data load failed.** {exc}\n\n"
            "On Streamlit Cloud, ensure `OLIST_DATA_DIR` points to "
            "the folder with the CSVs (or place them in `./data`)."
        )
        st.stop()
    except Exception as exc:
        _LOGGER.exception("Unexpected load error")
        st.error(f"❌ **Unexpected load error.** {exc}")
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
    avg_live = (
        f"{k_live['avg_review_score']:.2f}★"
        if k_live.get("avg_review_score") is not None
        else f"{REF['avg_review_score']:.2f}★"
    )
    late_pill = (
        f"🚚 {k_live['late_pct']:.1f}% late"
        if k_live.get("late_pct") is not None else "🚚 late = ?"
    )
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
        f"**Reference (filters = All):** Total Orders "
        f"{REF['total_orders']:,} · Delivered {REF['delivered_orders']:,} "
        f"({100.0*REF['delivered_orders']/REF['total_orders']:.1f}%) · "
        f"Avg Review {REF['avg_review_score']:.2f} · Late "
        f"{REF['late_deliveries']:,} "
        f"({100.0*REF['late_deliveries']/REF['delivered_orders']:.1f}% of "
        f"delivered) · Avg Delivery Days {REF['avg_delivery_days']:.2f} · "
        f"Headline gap {REF['star_gap']:.2f}★ "
        f"({100.0*REF['star_gap']/REF['on_time_score']:.1f}%) — "
        f"Q6: on-time {REF['on_time_score']:.2f} "
        f"(n={REF['q6_on_time_orders']:,}) vs late "
        f"{REF['late_score']:.2f} (n={REF['q6_late_orders']:,}). "
        "Every KPI, chart and insight line responds to the Year and State "
        "filters and reconciles to the verified SQL reference when both "
        "are set to All."
    )


if __name__ == "__main__":
    main()