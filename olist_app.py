# olist_app.py
"""
════════════════════════════════════════════════════════════════════════════
OLIST CUSTOMER SATISFACTION DASHBOARD
PORA Academy · Cohort 7 · Group 1 Capstone

A decision-first dashboard built on 99,441 Olist orders (2016–2018).
Primary finding: delivery punctuality is the #1 satisfaction lever —
late orders cost ~1.73★ (40% of the on-time score).

TAB LAYOUT (in-page horizontal navigation, not sidebar)
  1. Overview               KPIs + Review Score Distribution (NOT filterable)
  2. Customer Satisfaction  Delivery-time buckets, state scores, on-time by
                            state, monthly trend, Brazil reviews map
  3. Detail & Method        Delivery Punctuality vs Satisfaction (+ what-if),
                            limitations, whole-dataset diagnostics
  4. Recommendations        Priority matrix, findings, prioritised actions

DEPLOY
  Local:    streamlit run olist_app.py
  Cloud:    Streamlit Community Cloud → set OLIST_DATA_DIR if CSVs
            are not in ./data
REQUIREMENTS
  streamlit · pandas · matplotlib · requests · plotly · scipy
  See requirements.txt for tested deployment dependency ranges.
════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Sequence

# ────────────────────────────────────────────────────────────────────────────
# DEPLOYMENT SAFETY: Agg backend MUST be selected before pyplot import.
# On headless Streamlit Community Cloud, the default interactive backend
# raises TclError. This is a hard requirement.
# ────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")

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
    # ── Brand (WCAG AA on white) ─────────────────────────────────────────
    primary:       str = "#4F46E5"   # Indigo 600 — AA on white
    primary_light: str = "#A5B4FC"
    primary_dark:  str = "#3730A3"
    accent:        str = "#EA580C"   # Orange 700 — AA on white
    accent_light:  str = "#FDBA74"
    accent_dark:   str = "#9A3412"

    # ── Semantic (WCAG AA on white) ──────────────────────────────────────
    success:       str = "#059669"   # Emerald 600
    success_light: str = "#6EE7B7"
    success_dark:  str = "#047857"
    warn:          str = "#D97706"   # Amber 600
    warn_light:    str = "#FCD34D"
    warn_dark:     str = "#92400E"
    danger:        str = "#E11D48"   # Rose 600
    danger_light:  str = "#FDA4AF"
    danger_dark:   str = "#9F1239"

    # ── Neutrals ─────────────────────────────────────────────────────────
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
    # Signature-based keys invalidate on file change; no TTL needed.
    data_cache_ttl_seconds:    Final[int | None] = None
    geojson_cache_ttl_seconds: Final[int | None] = None
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
    "PSIjNEY0NkU1IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3Vu"
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
    """Return the configured directory containing the four Olist CSV files."""
    return os.environ.get(
        "OLIST_DATA_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    )


def _data_signature() -> str:
    """Fingerprint file metadata so edits invalidate Streamlit's data cache."""
    parts: list[str] = []
    directory = _data_dir()
    for table, filename in sorted(CONFIG.tables.items()):
        path = os.path.join(directory, filename)
        try:
            stat = os.stat(path)
            parts.append(f"{table}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(f"{table}:missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds,
               show_spinner="Loading Olist data…")
def load_data(data_signature: str = "") -> dict[str, pd.DataFrame]:
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

    for table, key in (("orders", "order_id"), ("customers", "customer_id")):
        duplicate_count = int(frames[table][key].duplicated().sum())
        if duplicate_count:
            raise ValueError(
                f"Table {table!r} contains {duplicate_count:,} duplicate "
                f"values in its key column {key!r}. Please clean the source CSV."
            )

    score_values = pd.to_numeric(
        frames["reviews"]["review_score"], errors="coerce"
    )
    invalid_scores = int((~score_values.dropna().between(1, 5)).sum())
    if invalid_scores:
        _LOGGER.warning(
            "Reviews CSV contains %s review scores outside the 1–5 range; "
            "these values should be investigated.", invalid_scores,
        )

    _LOGGER.info(
        "Data loaded: %d tables, %d rows",
        len(frames), sum(len(df) for df in frames.values()),
    )
    return frames


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds,
               show_spinner="Preparing analysis model…")
def build_model(frames_fingerprint: str) -> pd.DataFrame:
    frames = load_data(_data_signature())
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
    signature = _data_signature()
    return build_model(signature)


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
    frames = load_data(_data_signature())
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


def _insight(year: str, states: Sequence[str], text: str,
             scope: str | None = None) -> None:
    scope_txt = scope if scope is not None else scope_label(year, states)
    st.markdown(
        f"""
        <div class="insight">
          <span class="insight-chip">💡 Insight</span>
          <div class="insight-body">
            <span class="insight-scope">{scope_txt}</span>
            <span class="insight-text">{text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _impl(text: str) -> str:
    return ('<span class="insight-impl"><b>Business implication:</b> '
            f'{text}</span>')


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


_DECISION_LABELS: Final[dict[str, str]] = {
    "action":  "ACTION",
    "monitor": "MONITOR",
    "fyi":     "FYI",
}


def _chart_card(fn: Callable[..., None], *args,
                decision: str = "monitor", **kwargs) -> None:
    label = _DECISION_LABELS.get(decision, "MONITOR")
    with st.container(border=True):
        st.markdown(
            f'<span class="chip chip-{decision}">{label}</span>',
            unsafe_allow_html=True,
        )
        _safe_chart(fn, *args, **kwargs)


# ════════════════════════════════════════════════════════════════════════════
# §6  OVERVIEW CHART  (Q2) — NOT FILTERABLE
# ════════════════════════════════════════════════════════════════════════════

def chart_review_distribution() -> None:
    year: str = "All"
    states: tuple[str, ...] = ()

    st.markdown("### Review Score Distribution")
    st.caption(
        "🔒 This chart is **not affected by the Year / State filters** — "
        "it always shows the whole dataset as the baseline."
    )
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
    ax.set_title("Review Score Distribution (whole dataset)", pad=14,
                 loc="left")
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

    s1 = float(pct.get(1, 0.0)); s2 = float(pct.get(2, 0.0))
    s3 = float(pct.get(3, 0.0)); s5 = float(pct.get(5, 0.0))
    low_share  = float(pct.loc[pct.index.isin([1, 2])].sum())
    high_share = float(pct.loc[pct.index.isin([4, 5])].sum())
    net = high_share - low_share

    if s1 > s2 and s1 > s3:
        shape = (
            f"The distribution is <b>polarised (U-shaped)</b>: 1★ "
            f"({s1:.1f}%) is larger than both 2★ ({s2:.1f}%) and 3★ "
            f"({s3:.1f}%), so unhappy customers tend to be <i>very</i> "
            f"unhappy rather than mildly disappointed — the pattern of a "
            f"failed experience (such as a late or missing parcel) rather "
            f"than a marginal product gripe."
        )
    else:
        shape = (
            "Dissatisfaction is spread across the lower stars rather than "
            "concentrated at 1★, which points to milder, more diffuse "
            "disappointment."
        )

    _insight(
        year, states,
        f"<b>{s5:.1f}%</b> of {total:,} reviews are 5★ and <b>{s1:.1f}%</b> "
        f"are 1★; promoters (4–5★) hold <b>{high_share:.1f}%</b> vs "
        f"detractors (1–2★) at <b>{low_share:.1f}%</b>, a net margin of "
        f"<b>{net:+.1f} pp</b>. Overall sentiment is "
        f"strongly positive, so the business is not broadly failing. "
        f"{shape} The detractor tail is where the improvement opportunity "
        f"sits, and the next charts test whether delivery performance "
        f"explains it." + _impl(
            f"keep protecting the experience behind
