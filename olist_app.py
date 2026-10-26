# olist_app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis · Group 1 Capstone

Professional, decision-first dashboard.
  · Modern indigo/coral palette · Sora + Inter + JetBrains Mono
  · Narrative chapter layout — evidence → when → where → trend → geography
  · Cached pandas layer · lazy tab rendering
  · Advanced punctuality module (severity tiers, CI, what-if)
  · Streamlit Cloud ready · version-safe API shims

Deploy:   streamlit run olist_app.py
Cloud:    Streamlit Community Cloud — set OLIST_DATA_DIR if needed.
Requires: streamlit, pandas, matplotlib, requests
Optional: scipy (statistical CI), plotly (interactive matrix)
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Final, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from matplotlib.ticker import FuncFormatter

# ════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CHART_FIGSIZE: Final[tuple[float, float]] = (8.6, 4.8)
CHART_FIGSIZE_WIDE: Final[tuple[float, float]] = (13.5, 4.4)
PARTIAL_YEARS: Final[frozenset[str]] = frozenset({"2016", "2018"})

PUNCT_MIN_N_TIER:   Final[int] = 30
PUNCT_MIN_N_STATUS: Final[int] = 30
PUNCT_SCENARIOS:    Final[tuple[int, ...]] = (5, 10, 25, 50)

REF = {
    "total_orders":      99_441,
    "delivered_orders":  96_478,
    "avg_review_score":   4.09,
    "late_deliveries":    7_826,
    "avg_delivery_days": 12.56,
    "on_time_score":      4.29,
    "late_score":         2.57,
    "star_gap":           1.73,
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
# §2.x  STREAMLIT API SHIMS — handle use_container_width deprecation
# ════════════════════════════════════════════════════════════════════════════

def _st_pyplot(fig: plt.Figure) -> None:
    """st.pyplot with fallback across Streamlit versions."""
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)


def _st_dataframe(df: pd.DataFrame, *, hide_index: bool = True) -> None:
    """st.dataframe with fallback across Streamlit versions."""
    try:
        st.dataframe(df, hide_index=hide_index, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=hide_index, use_container_width=True)


def _st_plotly(fig) -> None:
    """st.plotly_chart with fallback across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


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
        frames[table] = pd.read_csv(path)

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

    reviews = (
        reviews.groupby("order_id", as_index=False)["review_score"].mean()
    )

    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(reviews, on="order_id", how="left")

    for col in (
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["year"] = (
        df["order_purchase_timestamp"].dt.year.astype("Int64").astype(str)
    )
    df["month"] = (
        df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    )

    delivered_mask = (
        df["order_status"].eq("delivered")
        & df["order_delivered_customer_date"].notna()
    )
    df["is_delivered"] = delivered_mask

    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["is_late"] = (
        delivered_mask
        & (df["order_delivered_customer_date"]
           > df["order_estimated_delivery_date"])
    )

    return df


def get_model() -> pd.DataFrame:
    frames = load_data()
    fingerprint = hashlib.md5(
        "|".join(f"{k}:{len(v)}" for k, v in sorted(frames.items()))
        .encode()
    ).hexdigest()
    return build_model(fingerprint)


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
def query_kpis(year: str, states_tuple: tuple[str, ...]) -> dict[str, Any]:
    df = apply_filters(get_model(), year, list(states_tuple))
    delivered = df[df["is_delivered"]]

    avg_review = (
        df["review_score"].mean()
        if df["review_score"].notna().any() else None
    )
    avg_days = (
        delivered["delivery_days"].mean()
        if delivered["delivery_days"].notna().any() else None
    )

    return {
        "total_orders":      int(df["order_id"].nunique()),
        "delivered_orders":  int(delivered["order_id"].nunique()),
        "avg_review_score":  float(avg_review) if avg_review is not None else None,
        "late_deliveries":   int(delivered["is_late"].sum()),
        "avg_delivery_days": float(avg_days) if avg_days is not None else None,
    }


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds, show_spinner=False)
def query_headline_gap(year: str,
                       states_tuple: tuple[str, ...]) -> dict[str, Any]:
    df = apply_filters(get_model(), year, list(states_tuple))
    d = df[df["is_delivered"] & df["review_score"].notna()].copy()
    d["status"] = d["is_late"].map({True: "Late", False: "On Time"})
    grp = d.groupby("status")["review_score"].mean()
    on_time = float(grp.get("On Time")) if "On Time" in grp else None
    late    = float(grp.get("Late"))    if "Late"    in grp else None
    gap     = (on_time - late) if (
        on_time is not None and late is not None
    ) else None
    return {"on_time": on_time, "late": late, "gap": gap}


def fmt_int(v: Any) -> str:
    return "—" if v is None or pd.isna(v) else f"{int(v):,}"


def fmt_float(v: Any, p: int = 2) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.{p}f}"


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
    """Render a chart inside a scoped card wrapper."""
    with st.container(border=True):
        st.markdown('<div class="chart-card-scope"></div>',
                    unsafe_allow_html=True)
        _safe_chart(fn, *args, **kwargs)


def _chapter(num: int, eyebrow: str, title: str, subtitle: str) -> None:
    """Render a chapter header — anchors the story, guides the reader."""
    st.markdown(
        f"""
        <div class="chapter-header">
          <div class="chapter-badge">CHAPTER {num}</div>
          <div class="chapter-body">
            <div class="chapter-eyebrow">{eyebrow}</div>
            <h3 class="chapter-title">{title}</h3>
            <p class="chapter-subtitle">{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# §6  CHARTS — Overview
# ════════════════════════════════════════════════════════════════════════════

def chart_review_distribution(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Review Score Distribution")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["review_score"].notna()]

    if df.empty:
        _no_data()
        return

    counts = df["review_score"].round().astype(int).value_counts().sort_index()
    total = int(counts.sum())
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
    for b, p in zip(bars, pct.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{p:.1f}%", ha="center", va="bottom", fontsize=11,
                color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    s5 = float(pct.get(5, 0))
    s1 = float(pct.get(1, 0))
    _insight(year, states,
             f"{s5:.1f}% of {total:,} reviews are 5★; {s1:.1f}% are 1★.")


def chart_review_volume_by_year(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Review Volume Share by Year")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["review_score"].notna() & df["year"].notna()]

    if df.empty:
        _no_data()
        return

    counts = df.groupby("year").size().sort_index()
    total = int(counts.sum())
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
        label = f"{p:.1f}%  ({int(n):,})" + (
            "  ⚠️" if y in PARTIAL_YEARS else ""
        )
        ax.text(b.get_width() + total * 0.01,
                b.get_y() + b.get_height() / 2, label,
                va="center", fontsize=11, color=PALETTE.slate_700,
                fontweight="bold")
    _finish(fig)

    biggest = counts.idxmax()
    smallest = counts.idxmin()
    _insight(year, states,
             f"{biggest} contributes the largest share "
             f"({int(counts[biggest]):,}, {pct[biggest]:.1f}%); "
             f"{smallest} the smallest ({int(counts[smallest]):,}, "
             f"{pct[smallest]:.1f}%).")


# ════════════════════════════════════════════════════════════════════════════
# §6.x  ADVANCED — Delivery Punctuality vs Satisfaction
# ════════════════════════════════════════════════════════════════════════════

def _late_severity(days_late: pd.Series) -> pd.Series:
    out = pd.Series("On Time", index=days_late.index, dtype="object")
    out = out.mask(days_late.between(1, 3), "Slightly Late")
    out = out.mask(days_late.between(4, 7), "Moderately Late")
    out = out.mask(days_late >= 8, "Severely Late")
    return out


@st.cache_data(ttl=CONFIG.data_cache_ttl_seconds, show_spinner=False)
def _punctuality_frame(year: str,
                       states_tuple: tuple[str, ...]) -> pd.DataFrame:
    df = apply_filters(get_model(), year, list(states_tuple))
    df = df[df["is_delivered"] & df["review_score"].notna()].copy()
    if df.empty:
        return df

    df["days_late"] = (
        df["order_delivered_customer_date"]
        - df["order_estimated_delivery_date"]
    ).dt.days
    df = df[df["days_late"].notna()]
    if df.empty:
        return df

    df["severity"] = _late_severity(df["days_late"])
    df["is_detractor"] = df["review_score"].le(2)
    df["is_promoter"]  = df["review_score"].ge(4)
    return df


def _welch_ci(a: pd.Series, b: pd.Series, alpha: float = 0.05
              ) -> tuple[float, float, float, float, float]:
    import numpy as np
    na, nb = len(a), len(b)
    diff = float(a.mean() - b.mean()) if na and nb else float("nan")
    if na < 2 or nb < 2:
        return diff, float("nan"), float("nan"), float("nan"), float("nan")

    try:
        from scipy import stats
        t, _p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
        se = float(np.sqrt(va / na + vb / nb))
        denom = ((va / na) ** 2 / (na - 1)
                 + (vb / nb) ** 2 / (nb - 1))
        df_w = (va / na + vb / nb) ** 2 / denom if denom > 0 else float("nan")
        tcrit = (float(stats.t.ppf(1 - alpha / 2, df_w))
                 if df_w == df_w else float("nan"))
        ci_low, ci_high = diff - tcrit * se, diff + tcrit * se
        pooled = float(
            np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        ) if (na + nb - 2) > 0 else float("nan")
        d = diff / pooled if pooled and pooled > 0 else float("nan")
        return diff, ci_low, ci_high, float(t), float(d)
    except Exception:
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
    """ADVANCED: severity-aware punctuality → satisfaction analysis.

    Chart card contains: dual-panel bar (mean + detractor) and the
    verdict. The what-if slider + metrics + sensitivity table render
    *outside* the card so they breathe properly.
    """
    st.markdown("### Delivery Punctuality vs Satisfaction")
    st.caption(
        "Severity-tiered dose-response · statistical confidence · "
        "operational upside."
    )

    df = _punctuality_frame(year, states)
    if df.empty:
        _no_data("No delivered orders with both a delivery date and a review.")
        return

    tier_order = ["On Time", "Slightly Late", "Moderately Late", "Severely Late"]
    grp = (
        df.groupby("severity", observed=False)
          .agg(
              orders=("order_id", "nunique"),
              avg_score=("review_score", "mean"),
              detractor_rate=("is_detractor", "mean"),
              avg_days_late=("days_late", "mean"),
          )
          .reindex(tier_order)
          .dropna(subset=["orders"])
    )
    if grp.empty:
        _no_data()
        return
    grp["reliable"] = grp["orders"] >= PUNCT_MIN_N_TIER

    tier_colors = {
        "On Time":         PALETTE.success,
        "Slightly Late":   PALETTE.warn_light,
        "Moderately Late": PALETTE.warn,
        "Severely Late":   PALETTE.danger,
    }
    colors = [tier_colors.get(str(t), PALETTE.slate_400) for t in grp.index]

    fig, axes = plt.subplots(
        1, 2,
        figsize=CHART_FIGSIZE_WIDE,
        gridspec_kw={"wspace": 0.32},
    )
    ax1, ax2 = axes

    bars1 = ax1.bar(grp.index.astype(str), grp["avg_score"],
                    color=colors, edgecolor=PALETTE.white, linewidth=1.4,
                    width=0.62)
    ax1.set_title("Mean score by lateness severity", pad=10, loc="left",
                  fontsize=12)
    ax1.set_ylabel("Avg review score (1–5)")
    ax1.set_ylim(1, 5.4)
    _style_axes(ax1)

    for b, s, n, reliable in zip(bars1, grp["avg_score"], grp["orders"],
                                 grp["reliable"]):
        label = f"{s:.2f}\nn={int(n):,}"
        if not reliable:
            label += " ⚠️"
        ax1.text(b.get_x() + b.get_width() / 2, s + 0.07,
                 label, ha="center", va="bottom", fontsize=9,
                 color=PALETTE.slate_700, fontweight="bold")

    sel_avg = float(df["review_score"].mean())
    ax1.axhline(sel_avg, color=PALETTE.slate_500, linestyle="--",
                linewidth=1.1, label=f"Selection avg {sel_avg:.2f}★")
    ax1.legend(loc="lower left", frameon=False, fontsize=9)

    ymax = max(5.0, float((grp["detractor_rate"] * 100).max()) * 1.25)
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
        if not reliable:
            label += " ⚠️"
        ax2.text(b.get_x() + b.get_width() / 2, p + ymax * 0.02,
                 label, ha="center", va="bottom", fontsize=9,
                 color=PALETTE.slate_700, fontweight="bold")

    _finish(fig)

    # Statistical verdict
    on_time_scores = df.loc[df["severity"] == "On Time", "review_score"]
    late_scores    = df.loc[df["severity"] != "On Time", "review_score"]
    n_on, n_late = len(on_time_scores), len(late_scores)

    if n_on >= PUNCT_MIN_N_STATUS and n_late >= PUNCT_MIN_N_STATUS:
        diff, ci_lo, ci_hi, t_stat, d = _welch_ci(on_time_scores, late_scores)
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

    # What-if scenario — rendered OUTSIDE the chart card for proper spacing
    st.markdown("#### 🎛️ What-if: convert late orders to on-time")
    if n_late >= PUNCT_MIN_N_STATUS and n_on >= PUNCT_MIN_N_STATUS:
        pct = st.slider(
            "Percentage of late orders made on-time",
            min_value=0, max_value=100, value=10, step=5,
            key=f"whatif_{year}_{len(states)}",
            help="Projects the effect on the selection average. "
                 "Assumes converted orders land at the on-time mean.",
        )
        on_mean = float(on_time_scores.mean())
        late_mean = float(late_scores.mean())
        new_total = n_on + n_late
        converted = n_late * pct / 100.0
        new_mean = (n_on * on_mean
                    + (n_late - converted) * late_mean
                    + converted * on_mean) / new_total
        c1, c2, c3 = st.columns(3)
        c1.metric("Late orders fixed", f"{converted:,.0f}",
                  delta=f"{pct}% of {n_late:,}")
        c2.metric("Expected avg score", f"{new_mean:.3f}★")
        c3.metric("Δ vs current", f"{new_mean - sel_avg:+.3f}★",
                  delta_color="normal")
    else:
        st.caption("Widen the filters to enable the what-if calculator.")

    # Full sensitivity table
    if n_late >= PUNCT_MIN_N_STATUS and n_on >= PUNCT_MIN_N_STATUS:
        on_mean = float(on_time_scores.mean())
        late_mean = float(late_scores.mean())
        new_total = n_on + n_late
        scen_rows = []
        for p in PUNCT_SCENARIOS:
            conv = n_late * p / 100.0
            nm = (n_on * on_mean
                  + (n_late - conv) * late_mean
                  + conv * on_mean) / new_total
            scen_rows.append({
                "Scenario":           f"{p}% of late → on-time",
                "Late orders fixed":  f"{conv:,.0f}",
                "Expected avg score": f"{nm:.3f}",
                "Δ vs current":       f"{nm - sel_avg:+.3f}★",
            })
        with st.expander("📋 Full sensitivity table (5%, 10%, 25%, 50%)",
                         expanded=False):
            _st_dataframe(pd.DataFrame(scen_rows))
            lift_1pct = (n_late * 0.01 * (on_mean - late_mean)) / new_total
            st.caption(
                f"Stars gained per 1% of late orders converted = "
                f"**{lift_1pct:+.4f}★**."
            )

    # Insight line
    if len(grp) > 1:
        worst = grp["avg_score"].idxmin()
        best  = grp["avg_score"].idxmax()
        gap_ot_late = (float(on_time_scores.mean() - late_scores.mean())
                       if n_on and n_late else float("nan"))
        noisy = int((~grp["reliable"]).sum())
        noisy_txt = (f" {noisy} tier(s) flagged ⚠️ (n < {PUNCT_MIN_N_TIER})."
                     if noisy else "")
        top_det = grp["detractor_rate"].idxmax()
        _insight(
            year, states,
            f"Score drops from {best} ({grp.loc[best,'avg_score']:.2f}) to "
            f"{worst} ({grp.loc[worst,'avg_score']:.2f}); on-time vs late "
            f"gap = {gap_ot_late:.2f}★. Detractor share rises "
            f"{grp.loc['On Time','detractor_rate']*100:.1f}% → "
            f"{grp.loc[top_det,'detractor_rate']*100:.1f}% as lateness "
            f"worsens.{noisy_txt}"
        )
    else:
        only = grp.index[0]
        _insight(
            year, states,
            f"Only the '{only}' tier is populated "
            f"({grp.iloc[0]['avg_score']:.2f}★, "
            f"n={int(grp.iloc[0]['orders']):,}) — no on-time vs late "
            "comparison is possible."
        )

    _partial_note(year)


def chart_satisfaction_by_delivery_bucket(year: str,
                                          states: tuple[str, ...]) -> None:
    st.markdown("### Satisfaction by Delivery-Time Bucket")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["is_delivered"] & df["review_score"].notna()
            & df["delivery_days"].notna()]
    if df.empty:
        _no_data()
        return

    bins = [0, 5, 10, 20, 30, float("inf")]
    labels = ["0-5 days", "6-10 days", "11-20 days", "21-30 days", "30+ days"]
    df = df.copy()
    df["bucket"] = pd.cut(df["delivery_days"], bins=bins,
                          labels=labels, include_lowest=True)

    grp = df.groupby("bucket", observed=False).agg(
        orders=("order_id", "nunique"),
        avg_score=("review_score", "mean"),
    ).reset_index()
    grp = grp[grp["orders"] > 0]
    if grp.empty:
        _no_data()
        return

    total_n = grp["orders"].sum()
    sel_avg = float((grp["avg_score"] * grp["orders"]).sum() / total_n)

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
    ax.set_title("Satisfaction by Delivery-Time Bucket",
                 pad=14, loc="left")
    ax.set_xlabel("Days from purchase to delivery")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5.4)
    ax.axhline(sel_avg, color=PALETTE.slate_500, linestyle="--",
               linewidth=1.2, label=f"Selection avg {sel_avg:.2f}")
    _style_axes(ax)
    ax.legend(loc="lower left", frameon=False)
    for b, s, n in zip(bars, grp["avg_score"], grp["orders"]):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.08,
                f"{s:.2f}\n(n={int(n):,})", ha="center", va="bottom",
                fontsize=10, color=PALETTE.slate_700, fontweight="bold")
    _finish(fig)

    if len(grp) == 1:
        text = (f"All delivered orders in the "
                f"{grp.iloc[0]['bucket']} bucket "
                f"({grp.iloc[0]['avg_score']:.2f}★).")
    else:
        first, last = grp.iloc[0], grp.iloc[-1]
        below = grp.loc[grp["avg_score"] < sel_avg, "bucket"]
        below_txt = (
            "Buckets below avg: " + ", ".join(map(str, below)) + "."
            if not below.empty else
            "No bucket below the selection average."
        )
        text = (f"Scores run {first['avg_score']:.2f} "
                f"({first['bucket']}) → {last['avg_score']:.2f} "
                f"({last['bucket']}). {below_txt}")
    _insight(year, states, text)


def chart_state_scores(year: str, states: tuple[str, ...]) -> None:
    df = apply_filters(get_model(), year, list(states))
    df = df[df["review_score"].notna() & df["customer_state"].notna()]

    if df.empty:
        st.markdown("### Average Review Score by State *(top 10)*")
        _no_data()
        return

    grp = df.groupby("customer_state").agg(
        orders=("order_id", "nunique"),
        avg_score=("review_score", "mean"),
    ).reset_index().sort_values("orders", ascending=False).head(10)

    n_states = len(grp)
    title_cap = ("top 10" if n_states >= 10
                 else f"{n_states} state{'s' if n_states != 1 else ''}")
    scope_txt = "in current selection" if states else "by orders"
    st.markdown(f"### Average Review Score by State *({title_cap} {scope_txt})*")

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
    ax.set_title(
        f"Average Review Score by State ({title_cap} {scope_txt})",
        pad=14, loc="left",
    )
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
        text = (f"{low['customer_state']} ({low['avg_score']:.2f}) "
                f"anchors the low end; {high['customer_state']} "
                f"({high['avg_score']:.2f}) the high end — spread "
                f"{high['avg_score'] - low['avg_score']:.2f}★.")
    _insight(year, states, text)


def chart_ontime_by_state(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### On-Time vs Late by State *(top 5)*")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["is_delivered"] & df["customer_state"].notna()]

    if df.empty:
        _no_data()
        return

    grp = df.groupby("customer_state").agg(
        delivered=("order_id", "nunique"),
        late=("is_late", "sum"),
    ).reset_index().sort_values("delivered", ascending=False).head(5)
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
    best = grp.loc[rate.idxmin()]
    overall = 100.0 * grp["late"].sum() / grp["delivered"].sum()
    _insight(year, states,
             f"Across top-5 states, {overall:.1f}% delivered late. "
             f"{worst['customer_state']} highest "
             f"({100 * worst['late'] / worst['delivered']:.1f}%); "
             f"{best['customer_state']} lowest "
             f"({100 * best['late'] / best['delivered']:.1f}%).")


def chart_monthly_score(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Monthly Average Review Score")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["review_score"].notna() & df["month"].notna()]

    if df.empty:
        _no_data()
        return

    grp = df.groupby("month").agg(
        reviews=("order_id", "nunique"),
        avg_score=("review_score", "mean"),
    ).reset_index().sort_values("month")

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
        xytext=(0, -46), textcoords="offset points",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE.slate_500, lw=1.2),
        fontsize=10, color=PALETTE.danger, fontweight="bold",
    )
    _finish(fig)

    noisy = int((grp["reviews"] < CONFIG.min_reviews_per_month).sum())
    note = (
        f" {noisy} month(s) have < {CONFIG.min_reviews_per_month} "
        "reviews — noisy averages."
    ) if noisy else ""
    _insight(year, states,
             f"Lowest {grp.loc[low_i,'month']} "
             f"({grp.loc[low_i,'avg_score']:.2f}); highest "
             f"{grp.loc[high_i,'month']} "
             f"({grp.loc[high_i,'avg_score']:.2f}).{note}")
    _partial_note(year)


# ── Brazil map ────────────────────────────────────────────────────────────

@st.cache_data(ttl=CONFIG.geojson_cache_ttl_seconds, show_spinner=False)
def load_brazil_geojson() -> dict[str, Any] | None:
    for url in CONFIG.geojson_urls:
        try:
            r = requests.get(url, timeout=CONFIG.geojson_timeout_seconds)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            _LOGGER.warning("GeoJSON mirror failed: %s — %s", url, exc)
    return None


def _geojson_to_gdf(geojson: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        sigla = (
            props.get("sigla") or props.get("SIGLA")
            or props.get("abbrev") or props.get("code")
            or props.get("name")
        )
        if not sigla:
            continue
        rows.append({"state": str(sigla).upper(),
                     "geometry": feat.get("geometry", {})})
    return pd.DataFrame(rows)


def _plot_geo(ax, geometry, facecolor,
              edgecolor=PALETTE.white, linewidth=0.6):
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
            ax.fill(xs, ys, facecolor=facecolor, edgecolor=edgecolor,
                    linewidth=linewidth,
                    zorder=2 if ring_idx == 0 else 3)


def chart_brazil_map(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Brazil Map · Delivered Orders")

    df = get_model()
    df = df[df["is_delivered"] & df["customer_state"].notna()]
    if year != "All":
        df = df[df["year"] == year]
    counts = df.groupby("customer_state")["order_id"].nunique()

    if counts.empty:
        _no_data("No state-level data for this year.")
        return

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
        if n is not None and in_scope:
            label = f"{st_code}\n{int(n):,}"
            color, weight = PALETTE.ink, "bold"
        elif n is not None:
            label = f"{st_code}\n{int(n):,}"
            color, weight = PALETTE.slate_400, "normal"
        else:
            label = st_code
            color, weight = PALETTE.slate_400, "normal"
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=6, color=color, fontweight=weight, zorder=5)

    ax.set_xlim(-74.0, -34.0)
    ax.set_ylim(-34.0, 6.0)
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
    mode = ("Selected states coloured; rest greyed."
            if filter_active else
            "All 27 states coloured by delivered count.")
    _insight(year, states,
             f"{lo_s} ({int(lo_v):,}) fewest; {hi_s} ({int(hi_v):,}) most "
             f"in scope. {mode}")


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


def render_headline_hero(year: str, states: tuple[str, ...]) -> None:
    g = query_headline_gap(year, states)
    scope = scope_label(year, states)

    if g["gap"] is None:
        body = (
            "Delivery punctuality is the strongest satisfaction driver "
            "in this dataset. Only one delivery status appears in the "
            "current selection — widen the filters to compare on-time "
            "and late orders."
        )
    else:
        body = (
            f'Late deliveries cost <b>{g["gap"]:.2f} stars</b> — '
            f'the single strongest satisfaction driver. '
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
    st.markdown(
        f"""
        <div class="headline-strip">
          <span class="headline-strip-icon">⚡</span>
          <div class="headline-strip-text">
            <span class="headline-strip-label">Headline finding · {scope_label(year, states)}</span>
            <p style="margin:0;">
              Late deliveries cost <b>{g['gap']:.2f}★</b> — the strongest
              satisfaction driver. See the Customer Satisfaction tab for the
              evidence.
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
    d1.metric("On-Time Avg Score", f"{g['on_time']:.2f}★")
    d2.metric("Late Avg Score", f"{g['late']:.2f}★")
    d3.metric("Star Gap (Driver)", f"{g['gap']:.2f}★",
              delta="Priority #1" if g["gap"] > 1.0 else None,
              delta_color="off")
    st.caption(
        "A gap above 1.00★ makes delivery punctuality the highest-ROI "
        "lever for satisfaction improvement."
    )


def render_executive_summary(year: str, states: tuple[str, ...]) -> None:
    """Top-of-page executive summary — one glance, one decision."""
    k = query_kpis(year, states)
    g = query_headline_gap(year, states)

    if g["gap"] is None:
        verdict_txt, verdict_color = "Inconclusive", PALETTE.slate_500
    elif g["gap"] >= 1.5:
        verdict_txt, verdict_color = "Critical priority", PALETTE.danger
    elif g["gap"] >= 1.0:
        verdict_txt, verdict_color = "High priority", PALETTE.warn
    elif g["gap"] >= 0.5:
        verdict_txt, verdict_color = "Moderate", PALETTE.accent
    else:
        verdict_txt, verdict_color = "Healthy", PALETTE.success

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
    """Interactive priority matrix — impact × effort, whole dataset."""
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
            size=[14] * len(items),
            color="impact",
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
        _st_plotly(fig)
    except ImportError:
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
    --brand:        #4F46E5;
    --brand-light:  #A5B4FC;
    --brand-dark:   #3730A3;
    --accent:       #F97316;
    --accent-light: #FDBA74;
    --success:      #10B981;
    --warn:         #F59E0B;
    --danger:       #E11D48;
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
    --fs-hero:    clamp(2.2rem, 1.8rem + 1.6vw, 3rem);
    --fs-h1:      clamp(1.4rem, 1.1rem + 1vw,   1.9rem);
    --fs-h2:      clamp(1.2rem, 1.05rem + 0.6vw, 1.45rem);
    --fs-h3:      clamp(1.05rem, 1rem + 0.3vw,   1.15rem);
    --fs-body:    0.95rem;
    --fs-small:   0.83rem;
    --fs-micro:   0.72rem;
  }

  html, body, [class*="css"], .stApp {
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 15px; line-height: 1.6; color: var(--ink);
    -webkit-font-smoothing: antialiased;
  }
  .stApp { background: var(--slate-50); }
  h1, h2, h3, .app-hero-title, .hero-headline, .exec-summary-title,
  .hero-label, .exec-summary-eyebrow, .chapter-title {
    font-family: 'Sora', 'Inter', sans-serif !important;
    letter-spacing: -0.015em;
  }
  code, .stCodeBlock { font-family: 'JetBrains Mono', monospace !important; }

  /* ── App hero (BIG title) ────────────────────────────────────────── */
  .app-hero {
    background: linear-gradient(135deg, #EEF2FF 0%, #FFFFFF 60%, #FFF7ED 100%);
    border: 1px solid var(--slate-200);
    border-radius: var(--s5);
    padding: var(--s6) var(--s7);
    margin-bottom: var(--s5);
    box-shadow: 0 4px 24px rgba(79,70,229,0.08);
  }
  .app-hero-eyebrow {
    font-size: 0.78rem; font-weight: 800;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--brand); margin-bottom: var(--s3);
    font-family: 'Sora', sans-serif !important;
  }
  .app-hero-title {
    font-size: var(--fs-hero) !important;
    font-weight: 800 !important;
    color: var(--ink) !important;
    line-height: 1.08 !important;
    margin: 0 0 var(--s3) 0 !important;
    padding: 0 !important;
    letter-spacing: -0.035em !important;
    display: block;
  }
  .app-hero-title b {
    color: var(--brand-dark);
    background: linear-gradient(120deg, var(--brand) 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  @supports not ((-webkit-background-clip: text) or (background-clip: text)) {
    .app-hero-title b {
      color: var(--brand) !important;
      background: none !important;
      -webkit-text-fill-color: var(--brand) !important;
    }
  }
  .app-hero-sub {
    font-size: 1.05rem; color: var(--slate-600);
    margin: 0 0 var(--s4) 0; line-height: 1.55;
    max-width: 780px;
  }
  .app-hero-meta {
    display: flex; gap: var(--s3); flex-wrap: wrap;
    align-items: center;
  }

  /* ── Chapter header ──────────────────────────────────────────────── */
  .chapter-header {
    display: flex; gap: var(--s4); align-items: flex-start;
    padding: var(--s4) 0 var(--s3) 0;
    margin-top: var(--s5);
    border-top: 1px solid var(--slate-200);
  }
  .chapter-badge {
    flex-shrink: 0;
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--brand) 0%, var(--brand-dark) 100%);
    color: #FFFFFF; font-family: 'Sora', sans-serif;
    font-size: 0.68rem; font-weight: 800;
    letter-spacing: 0.14em; text-transform: uppercase;
    padding: var(--s2) var(--s3); border-radius: var(--s2);
    box-shadow: 0 2px 8px rgba(79,70,229,0.25);
    min-width: 100px;
  }
  .chapter-body { flex: 1; min-width: 0; }
  .chapter-eyebrow {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--accent-dark); margin-bottom: 2px;
  }
  .chapter-title {
    font-size: var(--fs-h2) !important;
    font-weight: 800 !important;
    color: var(--ink) !important;
    margin: 0 0 var(--s2) 0 !important;
    padding: 0 !important;
    line-height: 1.25 !important;
  }
  .chapter-subtitle {
    font-size: var(--fs-body); color: var(--slate-600);
    margin: 0; line-height: 1.55;
  }

  h1 { color: var(--ink); font-weight: 800; font-size: var(--fs-h1);
       line-height: 1.2; margin-bottom: var(--s2); }
  h2 { color: var(--ink); font-weight: 700; font-size: var(--fs-h2);
       line-height: 1.25; margin-top: var(--s3) !important;
       margin-bottom: var(--s4) !important; }
  h3 { color: var(--ink-soft); font-weight: 700; font-size: var(--fs-h3);
       line-height: 1.3; margin-top: 0 !important;
       margin-bottom: var(--s3) !important; }
  h4 { color: var(--brand); font-size: 1rem; font-weight: 700;
       line-height: 1.35; margin-bottom: var(--s3) !important; }
  p, li, span, label { color: var(--slate-700); font-size: var(--fs-body); }

  .block-container {
    padding-top: var(--s5) !important;
    padding-bottom: var(--s8) !important;
    max-width: 1500px; padding-left: var(--s5) !important;
    padding-right: var(--s5) !important;
  }
  section[data-testid="stSidebar"] { padding-top: var(--s4); }

  /* ── Sidebar ─────────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
    border-right: 1px solid #312E81;
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
    font-size: 0.8rem !important; padding: var(--s2) var(--s3) !important;
  }
  section[data-testid="stSidebar"] .stButton > button {
    background: var(--brand) !important; color: #FFFFFF !important;
    border-radius: var(--s2); font-weight: 600; border: none;
    transition: background 0.15s ease;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--brand-dark) !important;
  }

  .sidebar-warning {
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.45);
    border-left: 3px solid #F59E0B;
    border-radius: var(--s2);
    padding: var(--s3);
    margin-top: var(--s3);
    font-size: var(--fs-small);
    line-height: 1.5;
  }
  .sidebar-warning strong { color: #FCD34D !important; }
  .sidebar-warning span   { color: #E2E8F0 !important; }

  .sidebar-headline {
    background: rgba(79, 70, 229, 0.20);
    border: 1px solid rgba(165, 180, 252, 0.45);
    border-left: 3px solid var(--brand-light);
    border-radius: var(--s2);
    padding: var(--s3) var(--s4);
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
    margin-bottom: var(--s1);
  }
  .sidebar-headline span { color: #CBD5E1 !important; font-size: 0.75rem; }
  .sidebar-headline span b { color: #FFFFFF !important; }

  /* ── Metric cards ────────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid var(--slate-200);
    border-left: 5px solid var(--brand);
    border-radius: var(--s3);
    padding: var(--s5);
    box-shadow: 0 1px 2px rgba(15,23,42,0.04),
                0 4px 12px rgba(15,23,42,0.04);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(15,23,42,0.05),
                0 12px 24px rgba(79,70,229,0.10);
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
    letter-spacing: -0.02em;
    line-height: 1.15;
    font-family: 'Sora', 'Inter', sans-serif !important;
    font-variant-numeric: tabular-nums;
  }
  [data-testid="stMetricDelta"] {
    font-weight: 600 !important;
    font-size: 0.78rem !important;
  }

  /* ── Chart containers — generous padding ─────────────────────────── */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--s3) !important;
    padding: var(--s5) var(--s6) !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03),
                0 4px 14px rgba(15,23,42,0.04);
    margin-bottom: var(--s4);
  }
  [data-testid="stVerticalBlockBorderWrapper"] h3 {
    margin-top: 0 !important;
  }

  /* ── Tabs ────────────────────────────────────────────────────────── */
  button[data-baseweb="tab"] {
    color: var(--slate-500); font-weight: 600; font-size: var(--fs-body);
    padding: var(--s3) var(--s5) !important;
    border-radius: var(--s2) var(--s2) 0 0;
    transition: background 0.15s ease, color 0.15s ease;
    font-family: 'Sora', 'Inter', sans-serif !important;
  }
  button[data-baseweb="tab"]:hover {
    background: var(--slate-100);
    color: var(--brand);
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--brand);
    border-bottom-color: var(--brand);
    border-bottom-width: 3px;
  }
  [data-baseweb="tab-highlight"] { background-color: var(--brand); }

  /* ── Buttons ─────────────────────────────────────────────────────── */
  .stButton > button {
    background: var(--brand); color: #FFFFFF;
    border: none; border-radius: var(--s2);
    padding: var(--s2) var(--s5); font-weight: 600;
    font-size: var(--fs-body);
    transition: background 0.15s ease, transform 0.1s ease,
                box-shadow 0.15s ease;
  }
  .stButton > button:hover {
    background: var(--brand-dark);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79,70,229,0.25);
  }

  /* ── Alerts / expander / slider ──────────────────────────────────── */
  [data-testid="stAlert"] {
    background-color: #FEF3C7;
    border-left: 4px solid var(--warn);
    color: #78350F;
    border-radius: var(--s3);
  }
  [data-testid="stExpander"] {
    border: 1px solid var(--slate-200);
    border-radius: var(--s3);
    background: #FFFFFF;
    margin-bottom: var(--s3);
  }
  [data-testid="stExpander"] summary {
    font-weight: 600; color: var(--brand);
    font-family: 'Sora', sans-serif !important;
  }

  hr { border-color: var(--slate-200); margin: var(--s5) 0; }

  /* ── Info cards ──────────────────────────────────────────────────── */
  .info-card {
    position: relative;
    background: #FFFFFF;
    border: 1px solid var(--slate-200);
    border-left: 5px solid var(--brand);
    border-radius: var(--s3);
    padding: var(--s5);
    margin-bottom: var(--s4);
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  .info-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(15,23,42,0.07);
  }
  .info-card.warn { border-left-color: var(--danger); }
  .info-card.ok   { border-left-color: var(--success); }
  .info-card.info { border-left-color: var(--brand); }
  .info-card.warn h4 { color: var(--danger); }
  .info-card.ok h4   { color: var(--success); }
  .info-card.info h4 { color: var(--brand); }
  .info-card p, .info-card li {
    color: var(--slate-700); line-height: 1.65;
    font-size: var(--fs-body);
  }
  .info-card .tag {
    display: inline-block; background: var(--slate-100);
    color: var(--slate-600); border-radius: 999px;
    padding: var(--s1) var(--s3); font-size: 0.72rem;
    font-weight: 600; margin-right: var(--s2); margin-bottom: var(--s2);
    letter-spacing: 0.02em;
  }
  .priority-bar {
    position: absolute; top: 0; right: 0;
    width: 5px; height: 100%;
    border-radius: 0 var(--s3) var(--s3) 0;
  }
  .priority-high   { background: var(--danger); }
  .priority-medium { background: var(--warn); }
  .priority-low    { background: var(--success); }

  /* ── Headline / hero / strip ─────────────────────────────────────── */
  .hero-banner, .headline-strip {
    display: flex; align-items: center; gap: var(--s3);
    border-radius: var(--s3);
    padding: var(--s3) var(--s4);
    margin: var(--s4) 0;
  }
  .hero-banner {
    background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 55%, #C7D2FE 100%);
    border: 1px solid #A5B4FC;
    border-left: 4px solid var(--brand);
    box-shadow: 0 2px 10px rgba(79,70,229,0.10);
  }
  .headline-strip {
    background: linear-gradient(90deg, #EEF2FF 0%, #FFFFFF 100%);
    border: 1px solid #C7D2FE;
    border-left: 4px solid var(--brand);
  }
  .hero-banner .hero-icon,
  .headline-strip-icon {
    font-size: 20px; line-height: 1; flex-shrink: 0;
  }
  .hero-banner .hero-text { flex: 1; min-width: 0; }
  .headline-strip .headline-strip-text { flex: 1; min-width: 0; }
  .hero-banner .hero-label,
  .headline-strip .headline-strip-label {
    display: block; font-size: 0.68rem; font-weight: 800;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--brand); margin-bottom: var(--s1);
  }
  .hero-banner .hero-headline,
  .headline-strip-text p {
    font-size: var(--fs-body); font-weight: 500;
    color: var(--slate-700); line-height: 1.55;
    margin: 0; letter-spacing: normal;
  }
  .hero-banner .hero-headline b,
  .headline-strip-text b {
    color: var(--brand-dark); font-weight: 700;
  }

  /* ── Executive summary ───────────────────────────────────────────── */
  .exec-summary {
    display: flex; align-items: stretch; gap: var(--s4);
    background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 100%);
    border: 1px solid var(--slate-200);
    border-radius: var(--s3);
    padding: var(--s5);
    margin: var(--s4) 0;
    box-shadow: 0 2px 12px rgba(79,70,229,0.06);
  }
  .exec-summary-main { flex: 1; min-width: 0; }
  .exec-summary-eyebrow {
    font-size: 0.68rem; font-weight: 800;
    letter-spacing: 0.16em; color: var(--brand);
    margin-bottom: var(--s2);
  }
  .exec-summary-title {
    font-size: var(--fs-h2); font-weight: 800;
    color: var(--ink); line-height: 1.25;
    margin-bottom: var(--s2);
  }
  .exec-summary-title b { color: var(--brand-dark); }
  .exec-summary-detail {
    color: var(--slate-600); font-size: var(--fs-body);
    line-height: 1.6;
  }
  .exec-summary-badge {
    display: flex; flex-direction: column; justify-content: center;
    align-items: center; text-align: center;
    min-width: 130px; padding: var(--s3) var(--s4);
    border: 2px solid; border-radius: var(--s3);
  }
  .exec-summary-badge-label {
    font-size: 0.65rem; font-weight: 800;
    letter-spacing: 0.14em; text-transform: uppercase;
    opacity: 0.85; margin-bottom: var(--s2);
    color: inherit;
  }
  .exec-summary-badge-value {
    font-family: 'Sora', sans-serif; font-weight: 800;
    font-size: 1rem; line-height: 1.2; color: inherit;
  }

  /* ── Partial-year banner ─────────────────────────────────────────── */
  .overview-banner {
    background: #FEF3C7;
    border-left: 4px solid var(--warn);
    border-radius: var(--s3);
    padding: var(--s3) var(--s5);
    margin-bottom: var(--s4);
    font-size: var(--fs-body); color: #78350F; line-height: 1.55;
  }
  .overview-banner b { color: #92400E; }

  /* ── Scope pill ──────────────────────────────────────────────────── */
  .scope-pill {
    display: inline-block;
    background: linear-gradient(135deg, var(--brand) 0%, var(--accent) 100%);
    color: #FFFFFF; border-radius: 999px;
    padding: var(--s2) var(--s4);
    font-size: var(--fs-small); font-weight: 600;
    box-shadow: 0 2px 10px rgba(79,70,229,0.30);
  }
  .scope-pill-subtle {
    display: inline-block;
    background: var(--slate-100);
    color: var(--slate-700);
    border: 1px solid var(--slate-200);
    border-radius: 999px;
    padding: var(--s2) var(--s4);
    font-size: var(--fs-small); font-weight: 600;
  }

  /* ── Insight ─────────────────────────────────────────────────────── */
  .insight {
    display: flex; gap: var(--s3); align-items: flex-start;
    background: #F5F3FF;
    border-left: 3px solid var(--brand-light);
    border-radius: var(--s2);
    padding: var(--s3) var(--s4);
    margin-top: var(--s3);
    font-size: var(--fs-small);
  }
  .insight-chip {
    flex-shrink: 0;
    background: var(--brand); color: #FFFFFF;
    padding: 2px 10px; border-radius: 999px;
    font-size: 0.7rem; font-weight: 700;
    white-space: nowrap;
  }
  .insight-body { display: flex; flex-direction: column; gap: 2px; }
  .insight-scope {
    font-size: 0.68rem; font-weight: 700; color: var(--brand);
    text-transform: uppercase; letter-spacing: 0.06em;
    opacity: 0.9;
  }
  .insight-text { color: var(--slate-600); line-height: 1.55; }
  .insight-text b { color: var(--ink); font-weight: 700; }

  /* ── Statistical verdict ─────────────────────────────────────────── */
  .verdict-card {
    display: flex; flex-direction: column; gap: var(--s1);
    background: linear-gradient(135deg, #ECFDF5 0%, #F0FDFA 100%);
    border: 1px solid #6EE7B7;
    border-left: 4px solid var(--success);
    border-radius: var(--s2);
    padding: var(--s3) var(--s4);
    margin: var(--s3) 0;
  }
  .verdict-label {
    font-size: 0.7rem; font-weight: 800;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--success);
  }
  .verdict-text {
    font-size: var(--fs-small); color: var(--slate-700);
    line-height: 1.55;
  }
  .verdict-text b { color: var(--ink); }

  /* ── Empty state ─────────────────────────────────────────────────── */
  .empty-state {
    text-align: center;
    padding: var(--s8) var(--s5);
    background: var(--slate-50);
    border: 1px dashed var(--slate-300);
    border-radius: var(--s3);
  }
  .empty-icon { font-size: 40px; }
  .empty-title {
    font-size: 1.15rem; font-weight: 700;
    color: var(--slate-700); margin: var(--s2) 0;
    font-family: 'Sora', sans-serif;
  }
  .empty-body { color: var(--slate-500); font-size: var(--fs-body); }
  .empty-hint {
    margin-top: var(--s3); color: var(--brand);
    font-size: var(--fs-small); font-weight: 500;
  }

  /* ── Dataframe / caption ─────────────────────────────────────────── */
  [data-testid="stDataFrame"] {
    border-radius: var(--s3); overflow: hidden;
    border: 1px solid var(--slate-200);
  }
  .stPyplot { margin-top: var(--s1); }
  .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--slate-500); font-size: var(--fs-small);
  }

  /* ── Intermediate breakpoint (tablets) ───────────────────────────── */
  @media (max-width: 1100px) {
    .app-hero { padding: var(--s5); }
    .chapter-header { gap: var(--s3); }
    .exec-summary { flex-direction: column; }
    .exec-summary-badge { min-width: auto; }
    [data-testid="stMetric"] { padding: var(--s4); }
  }

  /* ── Mobile ──────────────────────────────────────────────────────── */
  @media (max-width: 768px) {
    .block-container { padding: var(--s4) !important; }
    .app-hero { padding: var(--s5) var(--s4); }
    .app-hero-title { font-size: 1.8rem !important; }
    .hero-banner, .headline-strip {
      flex-direction: column; align-items: flex-start;
    }
    .exec-summary { flex-direction: column; }
    .exec-summary-badge { min-width: auto; }
    .chapter-header { flex-direction: column; gap: var(--s2); }
    .chapter-badge { min-width: auto; }
  }
</style>
"""


def _apply_global_style() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
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


# ════════════════════════════════════════════════════════════════════════════
# §9  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple[str, tuple[str, ...]]:
    st.sidebar.title("🎛️ Filters")
    st.sidebar.caption(
        "Both filters apply to **every KPI and chart**. "
        "Set both to *All* to reproduce the reference values."
    )

    def _reset_filters() -> None:
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
              <span>Year-over-year comparisons are misleading. See Detail &amp; Method.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    states = st.sidebar.multiselect(
        "State",
        options=CONFIG.all_states,
        default=[],
        placeholder="All states",
        help="Pick one or more states. Leaving empty shows all states.",
        key="filter_states",
    )
    states_tuple = tuple(states)

    if states or year != "All":
        st.sidebar.button(
            "↺  Reset filters",
            on_click=_reset_filters,
            use_container_width=True,
        )

    st.sidebar.divider()
    render_headline_sidebar(year, states_tuple)
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
    return year, states_tuple


# ════════════════════════════════════════════════════════════════════════════
# §10  TAB 1 — OVERVIEW  (Chapters 1-3)
# ════════════════════════════════════════════════════════════════════════════

def render_overview(year: str, states: tuple[str, ...],
                    filter_active: bool) -> None:
    st.markdown("## 📈 Overview — the shape of the business")

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

    # ── Chapter 1 — KPI strip ──────────────────────────────────────────
    _chapter(
        1, "Volume & sentiment",
        "How much business, and how satisfied?",
        "Five headline KPIs across the current selection — these anchor the "
        "story before we go into the charts.",
    )
    k = query_kpis(year, states)
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    c1.metric("Total Orders", fmt_int(k["total_orders"]))
    c2.metric("Delivered Orders", fmt_int(k["delivered_orders"]))

    if k["avg_review_score"] is None:
        c3.metric("Avg Review Score", "—")
    else:
        delta = (
            f"{k['avg_review_score'] - REF['avg_review_score']:+.2f} vs baseline"
            if filter_active else None
        )
        c3.metric("Avg Review Score", fmt_float(k["avg_review_score"]),
                  delta=delta)

    late_val, delivered_val = k["late_deliveries"], k["delivered_orders"]
    late_delta = (
        f"{100 * late_val / delivered_val:.1f}% of delivered"
        if delivered_val else None
    )
    c4.metric("Late Deliveries", fmt_int(late_val),
              delta=late_delta, delta_color="inverse")
    c5.metric("Avg Delivery Days", fmt_float(k["avg_delivery_days"]))

    if filter_active:
        st.info(
            f"Active filter — **{scope_label(year, states)}**. "
            "Set both filters to All to reproduce the sidebar reference values."
        )

    render_headline_strip(year, states)

    # ── Chapter 2 — Review distribution (full-width) ───────────────────
    _chapter(
        2, "Rating distribution",
        "Where do the reviews land?",
        "The full 1–5★ spread — shows the volume of delight (5★) versus "
        "damage (1–2★) before we ask *why*.",
    )
    _chart_card(chart_review_distribution, year, states)

    # ── Chapter 3 — Volume by year (full-width) ────────────────────────
    _chapter(
        3, "Volume by year",
        "When did the reviews happen?",
        "Year contribution — flags the partial years (2016, 2018) so nobody "
        "misreads the trend.",
    )
    _chart_card(chart_review_volume_by_year, year, states)


# ════════════════════════════════════════════════════════════════════════════
# §11  TAB 2 — CUSTOMER SATISFACTION  (Chapters 4-8)
# ════════════════════════════════════════════════════════════════════════════

def render_theme(year: str, states: tuple[str, ...]) -> None:
    st.markdown("## ⭐ Customer Satisfaction — the decision story")

    render_headline_hero(year, states)
    render_decision_snapshot(year, states)

    # ── Chapter 4 — Evidence ───────────────────────────────────────────
    _chapter(
        4, "Evidence — driver",
        "Does lateness hurt satisfaction?",
        "Severity-tiered dose-response, plus a statistical verdict and a "
        "what-if calculator. This is the core finding of the analysis.",
    )
    _chart_card(chart_punctuality_vs_satisfaction, year, states)

    # ── Chapter 5 — Timing ─────────────────────────────────────────────
    _chapter(
        5, "Timing — when",
        "Where in the delivery window does satisfaction collapse?",
        "Days from purchase to delivery, bucketed — reveals the exact "
        "threshold where ratings begin to drop.",
    )
    _chart_card(chart_satisfaction_by_delivery_bucket, year, states)

    # ── Chapter 6 — Geography (2-up) ───────────────────────────────────
    _chapter(
        6, "Geography — where",
        "Which states over- and under-perform?",
        "Two views side by side — satisfaction by state and the on-time "
        "vs late split for the highest-volume states.",
    )
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        _chart_card(chart_state_scores, year, states)
    with col_b:
        _chart_card(chart_ontime_by_state, year, states)

    # ── Chapter 7 — Trend ──────────────────────────────────────────────
    _chapter(
        7, "Trend — over time",
        "How has satisfaction moved month to month?",
        "Monthly average review score across the whole period — the "
        "trajectory matters as much as the level.",
    )
    _chart_card(chart_monthly_score, year, states)

    # ── Chapter 8 — Map ────────────────────────────────────────────────
    _chapter(
        8, "Geography — map",
        "Where does the volume sit?",
        "Delivered orders by state on the map — the geographic backbone "
        "of the whole analysis. Selected states are coloured; the rest are greyed.",
    )
    _chart_card(chart_brazil_map, year, states)


# ════════════════════════════════════════════════════════════════════════════
# §12  TAB 3 — DETAIL & METHOD  (Chapters 9-10)
# ════════════════════════════════════════════════════════════════════════════

def render_detail(year: str, states: tuple[str, ...]) -> None:
    st.markdown("## 🔬 Detail & Method")
    st.markdown(
        "The limitations that govern how far you can take these findings — "
        "read before quoting the numbers in a decision."
    )

    # ── Chapter 9 — Limitations ────────────────────────────────────────
    _chapter(
        9, "Limitations",
        "What to keep in mind",
        "Seven caveats that scope every claim in this dashboard.",
    )

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
        "L3 — Review fan-out",
        """<p>The dataset has <b>99,224 review rows across 98,673
        distinct orders</b> — 551 rows of fan-out. The dashboard
        <b>averages review scores per order</b> before analysis to keep
        order counts correct.</p>""",
        variant="warn", tags=["Data quality", "Fan-out"],
    )
    info_card(
        "L4 — Customer identity",
        """<p>The <code>customers</code> table has <b>99,441 rows</b> but
        only <b>96,096 distinct <code>customer_unique_id</code></b>
        values. Any customer count here is order-grain; a person-grain
        figure would use the unique identifier.</p>""",
        variant="info", tags=["Identity", "Grain"],
    )
    info_card(
        "L5 — No delivery-distance variable",
        """<p>The geolocation CSV is deliberately excluded. The analysis
        <b>cannot distinguish long-distance transit from carrier
        underperformance</b>. Adding postal-code centroids and a
        great-circle distance would sharpen the RJ and BA
        diagnostics.</p>""",
        variant="info", tags=["Scope", "Distance"],
    )
    info_card(
        "L6 — Dataset and generalisability boundaries",
        """<p>Findings concern Olist in Brazil during 2016–2018. They
        may not generalise to other countries, marketplaces, logistics
        networks or current operating conditions. The choropleth fetches
        a public GeoJSON at runtime, so map rendering depends on external
        availability.</p>""",
        variant="info", tags=["External validity", "Map proxy"],
    )
    info_card(
        "L7 — Small selections are noisy",
        """<p>Narrow selections (one state in one year) can contain only
        a few hundred reviews. Averages from small samples swing widely —
        read the <code>n=</code> labels before drawing conclusions. The
        punctuality chart flags tiers below <b>n=30</b> with ⚠️.</p>""",
        variant="info", tags=["Sample size", "Filters"],
    )

    # ── Chapter 10 — Diagnostics ───────────────────────────────────────
    _chapter(
        10, "Diagnostics",
        "Whole-dataset integrity checks",
        "Counts computed across the entire dataset — these do not respond "
        "to the filters because they describe the source data itself.",
    )

    frames = load_data()
    reviews = frames["reviews"]
    customers = frames["customers"]

    review_rows = len(reviews)
    distinct_orders = reviews["order_id"].nunique()
    duplicate_rows = review_rows - distinct_orders
    customer_rows = len(customers)
    unique_customers = customers["customer_unique_id"].nunique()

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Reviews fan-out**")
        _st_dataframe(pd.DataFrame([{
            "review_rows": review_rows,
            "distinct_orders": distinct_orders,
            "duplicate_rows": duplicate_rows,
        }]))
    with c2:
        st.markdown("**Customer identity**")
        _st_dataframe(pd.DataFrame([{
            "customer_rows": customer_rows,
            "unique_customers": unique_customers,
        }]))

    model = get_model()
    late = model[model["is_late"]]
    if not late.empty:
        dl = (
            late["order_delivered_customer_date"]
            - late["order_estimated_delivery_date"]
        ).dt.days
        st.markdown("**Days late when late (whole dataset)**")
        _st_dataframe(pd.DataFrame([{
            "late_orders": len(late),
            "avg_days_late": round(float(dl.mean()), 2),
            "worst_days_late": int(dl.max()),
        }]))
        st.caption(
            f"When late, orders run **{dl.mean():.2f} days past estimate** "
            f"on average; the worst case is **{int(dl.max())} days**."
        )


# ════════════════════════════════════════════════════════════════════════════
# §13  TAB 4 — RECOMMENDATIONS  (Chapters 11-12)
# ════════════════════════════════════════════════════════════════════════════

def render_recommendations() -> None:
    st.markdown("## 💡 Recommendations")
    st.markdown(
        "Prioritised actions following from the headline finding. "
        "They are based on the **whole dataset**, so they do not change "
        "when the sidebar filters move — a decision should not flip "
        "because someone narrowed the view."
    )

    # ── Chapter 11 — Prioritisation ────────────────────────────────────
    _chapter(
        11, "Prioritisation",
        "What to do first",
        "Impact × effort — top-left is the highest-leverage quadrant. "
        "R2 and R5 are quick wins; R1 is the largest ceiling.",
    )
    render_priority_matrix()

    # ── Chapter 12 — Action plan ───────────────────────────────────────
    _chapter(
        12, "Action plan",
        "Five prioritised recommendations",
        "Each card carries evidence, action, lead and measures — ready to "
        "hand to the team that owns it.",
    )

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
        variant="warn",
        tags=["Impact: High", "Effort: Medium", "RJ"],
        priority="high",
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
        variant="warn",
        tags=["Impact: High", "Effort: Low", "SLA"],
        priority="high",
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
        variant="info",
        tags=["Impact: Medium", "Effort: Medium", "Forecasting"],
        priority="medium",
    )
    info_card(
        "🟡 R4 — Pilot improvement in Bahia (BA)",
        """<p><b>Evidence.</b> BA scores 3.86 — the lowest of any state
        with 2,000+ orders — across ~3,340 orders.</p>
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
        """<p><b>Evidence.</b> On-time = 4.29; late = 2.57. The damage is
        in the <i>miss</i>, not the speed.</p>
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
    # ── CRITICAL: set_page_config MUST be the first Streamlit call ──────
    st.set_page_config(
        page_title="Customer Satisfaction · Olist",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
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

    year, states = render_sidebar()
    k_live = query_kpis(year, states)   # live values for hero meta

    # ── Hero header — LARGE, layered, attention-grabbing ────────────────
    avg_live = (
        f"{k_live['avg_review_score']:.2f}★"
        if k_live["avg_review_score"] else f"{REF['avg_review_score']:.2f}★"
    )
    st.markdown(
        f"""
        <div class="app-hero">
          <div class="app-hero-eyebrow">PORA Academy · Cohort 7 · Capstone</div>
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
        """,
        unsafe_allow_html=True,
    )

    render_executive_summary(year, states)

    st.write("")

    tab_overview, tab_theme, tab_detail, tab_recs = st.tabs([
        "📈 Overview",
        "⭐ Customer Satisfaction",
        "🔬 Detail & Method",
        "💡 Recommendations",
    ])

    filter_active = year != "All" or bool(states)
    with tab_overview:
        render_overview(year, states, filter_active=filter_active)
    with tab_theme:
        render_theme(year, states)
    with tab_detail:
        render_detail(year, states)
    with tab_recs:
        render_recommendations()

    st.divider()
    st.caption(
        f"**Reference (filters = All):** Total Orders "
        f"{REF['total_orders']:,} · Delivered {REF['delivered_orders']:,} · "
        f"Avg Review {REF['avg_review_score']:.2f} · Late "
        f"{REF['late_deliveries']:,} · Avg Delivery Days "
        f"{REF['avg_delivery_days']:.2f} · Headline gap "
        f"{REF['star_gap']:.2f}★ (on-time {REF['on_time_score']:.2f} vs "
        f"late {REF['late_score']:.2f}). "
        "Every KPI, chart and insight line responds to the Year and State "
        "filters and reconciles to the verified README tables when both "
        "are set to All."
    )


if __name__ == "__main__":
    main()