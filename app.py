# olist_app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis · Group 1 Capstone

Decision-first dashboard · teal/coral palette · cached in-memory pandas
layer · lazy tab rendering · Streamlit Cloud ready.

Deploy:   streamlit run olist_app.py
Cloud:    Streamlit Community Cloud — set OLIST_DATA_DIR if needed.
Requires: streamlit, pandas, matplotlib, requests
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

CHART_FIGSIZE: Final[tuple[float, float]] = (8.2, 4.6)
PARTIAL_YEARS: Final[frozenset[str]] = frozenset({"2016", "2018"})

# Reference figures (filters = All) — single source of truth
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
    """Satisfaction palette: teal (primary) · coral (accent) ·
    amber (warn) · rose (danger) over warm slate neutrals."""
    # Brand
    primary:       str = "#0D9488"
    primary_light: str = "#5EEAD4"
    primary_dark:  str = "#115E59"
    accent:        str = "#F97316"
    accent_light:  str = "#FDBA74"
    accent_dark:   str = "#C2410C"

    # Semantic (used consistently — do not repurpose)
    success:       str = "#10B981"   # on-time, positive
    success_light: str = "#6EE7B7"
    warn:          str = "#F59E0B"   # at risk
    warn_light:    str = "#FCD34D"
    danger:        str = "#E11D48"   # late, negative
    danger_light:  str = "#FDA4AF"

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
    """Load CSVs into memory as pandas frames. Cached for an hour."""
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
    """Pre-joined analytical model — one row per order.

    `frames_fingerprint` auto-invalidates the cache when data changes.
    """
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
    """Stable entry point — fingerprint derived from data shape."""
    frames = load_data()
    fingerprint = hashlib.md5(
        "|".join(f"{k}:{len(v)}" for k, v in sorted(frames.items()))
        .encode()
    ).hexdigest()
    return build_model(fingerprint)


# ── Filter helpers ─────────────────────────────────────────────────────────

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


# ── Formatters ────────────────────────────────────────────────────────────

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
          <div class="empty-body">{msg} Try widening the Year or State filters.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _insight(year: str, states: Sequence[str], text: str) -> None:
    st.markdown(
        f"""
        <div class="insight">
          <span class="insight-scope">{scope_label(year, states)}</span>
          <span class="insight-text">{text}</span>
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
    st.pyplot(fig)


def _partial_note(year: str) -> None:
    if not is_partial_year(year):
        return
    detail = {
        "2016": "2016 has only 3 active months (Sep, Oct, Dec) — "
                "November has zero orders.",
        "2018": "2018 ends 17 Oct — not a full calendar year.",
    }[year]
    st.caption(f"⚠️ **Partial year — {year}.** {detail}")


def _safe_chart(fn, *args) -> None:
    """Render a chart, catching errors so one failure doesn't kill a tab."""
    try:
        fn(*args)
    except Exception as exc:
        _LOGGER.exception("Chart %s failed", fn.__name__)
        st.warning(f"⚠️ This chart could not render: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# §6  CHARTS
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

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.bar(counts.index.astype(str), counts.values, color=shades,
                  edgecolor=PALETTE.white, linewidth=1.2)
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
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.barh(counts.index.astype(str), counts.values, color=colors,
                   edgecolor=PALETTE.white, linewidth=1.2)
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

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
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


def chart_state_scores(year: str, states: tuple[str, ...]) -> None:
    st.markdown("### Average Review Score by State *(top 10)*")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["review_score"].notna() & df["customer_state"].notna()]

    if df.empty:
        _no_data()
        return

    grp = df.groupby("customer_state").agg(
        orders=("order_id", "nunique"),
        avg_score=("review_score", "mean"),
    ).reset_index().sort_values("orders", ascending=False).head(10)

    plot_df = grp.sort_values("avg_score", ascending=True)
    colors = [
        PALETTE.danger if s < 4.00
        else PALETTE.warn if s < 4.10
        else PALETTE.success
        for s in plot_df["avg_score"]
    ]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.barh(plot_df["customer_state"], plot_df["avg_score"],
                   color=colors, edgecolor=PALETTE.white, linewidth=1.2)
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


def chart_punctuality_vs_satisfaction(year: str,
                                      states: tuple[str, ...]) -> None:
    st.markdown("### Delivery Punctuality vs Satisfaction")
    df = apply_filters(get_model(), year, list(states))
    df = df[df["is_delivered"] & df["review_score"].notna()].copy()
    if df.empty:
        _no_data()
        return

    df["status"] = df["is_late"].map({True: "Late", False: "On Time"})
    grp = df.groupby("status").agg(
        orders=("order_id", "nunique"),
        avg_score=("review_score", "mean"),
        avg_days=("delivery_days", "mean"),
    ).reset_index().sort_values("avg_score", ascending=False)

    color_map = {"On Time": PALETTE.success, "Late": PALETTE.danger}
    colors = [color_map.get(s, PALETTE.slate_400) for s in grp["status"]]

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.bar(grp["status"], grp["avg_score"], color=colors,
                  edgecolor=PALETTE.white, linewidth=1.2, width=0.55)
    ax.set_title("Delivery Punctuality vs Satisfaction",
                 pad=14, loc="left")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5.6)
    _style_axes(ax)
    for b, s, n, d in zip(bars, grp["avg_score"], grp["orders"],
                          grp["avg_days"]):
        d_txt = f"{d:.1f}d" if pd.notna(d) else "—"
        ax.text(b.get_x() + b.get_width() / 2, s + 0.08,
                f"{s:.2f}\n(n={int(n):,}; {d_txt} avg)",
                ha="center", va="bottom", fontsize=11,
                color=PALETTE.slate_700, fontweight="bold")

    scores = dict(zip(grp["status"], grp["avg_score"]))
    if {"On Time", "Late"} <= scores.keys():
        on, late = scores["On Time"], scores["Late"]
        gap = on - late
        ax.annotate("", xy=(1, late + 0.1), xytext=(1, on - 0.1),
                    arrowprops=dict(arrowstyle="<->",
                                    color=PALETTE.primary, lw=2.2))
        ax.text(1.18, (on + late) / 2, f"Δ {gap:.2f}★",
                ha="left", va="center", fontsize=14,
                color=PALETTE.primary, fontweight="bold")
    _finish(fig)

    if {"On Time", "Late"} <= scores.keys():
        gap = scores["On Time"] - scores["Late"]
        on_time_orders = int(
            grp.loc[grp["status"] == "On Time", "orders"].iloc[0]
        )
        late_orders = int(
            grp.loc[grp["status"] == "Late", "orders"].iloc[0]
        )
        _insight(
            year, states,
            f"On-time orders ({on_time_orders:,}) average "
            f"{scores['On Time']:.2f}★; late orders ({late_orders:,}) "
            f"average {scores['Late']:.2f}★ — a gap of {gap:.2f}★, "
            f"the largest single driver in this dataset.",
        )
    else:
        only = next(iter(scores))
        _insight(
            year, states,
            f"Only {only.lower()} orders in this selection — no on-time "
            f"vs late gap can be computed.",
        )


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

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    bars = ax.bar(grp["bucket"].astype(str), grp["avg_score"],
                  color=colors, edgecolor=PALETTE.white, linewidth=1.2)
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

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
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
    mode = (
        "Selected states coloured; rest greyed."
        if filter_active else
        "All 27 states coloured by delivered count."
    )
    _insight(year, states,
             f"{lo_s} ({int(lo_v):,}) fewest; {hi_s} ({int(hi_v):,}) most "
             f"in scope. {mode}")


# ════════════════════════════════════════════════════════════════════════════
# §7  UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

def info_card(title: str, body: str, *, variant: str = "info",
              tags: Sequence[str] | None = None,
              priority: str = "") -> None:
    """Info card with optional priority bar (high/medium/low)."""
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
    if g["gap"] is None:
        st.markdown(
            """
            <div class="hero-banner">
              <div class="hero-icon">📊</div>
              <div class="hero-text">
                <div class="hero-label">Headline finding</div>
                <p class="hero-headline">
                  Delivery punctuality is the strongest satisfaction driver
                  in this dataset.
                </p>
                <p class="hero-sub">
                  Only one delivery status appears in the current selection —
                  widen the filters to compare on-time and late orders.
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="hero-banner">
          <div class="hero-icon">⚡</div>
          <div class="hero-text">
            <div class="hero-label">Headline finding · {scope_label(year, states)}</div>
            <p class="hero-headline">
              Late deliveries cost <b>{g["gap"]:.2f} stars</b>
              — the single strongest satisfaction driver.
            </p>
            <p class="hero-sub">
              On-time orders average <b>{g["on_time"]:.2f}</b> stars;
              late orders average <b>{g["late"]:.2f}</b> stars.
              This gap dwarfs every state-level difference.
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


# ════════════════════════════════════════════════════════════════════════════
# §8  GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS: Final[str] = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
  :root {
    --brand: #0D9488;
    --brand-light: #5EEAD4;
    --brand-dark: #115E59;
    --accent: #F97316;
    --success: #10B981;
    --warn: #F59E0B;
    --danger: #E11D48;
    --ink: #0F172A;
    --ink-soft: #1E293B;
    --slate-700: #334155;
    --slate-600: #475569;
    --slate-500: #64748B;
    --slate-400: #94A3B8;
    --slate-300: #CBD5E1;
    --slate-200: #E2E8F0;
    --slate-100: #F1F5F9;
    --slate-50: #F8FAFC;

    /* Spacing scale */
    --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px;
    --s5: 24px; --s6: 32px; --s7: 48px;
  }

  html, body, [class*="css"], .stApp {
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 15px;
    line-height: 1.6;
    color: var(--ink);
  }
  .stApp { background: var(--slate-50); }
  code, .stCodeBlock { font-family: 'JetBrains Mono', monospace !important; }

  /* Typography scale */
  h1 { color: var(--ink); font-weight: 800; letter-spacing: -0.025em;
       font-size: 34px; line-height: 1.2; }
  h2 { color: var(--ink); font-weight: 700; letter-spacing: -0.018em;
       font-size: 24px; line-height: 1.25;
       margin-top: var(--s2) !important; margin-bottom: var(--s4) !important; }
  h3 { color: var(--ink-soft); font-weight: 700; letter-spacing: -0.012em;
       font-size: 18px; line-height: 1.3;
       margin-top: 0 !important; margin-bottom: var(--s3) !important; }
  h4 { color: var(--brand); font-size: 16px; font-weight: 700;
       line-height: 1.35; margin-bottom: var(--s3) !important; }
  p, li, span, label { color: var(--slate-700); }

  .block-container {
    padding-top: var(--s5) !important;
    padding-bottom: var(--s6) !important;
    max-width: 1500px;
  }
  section[data-testid="stSidebar"] { padding-top: var(--s4); }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #020617 100%);
    border-right: 1px solid #334155;
  }
  section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important; font-weight: 700;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label {
    color: #94A3B8 !important; font-weight: 600;
    text-transform: uppercase; font-size: 11px; letter-spacing: 0.08em;
  }
  section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
  }
  section[data-testid="stSidebar"] table { color: #E2E8F0 !important; }
  section[data-testid="stSidebar"] th,
  section[data-testid="stSidebar"] td {
    color: #E2E8F0 !important; border-color: #334155 !important;
    font-size: 13px !important; padding: var(--s2) var(--s3) !important;
  }
  section[data-testid="stSidebar"] .stButton > button {
    background: var(--brand) !important; color: #FFFFFF !important;
    border-radius: var(--s2); font-weight: 600; border: none;
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
    font-size: 13px;
    line-height: 1.5;
  }
  .sidebar-warning strong { color: #FCD34D !important; }
  .sidebar-warning span   { color: #E2E8F0 !important; }

  .sidebar-headline {
    background: rgba(13, 148, 136, 0.18);
    border: 1px solid rgba(94, 234, 212, 0.45);
    border-left: 3px solid #5EEAD4;
    border-radius: var(--s2);
    padding: var(--s3) var(--s4);
    margin-top: var(--s4);
    line-height: 1.55;
  }
  .sidebar-headline-label {
    font-size: 10px; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: #5EEAD4 !important;
    margin-bottom: var(--s2);
  }
  .sidebar-headline strong {
    color: #FFFFFF !important; font-size: 14px; display: block;
    margin-bottom: var(--s1);
  }
  .sidebar-headline span { color: #CBD5E1 !important; font-size: 12px; }
  .sidebar-headline span b { color: #FFFFFF !important; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid var(--slate-200);
    border-left: 5px solid var(--brand);
    border-radius: var(--s3);
    padding: var(--s5) var(--s5);
    box-shadow: 0 1px 3px rgba(15,23,42,0.06),
                0 1px 2px rgba(15,23,42,0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(13,148,136,0.14);
  }
  [data-testid="stMetricLabel"] {
    color: var(--slate-500) !important; font-size: 12px !important;
    font-weight: 600 !important; text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  [data-testid="stMetricValue"] {
    color: var(--ink) !important; font-weight: 800 !important;
    font-size: 30px !important; letter-spacing: -0.02em;
    line-height: 1.15;
  }
  [data-testid="stMetricDelta"] {
    font-weight: 600 !important; font-size: 13px !important;
  }

  /* Tabs */
  button[data-baseweb="tab"] {
    color: var(--slate-500); font-weight: 600; font-size: 15px;
    padding: var(--s3) var(--s5) !important;
    border-radius: var(--s2) var(--s2) 0 0;
    transition: background 0.15s ease;
  }
  button[data-baseweb="tab"]:hover {
    background: var(--slate-100);
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--brand); border-bottom-color: var(--brand);
    border-bottom-width: 3px;
  }
  [data-baseweb="tab-highlight"] { background-color: var(--brand); }

  /* Buttons */
  .stButton > button {
    background: var(--brand); color: #FFFFFF;
    border: none; border-radius: var(--s2);
    padding: var(--s2) var(--s5); font-weight: 600; font-size: 14px;
    transition: background 0.15s ease, transform 0.1s ease;
  }
  .stButton > button:hover {
    background: var(--brand-dark); transform: translateY(-1px);
  }

  [data-testid="stAlert"] {
    background-color: #FEF3C7;
    border-left: 4px solid var(--warn);
    color: #78350F; border-radius: var(--s3);
  }

  hr { border-color: var(--slate-200); margin: var(--s5) 0; }

  /* Info cards */
  .info-card {
    position: relative;
    background: #FFFFFF;
    border: 1px solid var(--slate-200);
    border-left: 5px solid var(--brand);
    border-radius: var(--s3);
    padding: var(--s5) var(--s5);
    margin-bottom: var(--s4);
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  .info-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15,23,42,0.08);
  }
  .info-card.warn { border-left-color: var(--danger); }
  .info-card.ok   { border-left-color: var(--success); }
  .info-card.info { border-left-color: var(--brand); }
  .info-card.warn h4 { color: var(--danger); }
  .info-card.ok h4   { color: var(--success); }
  .info-card.info h4 { color: var(--brand); }
  .info-card p, .info-card li { color: var(--slate-700); line-height: 1.65; }
  .info-card .tag {
    display: inline-block; background: var(--slate-100);
    color: var(--slate-600); border-radius: 999px;
    padding: var(--s1) var(--s3); font-size: 12px;
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

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 55%, #99F6E4 100%);
    border-radius: var(--s4);
    padding: var(--s5) var(--s6);
    margin-bottom: var(--s5);
    border: 1px solid #5EEAD4;
    box-shadow: 0 6px 20px rgba(13,148,136,0.12);
    display: flex;
    align-items: center;
    gap: var(--s5);
  }
  .hero-banner .hero-icon { font-size: 42px; line-height: 1; flex-shrink: 0; }
  .hero-banner .hero-text { flex: 1; }
  .hero-banner .hero-label {
    font-size: 11px; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--brand);
    margin-bottom: var(--s2);
  }
  .hero-banner .hero-headline {
    font-size: 22px; font-weight: 800; color: var(--ink);
    line-height: 1.32; margin: 0 0 var(--s2) 0;
    letter-spacing: -0.02em;
  }
  .hero-banner .hero-headline b { color: var(--brand-dark); font-weight: 900; }
  .hero-banner .hero-sub {
    font-size: 14px; color: var(--slate-600); line-height: 1.55;
  }
  .hero-banner .hero-sub b { color: var(--ink-soft); font-weight: 700; }

  /* Partial year banner */
  .overview-banner {
    background: #FEF3C7;
    border-left: 4px solid var(--warn);
    border-radius: var(--s3);
    padding: var(--s3) var(--s5);
    margin-bottom: var(--s4);
    font-size: 14px; color: #78350F; line-height: 1.55;
  }
  .overview-banner b { color: #92400E; }

  /* Scope pill */
  .scope-pill {
    display: inline-block; background: var(--brand); color: #FFFFFF;
    border-radius: 999px; padding: var(--s2) var(--s4);
    font-size: 13px; font-weight: 600;
    margin-bottom: var(--s2);
    box-shadow: 0 2px 8px rgba(13,148,136,0.25);
  }

  /* Insight line */
  .insight {
    display: flex; flex-direction: column; gap: var(--s1);
    background: var(--slate-50);
    border-left: 3px solid var(--brand-light);
    border-radius: var(--s2);
    padding: var(--s3) var(--s4);
    margin-top: var(--s2);
    font-size: 13px;
  }
  .insight-scope {
    font-size: 10px; font-weight: 700; color: var(--brand);
    text-transform: uppercase; letter-spacing: 0.06em;
    opacity: 0.85;
  }
  .insight-text { color: var(--slate-600); line-height: 1.55; }

  /* Empty state */
  .empty-state {
    text-align: center;
    padding: var(--s7) var(--s5);
    background: var(--slate-50);
    border: 1px dashed var(--slate-300);
    border-radius: var(--s3);
  }
  .empty-icon { font-size: 40px; }
  .empty-title {
    font-size: 18px; font-weight: 700; color: var(--slate-700);
    margin: var(--s2) 0;
  }
  .empty-body { color: var(--slate-500); font-size: 14px; }

  /* Dataframe */
  [data-testid="stDataFrame"] {
    border-radius: var(--s3); overflow: hidden;
    border: 1px solid var(--slate-200);
  }
  .stPyplot { margin-top: var(--s1); }

  /* Caption */
  .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--slate-500); font-size: 13px;
  }

  /* Mobile breakpoints */
  @media (max-width: 768px) {
    .block-container { padding: var(--s4) !important; }
    h1 { font-size: 26px; }
    h2 { font-size: 20px; }
    [data-testid="stMetricValue"] { font-size: 24px !important; }
    .hero-banner { flex-direction: column; text-align: left;
                   padding: var(--s4); gap: var(--s3); }
    .hero-banner .hero-headline { font-size: 18px; }
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
        if st.sidebar.button("↺  Reset filters", use_container_width=True):
            st.session_state["filter_year"] = "All"
            st.session_state["filter_states"] = []
            st.rerun()

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
# §10  TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

def render_overview(year: str, states: tuple[str, ...],
                    filter_active: bool) -> None:
    st.markdown("## Overview")

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
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders", fmt_int(k["total_orders"]))
    c2.metric("Delivered Orders", fmt_int(k["delivered_orders"]))

    if k["avg_review_score"] is None:
        c3.metric("Avg Review Score", "—")
    else:
        delta = (
            f"{k['avg_review_score'] - REF['avg_review_score']:+.2f} vs ref"
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

    st.divider()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        with st.spinner("Rendering distribution…"):
            _safe_chart(chart_review_distribution, year, states)
    with col_right:
        with st.spinner("Rendering volume by year…"):
            _safe_chart(chart_review_volume_by_year, year, states)


# ════════════════════════════════════════════════════════════════════════════
# §11  TAB 2 — CUSTOMER SATISFACTION
# ════════════════════════════════════════════════════════════════════════════

def render_theme(year: str, states: tuple[str, ...]) -> None:
    st.markdown("## Customer Satisfaction")

    render_headline_hero(year, states)
    render_decision_snapshot(year, states)

    st.divider()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        with st.spinner("Scoring states…"):
            _safe_chart(chart_state_scores, year, states)
    with col_b:
        with st.spinner("Mapping Brazil…"):
            _safe_chart(chart_brazil_map, year, states)

    st.divider()

    col_c, col_d = st.columns([1, 1])
    with col_c:
        with st.spinner("Comparing on-time vs late…"):
            _safe_chart(chart_ontime_by_state, year, states)
    with col_d:
        with st.spinner("Bucketing delivery times…"):
            _safe_chart(chart_satisfaction_by_delivery_bucket, year, states)

    st.divider()

    col_e, col_f = st.columns([1, 1])
    with col_e:
        with st.spinner("Computing punctuality driver…"):
            _safe_chart(chart_punctuality_vs_satisfaction, year, states)
    with col_f:
        with st.spinner("Trending months…"):
            _safe_chart(chart_monthly_score, year, states)


# ════════════════════════════════════════════════════════════════════════════
# §12  TAB 3 — DETAIL & METHOD
# ════════════════════════════════════════════════════════════════════════════

def render_detail(year: str, states: tuple[str, ...]) -> None:
    st.markdown("## Detail & Method")
    st.markdown("### Data limitations")

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
        read the <code>n=</code> labels before drawing conclusions.</p>""",
        variant="info", tags=["Sample size", "Filters"],
    )

    st.divider()
    st.markdown("### Whole-dataset diagnostics")

    frames = load_data()
    reviews = frames["reviews"]
    customers = frames["customers"]

    review_rows = len(reviews)
    distinct_orders = reviews["order_id"].nunique()
    duplicate_rows = review_rows - distinct_orders
    customer_rows = len(customers)
    unique_customers = customers["customer_unique_id"].nunique()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Reviews fan-out**")
        st.dataframe(
            pd.DataFrame([{
                "review_rows": review_rows,
                "distinct_orders": distinct_orders,
                "duplicate_rows": duplicate_rows,
            }]),
            hide_index=True,
            width="stretch",
        )
    with c2:
        st.markdown("**Customer identity**")
        st.dataframe(
            pd.DataFrame([{
                "customer_rows": customer_rows,
                "unique_customers": unique_customers,
            }]),
            hide_index=True,
            width="stretch",
        )

    model = get_model()
    late = model[model["is_late"]]
    if not late.empty:
        dl = (
            late["order_delivered_customer_date"]
            - late["order_estimated_delivery_date"]
        ).dt.days
        st.markdown("**Days late when late (whole dataset)**")
        st.dataframe(
            pd.DataFrame([{
                "late_orders": len(late),
                "avg_days_late": round(float(dl.mean()), 2),
                "worst_days_late": int(dl.max()),
            }]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"When late, orders run **{dl.mean():.2f} days past estimate** "
            f"on average; the worst case is **{int(dl.max())} days**."
        )


# ════════════════════════════════════════════════════════════════════════════
# §13  TAB 4 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════

def render_recommendations() -> None:
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
    st.set_page_config(
        page_title="Customer Satisfaction · Olist",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_global_style()

    try:
        load_data()
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.exception("Data load failed")
        st.error(f"❌ **Data load failed.** {exc}")
        st.stop()

    year, states = render_sidebar()
    filter_active = year != "All" or bool(states)

    st.markdown(
        """
        <h1 style="color:#0F172A; font-size:34px; font-weight:800;
                   letter-spacing:-0.03em; margin-bottom:4px; padding-bottom:0;">
          Customer Satisfaction Dashboard
        </h1>
        <p style='color:#64748B; font-size:15px; margin-top:0;'>
          PORA Academy Cohort 7 — Group 1 Capstone Project
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="scope-pill">Scope: {scope_label(year, states)}</span>',
        unsafe_allow_html=True,
    )

    st.write("")

    tab_overview, tab_theme, tab_detail, tab_recs = st.tabs([
        "📈 Overview",
        "⭐ Customer Satisfaction",
        "🔬 Detail & Method",
        "💡 Recommendations",
    ])

    with tab_overview:
        render_overview(year, states, filter_active)
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