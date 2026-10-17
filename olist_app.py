# olist_app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis and AI Automation

Executive-grade Streamlit dashboard with:
  • Animated "screen projection" executive headline (CSS-only)
  • Filter-proof guards (no crashes on empty/invalid selections)
  • 6 curated charts
  • 8 tabs: Overview, Customer Satisfaction, Detail & Method,
            Limitations, Recommendations, Departments, Future Research,
            Executive Report
  • Fully connected Year + State filters on every KPI and chart
"""

import os
import json
import sqlite3

import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import requests


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Page config (must be the first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Satisfaction | Olist Executive Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — includes keyframe animations for the projection headline
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
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
        font-weight: 600;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.06em;
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
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(109, 63, 181, 0.08);
      }
      [data-testid="stMetricLabel"] {
        color: #6D3FB5;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }
      [data-testid="stMetricValue"] {
        color: #1F2937; font-weight: 700; font-size: 28px;
      }

      button[data-baseweb="tab"] {
        color: #6B7280; font-weight: 600; font-size: 14px;
      }
      button[data-baseweb="tab"][aria-selected="true"] {
        color: #6D3FB5;
        border-bottom-color: #8B5CF6;
        border-bottom-width: 3px;
      }

      hr { border-color: #E5DEF7; }
      [data-testid="stAlert"] {
        background-color: #F5F0FF;
        border-left: 4px solid #8B5CF6;
        color: #4C1D95;
      }
      code {
        background-color: #F5F0FF;
        color: #6D3FB5;
        padding: 2px 6px;
        border-radius: 4px;
      }

      .info-card {
        background: #FFFFFF;
        border: 1px solid #E5DEF7;
        border-left: 5px solid #8B5CF6;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 3px 12px rgba(109, 63, 181, 0.08);
      }
      .info-card.warn { border-left-color: #D64545; }
      .info-card.ok   { border-left-color: #16A34A; }
      .info-card.info { border-left-color: #0EA5E9; }
      .info-card h4 {
        color: #4C1D95; margin-top: 0; margin-bottom: 8px;
        font-size: 16px;
      }
      .info-card p, .info-card li { color: #3F3D56; line-height: 1.7; }
      .info-card .tag {
        display: inline-block;
        background: #F5F0FF;
        color: #6D3FB5;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
      }

      .report-body {
        background: #FFFFFF;
        border: 1px solid #E5DEF7;
        border-radius: 12px;
        padding: 26px 34px;
        margin-bottom: 20px;
        box-shadow: 0 3px 14px rgba(109, 63, 181, 0.08);
        line-height: 1.75;
        color: #3F3D56;
      }
      .report-body h2 {
        color: #4C1D95;
        border-bottom: 2px solid #EDE7FA;
        padding-bottom: 6px;
        margin-top: 28px;
        margin-bottom: 14px;
      }
      .report-body h3 {
        color: #6D3FB5;
        margin-top: 22px;
        margin-bottom: 10px;
      }
      .report-body table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
        font-size: 14px;
      }
      .report-body table th {
        background: #F5F0FF;
        color: #4C1D95;
        padding: 10px 12px;
        text-align: left;
        border-bottom: 2px solid #EDE7FA;
      }
      .report-body table td {
        padding: 10px 12px;
        border-bottom: 1px solid #F0EBFA;
        vertical-align: top;
      }
      .report-body .callout {
        background: linear-gradient(90deg, #F5F0FF 0%, #EDE7FA 100%);
        border-left: 4px solid #6D3FB5;
        padding: 14px 20px;
        border-radius: 8px;
        margin: 16px 0;
      }

      /* ═══════════════════════════════════════════════════════════════
         ANIMATED SCREEN-PROJECTION HEADLINE
         ═══════════════════════════════════════════════════════════════ */

      /* ── Global keyframes ─────────────────────────────────────────── */
      @keyframes projectionFadeIn {
        0%   { opacity: 0; transform: translateY(14px) scale(0.97); filter: blur(6px); }
        60%  { opacity: 0.9; filter: blur(1px); }
        100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
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

      /* ── Projection frame ─────────────────────────────────────────── */
      .projection-frame {
        position: relative;
        overflow: hidden;
        background:
          radial-gradient(circle at 15% 20%, rgba(139,92,246,0.10) 0%, transparent 45%),
          radial-gradient(circle at 85% 80%, rgba(214,69,69,0.07) 0%, transparent 45%),
          linear-gradient(120deg, #FBF8FF 0%, #F2ECFB 60%, #EFE9F7 100%);
        border: 1px solid #E5DEF7;
        border-radius: 14px;
        padding: 28px 36px 26px 36px;
        margin: 8px 0 22px 0;
        animation: projectionGlow 4.5s ease-in-out infinite;
      }

      /* Moving scan line for the "projection" feel */
      .projection-frame::after {
        content: "";
        position: absolute;
        left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg,
                     transparent 0%,
                     rgba(139,92,246,0.55) 50%,
                     transparent 100%);
        filter: blur(0.5px);
        animation: scanLine 3.2s ease-in-out infinite;
        pointer-events: none;
      }

      /* Content of the projection frame */
      .projection-content {
        position: relative;
        z-index: 2;
      }

      .projection-title {
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #6D3FB5;
        margin: 0 0 6px 0;
        opacity: 0;
        animation: projectionFadeIn 0.9s ease-out 0.05s forwards;
      }

      .projection-underline {
        height: 2px;
        background: linear-gradient(90deg, #6D3FB5 0%, #8B5CF6 60%, transparent 100%);
        margin: 0 0 18px 0;
        transform: scaleX(0);
        transform-origin: left;
        animation: underlineSweep 1.1s ease-out 0.35s forwards;
      }

      .projection-line {
        font-size: 17px;
        color: #3F3D56;
        line-height: 1.85;
        margin: 0 0 10px 0;
        opacity: 0;
        animation: projectionFadeIn 1.0s ease-out forwards;
      }
      .projection-line.l1 { animation-delay: 0.55s; }
      .projection-line.l2 { animation-delay: 1.15s; }
      .projection-line.l3 { animation-delay: 1.75s; }

      .projection-line .hl {
        color: #6D3FB5;
        font-weight: 700;
      }

      .projection-number {
        display: inline-block;
        color: #D64545;
        font-size: 24px;
        font-weight: 800;
        padding: 0 4px;
        letter-spacing: -0.01em;
        animation: numberPulse 2.4s ease-in-out infinite;
      }
      .projection-number.small {
        font-size: 20px;
      }
      .projection-number.purple {
        color: #6D3FB5;
        animation: none;
      }

      .projection-priority {
        margin-top: 16px;
        padding: 14px 20px;
        background: linear-gradient(90deg,
                    rgba(139,92,246,0.10) 0%,
                    rgba(139,92,246,0.04) 100%);
        border-left: 4px solid #8B5CF6;
        border-radius: 8px;
        opacity: 0;
        transform: translateX(-24px);
        animation: slideInLeft 1.1s ease-out 2.4s forwards;
      }

      .projection-priority .priority-label {
        display: inline-block;
        font-weight: 800;
        color: #4C1D95;
        letter-spacing: 0.05em;
        margin-right: 6px;
      }

      .projection-priority .priority-text {
        font-size: 15.5px;
        color: #3F3D56;
        line-height: 1.7;
      }

      .projection-caveat {
        display: block;
        margin-top: 10px;
        color: #6B7280;
        font-style: italic;
        font-size: 14px;
        opacity: 0;
        animation: projectionFadeIn 1.0s ease-out 3.1s forwards;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Global matplotlib style ────────────────────────────────────────────────
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


# ── Palette ────────────────────────────────────────────────────────────────
PURPLE       = "#8B5CF6"
PURPLE_DARK  = "#6D3FB5"
RED          = "#D64545"
GREY         = "#9CA3AF"
GREEN        = "#16A34A"
BLUE         = "#0EA5E9"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LAYER (fully guarded)
# ════════════════════════════════════════════════════════════════════════════

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get(
    "OLIST_DATA_DIR",
    os.path.join(_HERE, "data")
)


TABLES = {
    "orders":      "olist_orders_dataset.csv",
    "customers":   "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments":    "olist_order_payments_dataset.csv",
    "reviews":     "olist_order_reviews_dataset.csv",
}


@st.cache_resource
def get_conn():
    """Create one shared in-memory SQLite connection with all Olist tables."""
    missing_files = [
        filename
        for filename in TABLES.values()
        if not os.path.isfile(os.path.join(DATA_DIR, filename))
    ]

    if missing_files:
        raise FileNotFoundError(
            f"DATA_DIR is set to: {DATA_DIR}. "
            f"Missing CSV files: {missing_files}"
        )

    conn = sqlite3.connect(":memory:", check_same_thread=False)

    for table_name, filename in TABLES.items():
        file_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(file_path)
        df.to_sql(table_name, conn, index=False, if_exists="replace")

    return conn


try:
    conn = get_conn()
except Exception as _e:
    st.error(
        f"❌ **Data load failed.** {_e}\n\n"
        f"Expected CSVs at: `{DATA_DIR}`"
    )
    st.stop()


# ── Filter contract helpers ────────────────────────────────────────────────
def year_params(year):              return (year, year)
def multi_params(multi_json):       return (multi_json, multi_json)
def year_multi(year, multi_json):   return (year, year, multi_json, multi_json)


# ── Safe parameter builder ─────────────────────────────────────────────────
def safe_state_json(raw_states):
    """Build a JSON array string for the state filter with full guards."""
    try:
        if raw_states is None:
            return "[]"
        if not isinstance(raw_states, (list, tuple, set)):
            return "[]"
        cleaned = [s for s in raw_states if isinstance(s, str) and s]
        if not cleaned or "All" in cleaned:
            return "[]"
        return json.dumps(sorted(set(cleaned)))
    except Exception:
        return "[]"


# ── Safe query wrapper ─────────────────────────────────────────────────────
def safe_query(conn, sql, params=(), *, label="query"):
    """Run a query and always return a DataFrame."""
    try:
        return pd.read_sql(sql, conn, params=params)
    except Exception as _e:
        st.warning(
            f"⚠️ {label} failed gracefully — showing empty result. "
            f"Details: {type(_e).__name__}"
        )
        return pd.DataFrame()


# ── Safe scalar helper ─────────────────────────────────────────────────────
def safe_scalar(df, col, default=None):
    """Safely read a single scalar value from a DataFrame's first row."""
    try:
        if df is None or df.empty:
            return default
        val = df.iloc[0][col]
        if pd.isna(val):
            return default
        return val
    except Exception:
        return default


# ── The 27 Brazilian state codes ───────────────────────────────────────────
ALL_STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
    "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIDEBAR (fully guarded)
# ════════════════════════════════════════════════════════════════════════════
STATES = ["All"] + ALL_STATES

st.sidebar.title("📊 Executive Filters")
st.sidebar.markdown(
    "Both filters apply to every KPI card and every chart. "
    "Select *All* to reproduce the unfiltered reference values."
)

try:
    selected_year = st.sidebar.selectbox(
        "Reporting Year", ["All", "2016", "2017", "2018"], index=0
    )
except Exception:
    selected_year = "All"

try:
    selected_state = st.sidebar.multiselect(
        "Customer State",
        options=STATES,
        default=["All"],
        help='Select one or more states. "All" (default) applies no state filter.',
    )
except Exception:
    selected_state = ["All"]


# ── Normalise the multiselect result ───────────────────────────────────────
if not selected_state:
    selected_state = ["All"]
if "All" in selected_state and len(selected_state) > 1:
    selected_state = ["All"]

state_json = safe_state_json(selected_state)

if selected_year not in ("All", "2016", "2017", "2018"):
    selected_year = "All"

filter_active = (selected_year != "All") or ("All" not in selected_state)


st.sidebar.divider()
st.sidebar.markdown("**Reference — filters = All**")
st.sidebar.markdown(
    "| Indicator | Value |\n"
    "|---|---|\n"
    "| Total Orders | 99,441 |\n"
    "| Delivered | 96,478 |\n"
    "| Avg Review | 4.09 |\n"
    "| Late | 7,826 |\n"
    "| Avg Delivery Days | 12.56 |"
)

if filter_active:
    state_display = ", ".join(s for s in selected_state if s != "All") or "All"
    st.sidebar.markdown(
        f"**Active filter:** Year = `{selected_year}` · "
        f"States = `{state_display}`"
    )
else:
    st.sidebar.markdown("**Active filter:** `All` (unfiltered reference)")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — KPI ROW (guarded, connected to both filters)
# ════════════════════════════════════════════════════════════════════════════
def kpi_values(conn, year, state_json):
    """Executive KPI row — connected to BOTH Year and State filters."""
    df = safe_query(conn, """
        SELECT COUNT(DISTINCT o.order_id) AS total_orders,
               COUNT(DISTINCT CASE WHEN o.order_status='delivered'
                                    THEN o.order_id END) AS delivered_orders,
               ROUND(AVG(CAST(r.review_score AS REAL)),2)
                   AS avg_review_score,
               SUM(CASE WHEN o.order_status='delivered'
                         AND o.order_delivered_customer_date
                              > o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS late_deliveries,
               ROUND(AVG(CASE WHEN o.order_status='delivered'
                                AND o.order_delivered_customer_date IS NOT NULL
                              THEN julianday(o.order_delivered_customer_date)
                                 - julianday(o.order_purchase_timestamp) END),2)
                   AS avg_delivery_days
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
    """, params=year_multi(year, state_json), label="KPI query")

    return {
        "total_orders":      safe_scalar(df, "total_orders", 0),
        "delivered_orders":  safe_scalar(df, "delivered_orders", 0),
        "avg_review_score":  safe_scalar(df, "avg_review_score"),
        "late_deliveries":   safe_scalar(df, "late_deliveries", 0),
        "avg_delivery_days": safe_scalar(df, "avg_delivery_days"),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════════════════
def no_data_notice(msg="No data for the current filter selection."):
    st.info(f"ℹ️ {msg} Try widening the reporting year or state filters.")


def safe_metric_display(value, fmt="{:,.0f}", default="—"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return fmt.format(value)
    except Exception:
        return str(value)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CURATED CHARTS (6 total, all guard-protected)
# ════════════════════════════════════════════════════════════════════════════

# ── Chart 1 — Review Score Distribution (whole-dataset) ────────────────────
def chart_review_distribution(conn):
    df = safe_query(conn, """
        SELECT CAST(review_score AS INTEGER) AS score,
               COUNT(*)                     AS reviews,
               ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM reviews),1) AS pct
        FROM reviews GROUP BY 1 ORDER BY 1
    """, label="Review distribution")

    if df.empty:
        no_data_notice("No reviews found in the dataset.")
        return

    shades = []
    for s in df["score"]:
        if s == 5:   shades.append("#6D3FB5")
        elif s == 1: shades.append("#D64545")
        else:        shades.append("#C4B5FD")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(df["score"].astype(str), df["reviews"], color=shades)
    ax.set_title("Review Score Distribution (Whole Dataset)")
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Number of reviews")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar, pct in zip(bars, df["pct"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10,
                color="#4C1D95", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Executive insight.** The distribution is bimodal — 57.8% of "
        "reviews are five-star, 11.5% are one-star. Deliberately unfiltered: "
        "a whole-dataset structural fact."
    )


# ── Chart 2 — Monthly Average Review Score ─────────────────────────────────
def chart_monthly_score(conn, year, state_json):
    df = safe_query(conn, """
        SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
               COUNT(*) AS reviews,
               ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
        GROUP BY month ORDER BY month
    """, params=year_multi(year, state_json), label="Monthly score")

    if df.empty:
        no_data_notice()
        return

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["month"], df["avg_score"], marker="o", color=PURPLE_DARK,
            linewidth=2.5, markersize=5)
    ax.set_title("Monthly Average Review Score")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average review score (1–5)")
    ax.set_ylim(3.5, 4.6)
    plt.xticks(rotation=45, ha="right")

    if "2017-11" in df["month"].values:
        nov = df.loc[df["month"] == "2017-11", "avg_score"].iloc[0]
        ax.scatter(["2017-11"], [nov], s=160, color=RED, zorder=5,
                   edgecolor="white", linewidth=1.8)
        ax.annotate(
            f"Nov 2017\n{nov:.2f}",
            xy=("2017-11", nov), xytext=(-55, -35),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2),
            fontsize=10, color=RED, fontweight="bold",
        )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Executive insight.** Average scores hold in the 4.0–4.2 band "
        "except November 2017. November 2016 is absent — zero orders that "
        "month, a genuine data gap, not missing values."
    )


# ── Chart 3 — Brazil choropleth (top 10, labelled) ─────────────────────────
@st.cache_data(show_spinner=False)
def _load_brazil_geojson():
    urls = [
        ("https://raw.githubusercontent.com/codeforamerica/"
         "click_that_hood/master/public/data/brazil-states.geojson"),
        ("https://raw.githubusercontent.com/tbrugz/"
         "vector-json/master/brazil-states.geojson"),
    ]
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not fetch Brazil GeoJSON from any mirror. "
        f"Last error: {last_err}"
    )


@st.cache_data(show_spinner=False)
def _state_centroids():
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


def chart_brazil_map(conn, year, state_json):
    all_states_df = safe_query(conn, """
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

    if all_states_df.empty:
        no_data_notice("No states with review data for this reporting year.")
        return

    try:
        selected = json.loads(state_json) if state_json else []
        if not isinstance(selected, list):
            selected = []
    except Exception:
        selected = []

    filter_active_local = len(selected) > 0
    top10 = all_states_df["state"].tolist()
    keep_states = sorted(set(top10) | set(selected)) if filter_active_local else top10

    kept_df = pd.DataFrame({"state": keep_states}).merge(
        all_states_df[["state", "orders", "avg_score"]],
        on="state", how="left",
    )
    kept_df["orders"]    = kept_df["orders"].fillna(0).astype(int)
    kept_df["avg_score"] = kept_df["avg_score"].fillna(0)

    if filter_active_local:
        kept_df["display_score"] = kept_df.apply(
            lambda row: row["avg_score"] if row["state"] in selected else -1,
            axis=1,
        )
    else:
        kept_df["display_score"] = kept_df["avg_score"]

    def hover_label(row):
        if filter_active_local and row["state"] not in selected:
            return "Not in current filter"
        if row["orders"] == 0:
            return "No orders"
        return f"{row['avg_score']:.2f} average · {row['orders']:,} orders"

    kept_df["hover"] = kept_df.apply(hover_label, axis=1)

    centroids = _state_centroids()
    kept_df["lat"] = kept_df["state"].map(
        lambda s: centroids.get(s, (None, None))[0]
    )
    kept_df["lon"] = kept_df["state"].map(
        lambda s: centroids.get(s, (None, None))[1]
    )

    if filter_active_local:
        label_df = kept_df[kept_df["state"].isin(selected)].dropna(
            subset=["lat", "lon"]
        )
    else:
        label_df = kept_df.dropna(subset=["lat", "lon"])

    label_df = label_df.assign(
        label=label_df.apply(
            lambda r: (f"{r['state']}\n{r['avg_score']:.2f}"
                       if r["orders"] > 0 else r["state"]),
            axis=1,
        )
    )

    try:
        geojson = _load_brazil_geojson()
    except Exception as _e:
        st.warning(
            f"⚠️ Could not load Brazil GeoJSON — map unavailable. "
            f"Reason: {_e}"
        )
        return

    color_scale = [
        [0.00, "#3F3F5A"], [0.05, "#3F3F5A"], [0.10, "#B83232"],
        [0.45, "#F0A0A0"], [0.65, "#C4B5FD"], [1.00, "#6D3FB5"],
    ]

    try:
        fig = px.choropleth(
            kept_df,
            geojson=geojson,
            locations="state",
            featureidkey="properties.sigla",
            color="display_score",
            color_continuous_scale=color_scale,
            range_color=(-1, 4.3),
            scope="south america",
            labels={"display_score": "Average review score",
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
                lat=label_df["lat"],
                lon=label_df["lon"],
                text=label_df["label"],
                mode="text",
                textfont=dict(size=11, color="#1F2937", family="sans-serif"),
                hoverinfo="skip",
                showlegend=False,
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
                font=dict(size=14, color="#4C1D95", family="sans-serif"),
            ),
            margin=dict(l=0, r=0, t=48, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(
                title="Average score",
                tickfont=dict(color="#4C1D95"),
                title_font=dict(color="#4C1D95"),
                thickness=14, len=0.7,
                tickvals=[-1, 3.9, 4.0, 4.1, 4.2, 4.3],
                ticktext=["Not selected", "3.9", "4.0", "4.1", "4.2", "4.3"],
            ),
            font=dict(family="sans-serif", color="#3F3D56"),
            height=620,
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as _e:
        st.warning(f"⚠️ Map render failed: {type(_e).__name__}")
        return

    st.caption(
        "**Executive insight.** Top 10 states by order volume. BA (3.86) "
        "and RJ (3.88) anchor the low end; PR (4.18) and SP (4.17) hold "
        "the high end. RJ's low score against its wealthier Southeast "
        "neighbours signals an operational problem rather than a "
        "regional characteristic."
    )


# ── Chart 4 — On-Time vs Late by State ─────────────────────────────────────
def chart_ontime_by_state(conn, year, state_json):
    df = safe_query(conn, """
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS delivered,
               SUM(CASE WHEN o.order_delivered_customer_date
                            > o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS late,
               SUM(CASE WHEN o.order_delivered_customer_date
                            <= o.order_estimated_delivery_date
                        THEN 1 ELSE 0 END) AS on_time
        FROM orders o JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_status='delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
        GROUP BY state ORDER BY delivered DESC LIMIT 5
    """, params=year_multi(year, state_json), label="On-time by state")

    if df.empty:
        no_data_notice()
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(df))
    width = 0.4
    ax.bar([i - width/2 for i in x], df["on_time"], width,
           label="On Time", color=PURPLE)
    ax.bar([i + width/2 for i in x], df["late"], width,
           label="Late", color=RED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["state"])
    ax.set_title("On-Time vs Late Deliveries by State (Top 5 by Volume)")
    ax.set_xlabel("Customer state")
    ax.set_ylabel("Delivered orders")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(frameon=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Executive insight.** RJ's late-delivery rate is 13.5% — 2.3× "
        "the 5.9% national benchmark. SP has the highest volume (40,501) "
        "at a healthy 5.9%. The problem is concentrated in RJ, not "
        "spread nationally."
    )


# ── Chart 5 — Punctuality vs Satisfaction (headline) ───────────────────────
def chart_punctuality_vs_satisfaction(conn, year, state_json):
    df = safe_query(conn, """
        SELECT CASE WHEN o.order_delivered_customer_date
                         <= o.order_estimated_delivery_date
                    THEN 'On Time' ELSE 'Late' END AS delivery_status,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score,
               ROUND(AVG(julianday(o.order_delivered_customer_date)
                       - julianday(o.order_purchase_timestamp)),2)
                   AS avg_delivery_days
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE o.order_status='delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
        GROUP BY 1 ORDER BY avg_score DESC
    """, params=year_multi(year, state_json), label="Punctuality vs satisfaction")

    if df.empty:
        no_data_notice()
        return

    color_map = {"On Time": PURPLE, "Late": RED}
    colors = [color_map.get(s, GREY) for s in df["delivery_status"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(df["delivery_status"], df["avg_score"], color=colors)
    ax.set_title("Delivery Punctuality and Customer Satisfaction")
    ax.set_xlabel("")
    ax.set_ylabel("Average review score (1–5)")
    ax.set_ylim(2, 4.8)
    for bar, score, n, days in zip(bars, df["avg_score"],
                                    df["orders"], df["avg_delivery_days"]):
        ax.text(bar.get_x() + bar.get_width()/2, score + 0.06,
                f"{score:.2f}\n(n={n:,}; {days}d average)",
                ha="center", va="bottom", fontsize=11,
                color="#4C1D95", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Executive insight.** Late deliveries cost **1.73 stars** — "
        "4.29 for on-time versus 2.57 for late. This is the single "
        "strongest satisfaction driver observed in the dataset."
    )


# ── Chart 6 — Satisfaction by Delivery-Time Bucket ─────────────────────────
def chart_satisfaction_by_delivery_bucket(conn, year, state_json):
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
               ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE o.order_status='delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
        GROUP BY 1
        ORDER BY CASE delivery_bucket
                   WHEN '0-5 days'   THEN 1
                   WHEN '6-10 days'  THEN 2
                   WHEN '11-20 days' THEN 3
                   WHEN '21-30 days' THEN 4
                   ELSE 5 END
    """, params=year_multi(year, state_json), label="Delivery buckets")

    if df.empty:
        no_data_notice()
        return

    gradient = ["#6D3FB5", "#8B5CF6", "#A78BFA", "#F0A0A0", "#D64545"]
    colors = gradient[:len(df)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(df["delivery_bucket"], df["avg_score"], color=colors)
    ax.set_title("Customer Satisfaction by Delivery-Time Category")
    ax.set_xlabel("Days from purchase to delivery")
    ax.set_ylabel("Average review score (1–5)")
    ax.set_ylim(1, 5)
    ax.axhline(4.09, color=GREY, linestyle="--", linewidth=1,
               label="National reference average (4.09)")
    ax.legend(loc="lower left", frameon=False)
    for bar, score, n in zip(bars, df["avg_score"], df["orders"]):
        ax.text(bar.get_x() + bar.get_width()/2, score + 0.06,
                f"{score:.2f}\n(n={n:,})",
                ha="center", va="bottom", fontsize=10,
                color="#4C1D95", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Executive insight.** The decline is progressive, not a cliff: "
        "4.45 → 4.36 → 4.22 → 3.69 → 2.26. Every category beyond 20 days "
        "falls below the 4.09 reference average — the evidence base for "
        "introducing a proactive 20-day intervention threshold."
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HELPER CARD RENDERER
# ════════════════════════════════════════════════════════════════════════════
def info_card(title, body, variant="info", tags=None):
    tag_html = ""
    if tags:
        tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
        tag_html = f"<div style='margin-bottom:10px;'>{tag_html}</div>"

    st.markdown(
        f"""
        <div class="info-card {variant}">
          <h4>{title}</h4>
          {tag_html}
          {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — APP BODY
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #6D3FB5 0%, #8B5CF6 50%, #D64545 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 40px;
        font-weight: 800;
        letter-spacing: -0.02em;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    ">
      Customer Satisfaction — Executive Dashboard
    </h1>
    <p style='color: #6B7280; font-size: 14px;
              margin-top: 4px; text-align: center;'>
      PORA Academy Cohort 7 · Data Analysis and AI Automation · Group 1 Capstone
    </p>
    """,
    unsafe_allow_html=True,
)


# ── Animated screen-projection headline ───────────────────────────────────
st.markdown(
    """
    <div class="projection-frame">
      <div class="projection-content">
        <p class="projection-title">Executive Headline · Screen Projection</p>
        <div class="projection-underline"></div>

        <p class="projection-line l1">
          <span class="hl">On-time deliveries average 4.29 stars</span>,
          compared with <span class="hl">2.57 stars for late deliveries</span>
          — a <span class="projection-number">1.73-star gap</span>.
        </p>

        <p class="projection-line l2">
          Late deliveries affect
          <span class="hl">8.1% of delivered orders</span>, averaging
          <span class="projection-number small">9.55 days</span> beyond
          the estimated date.
        </p>

        <p class="projection-line l3">
          <span class="hl">Rio de Janeiro</span> has a higher late-delivery
          rate (<span class="projection-number">13.5%</span>) than the
          national benchmark
          (<span class="projection-number small purple">5.9%</span>).
        </p>

        <div class="projection-priority">
          <span class="priority-label">PRIORITY ·</span>
          <span class="priority-text">
            Improve delivery reliability, realistic delivery estimates,
            proactive communication, and regional logistics.
          </span>
          <span class="projection-caveat">
            These findings show an association, not proof of causation.
          </span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── KPI computation ────────────────────────────────────────────────────────
k = kpi_values(conn, selected_year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TABS
# ════════════════════════════════════════════════════════════════════════════
(
    tab_overview,
    tab_theme,
    tab_detail,
    tab_limitations,
    tab_recommendations,
    tab_departments,
    tab_research,
    tab_report,
) = st.tabs(
    [
        "🏠 Overview",
        "😊 Customer Satisfaction",
        "🔍 Detail & Method",
        "⚠️ Limitations",
        "✅ Recommendations",
        "🏢 Departments",
        "🔬 Future Research",
        "📄 Executive Report",
    ]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Orders", safe_metric_display(k["total_orders"]))
    c2.metric("Delivered Orders", safe_metric_display(k["delivered_orders"]))

    avg_score_val = k["avg_review_score"]
    if avg_score_val is None:
        c3.metric("Average Review Score", "—")
    else:
        delta_val = None
        if filter_active:
            delta_val = f"{avg_score_val - 4.09:+.2f} vs reference"
        c3.metric(
            "Average Review Score",
            f"{avg_score_val:.2f}",
            delta=delta_val,
        )

    late_val = k["late_deliveries"]
    delivered_val = k["delivered_orders"]
    late_pct = None
    try:
        if delivered_val and delivered_val > 0:
            late_pct = f"{100 * late_val / delivered_val:.1f}% of delivered"
    except Exception:
        late_pct = None

    c4.metric(
        "Late Deliveries",
        safe_metric_display(late_val),
        delta=late_pct,
        delta_color="inverse",
    )

    avg_days = k["avg_delivery_days"]
    c5.metric(
        "Average Delivery Days",
        f"{avg_days:.2f}" if avg_days is not None else "—",
    )

    if filter_active:
        state_display = ", ".join(s for s in selected_state if s != "All") or "All"
        st.info(
            f"Reporting Year: **{selected_year}** · "
            f"Customer State(s): **{state_display}**. "
            "Set both filters to *All* to reproduce the unfiltered "
            "reference values shown in the sidebar."
        )

    st.divider()

    st.markdown(
        """
        <div class="info-card">
          <h4>Three executive insights</h4>
          <ol style="margin-bottom:0; color:#3F3D56; line-height:1.75;">
            <li><b>Late deliveries cost 1.73 stars.</b> On-time = 4.29;
                late = 2.57. Only 8.1% of orders, yet this is the
                strongest satisfaction driver observed.</li>
            <li><b>RJ is the operational outlier.</b> A 13.5%
                late-delivery rate — 2.3× the national benchmark — with
                12,687 orders of leverage.</li>
            <li><b>The decline is progressive, not a cliff.</b> Scores
                drop from 4.45 to 2.26 across five delivery-time
                categories; every category beyond 20 days falls below
                the 4.09 reference average.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    chart_review_distribution(conn)

    st.divider()
    chart_punctuality_vs_satisfaction(conn, selected_year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB — CUSTOMER SATISFACTION
# ════════════════════════════════════════════════════════════════════════════
with tab_theme:
    chart_monthly_score(conn, selected_year, state_json)
    st.divider()
    chart_brazil_map(conn, selected_year, state_json)
    st.divider()
    chart_ontime_by_state(conn, selected_year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB — DETAIL & METHOD
# ════════════════════════════════════════════════════════════════════════════
with tab_detail:
    chart_satisfaction_by_delivery_bucket(conn, selected_year, state_json)

    st.divider()
    st.markdown("## Written data-limitation note")

    st.markdown(
        """
        ### The limitation that matters most

        **2016 and 2018 are both partial years, and November 2016 is
        empty.** 2016 covers only September, October, and December —
        November has zero orders, which is why the monthly chart shows
        an honest gap rather than an interpolated line. 2018 covers
        1 January – 17 October. Any year-over-year comparison involving
        2016 or 2018 is unreliable unless explicitly qualified: 2016 has
        329 orders across three active months; 2017 has 45,101 across
        twelve; 2018 has 54,011 across ten and a half.

        ### Cleaning decisions — what was changed, and what was not

        **The review fan-out was not dropped.** `reviews` has 99,224 rows
        across 98,673 distinct orders (543 orders carry two reviews, 4
        carry three). Every order count on this dashboard uses
        `COUNT(DISTINCT o.order_id)` — the reviews table remains
        untouched.

        **Delivered-only filtering was not applied to Total Orders.** The
        reference Total Orders = 99,441 counts every order in the table,
        including cancelled and in-transit. Delivered-only metrics are
        reported separately (96,478) and labelled as such.

        **Product-category joins were not applied.** Group 1's scope is
        satisfaction by *delivery* and *state*, not by product.

        **Geolocation was not loaded.** The 1.0M-row geolocation CSV is
        excluded; the Brazil choropleth uses a small public GeoJSON
        (~600 KB) fetched once and cached.

        ### GMV definition

        This dashboard does not compute GMV. If a revenue view were added
        later, the canonical definition would be
        `SUM(price + freight_value)` across **all orders**. Total GMV
        under that definition is **R$ 15,843,553.24**. Adding
        `WHERE order_status = 'delivered'` yields
        **R$ 15,419,773.75** — a legitimate *delivered GMV* metric, but a
        different question. The −2.7% gap is the value of orders that
        never reached the customer's door.
        """
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB — LIMITATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_limitations:
    st.markdown(
        "## Analytical Limitations, Data Quality Risks "
        "and Interpretation Considerations"
    )
    st.markdown(
        "The dashboard provides useful insights into customer satisfaction "
        "and delivery performance; however, its findings must be "
        "interpreted within the constraints of the available data and "
        "analytical methods."
    )

    info_card(
        "1. Correlation does not establish causation",
        """
        <p>The observed difference in customer ratings between on-time
        and late deliveries does <b>not independently prove that
        delivery delays cause lower satisfaction</b>. Other factors,
        including product quality, seller performance, delivery
        distance, and customer expectations, may influence the results.</p>
        <p>Controlled pilots or multivariable analysis would be required
        to assess these relationships more rigorously.</p>
        """,
        variant="warn",
        tags=["Causality", "Confounders"],
    )

    info_card(
        "2. Incomplete reporting periods",
        """
        <p>The dataset contains only <b>September, October, and December
        2016</b> transactions, while 2018 ends on <b>17 October</b>.
        Consequently, <b>2017 is the only complete calendar year</b>.</p>
        <p>Comparisons across years should account for differences in
        coverage and should use equivalent reporting periods wherever
        possible.</p>
        """,
        variant="warn",
        tags=["Time coverage", "YoY risk"],
    )

    info_card(
        "3. Duplicate-related risks and customer identification",
        """
        <p>The review dataset contains <b>99,224 records associated with
        98,673 distinct orders</b>. Multiple reviews linked to individual
        orders may distort results when joins and aggregations are not
        handled carefully.</p>
        <p>The customer table also contains repeated customer records;
        analyses of retention and lifetime value should use a validated,
        consistent customer identifier.</p>
        """,
        variant="warn",
        tags=["Data quality", "Identity resolution"],
    )

    info_card(
        "4. Potential review-selection bias",
        """
        <p>Customer reviews are voluntary and may not represent the
        experiences of all buyers. The relationship between order records
        and review records should be reconciled before publishing
        definitive coverage statistics or drawing conclusions about the
        entire customer population.</p>
        """,
        variant="info",
        tags=["Selection bias", "Coverage"],
    )

    info_card(
        "5. Limited delivery-distance information",
        """
        <p>Because the geolocation dataset is excluded, the current
        analysis <b>cannot adequately distinguish the effects of
        long-distance transportation from carrier inefficiency or other
        operational constraints</b>.</p>
        """,
        variant="info",
        tags=["Scope", "Distance"],
    )

    info_card(
        "6. Scope and generalisability",
        """
        <p>The findings relate to Olist marketplace activity in Brazil
        between 2016 and 2018. They should not automatically be
        generalised to other marketplaces, countries, or periods.</p>
        <p>The geographic visualisation also relies on an external
        GeoJSON resource, creating a dependency on external data
        availability.</p>
        """,
        variant="info",
        tags=["External validity"],
    )

    info_card(
        "7. Metric consistency and dashboard validation",
        """
        <p>Before the dashboard is used for executive reporting, the
        analytical team should reconcile calculation denominators, verify
        state-level late-delivery rates, test all filter combinations,
        and standardise rounding rules. Differences in reported state
        averages should be resolved using a single validated calculation
        method.</p>
        """,
        variant="warn",
        tags=["Metric consistency", "Validation"],
    )

    st.divider()
    st.markdown("### Management implication")
    st.markdown(
        """
        These limitations do not invalidate the observed patterns, but
        they define the level of confidence appropriate for
        decision-making. Operational interventions should be supported
        by **validated metrics**, **documented assumptions**, and
        **ongoing performance monitoring**.
        """
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_recommendations:
    st.markdown(
        "## Strategic Recommendations and Operational Improvement Priorities"
    )
    st.markdown(
        "The following actions translate the dashboard findings into a "
        "practical improvement programme designed to strengthen delivery "
        "reliability, improve customer experience, and establish "
        "measurable accountability. Priority is graded 🔴 High · "
        "🟡 Medium · 🟢 Strategic."
    )

    info_card(
        "🔴 Priority 1 — Conduct a targeted logistics review in Rio de Janeiro",
        """
        <p><b>Evidence.</b> RJ reports a 13.5% late-delivery rate against
        the 5.9% national benchmark, across 12,687 orders. RJ's average
        score (3.88) sits 0.29 stars below PR (4.18) despite being a
        wealthier Southeast state.</p>
        <p><b>Action.</b> Investigate carrier performance, warehouse
        allocation, last-mile delivery routes, dispatch hand-offs, and
        destination-level delays. Implement a controlled improvement
        pilot before extending successful interventions to other
        regions.</p>
        <p><b>Lead.</b> Logistics and Operations; Regional Management.</p>
        <p><b>Success measures.</b> Late-delivery rate; on-time delivery
        rate; median and 90th-percentile transit times; average customer
        review score.</p>
        """,
        variant="warn",
        tags=["High priority", "Operations", "RJ"],
    )

    info_card(
        "🔴 Priority 2 — Introduce proactive intervention for ageing orders",
        """
        <p><b>Evidence.</b> Delivery-time categories degrade
        progressively: 4.45 → 4.36 → 4.22 → 3.69 → 2.26. The largest
        step (−0.53 stars) occurs between "11–20 days" and "21–30 days".</p>
        <p><b>Action.</b> Establish alerts for orders approaching or
        exceeding 20 days in transit. Escalate at-risk shipments,
        communicate revised delivery estimates promptly, and consider
        expedited delivery or approved service-recovery measures where
        appropriate.</p>
        <p><b>Lead.</b> Logistics and Operations; Customer Experience.</p>
        <p><b>Success measures.</b> Number of orders exceeding 20 days;
        alert-to-resolution time; complaints associated with delays;
        review scores for delayed orders.</p>
        """,
        variant="warn",
        tags=["High priority", "SLA", "Threshold"],
    )

    info_card(
        "🟡 Priority 3 — Improve delivery-date estimation",
        """
        <p><b>Evidence.</b> When orders are late, they run <b>9.55 days
        past estimate</b> on average; the worst case is 189 days. This
        is systematic optimism, not random slippage.</p>
        <p><b>Action.</b> Use historical delivery performance segmented
        by carrier and destination state to improve estimated delivery
        dates. Compare promised dates with actual outcomes and
        periodically recalibrate.</p>
        <p><b>Lead.</b> Data Science and Analytics; Logistics.</p>
        <p><b>Success measures.</b> Delivery-estimation error; percentage
        of orders delivered by the promised date; forecast accuracy by
        carrier and state.</p>
        """,
        variant="info",
        tags=["Medium priority", "Forecasting", "Customer comms"],
    )

    info_card(
        "🟡 Priority 4 — Investigate customer experience in Bahia",
        """
        <p><b>Evidence.</b> BA reports a low average review score of
        approximately <b>3.86</b> — the lowest of any state with 2,000+
        orders, across roughly 3,340 orders.</p>
        <p><b>Action.</b> Examine delivery patterns, seller hand-off
        performance, carrier reliability, and customer comments to
        determine whether the issues resemble those identified in RJ or
        require a different intervention.</p>
        <p><b>Lead.</b> Regional Operations; Logistics; Customer
        Experience.</p>
        <p><b>Success measures.</b> Average review score; late-delivery
        rate; complaint volume; before-and-after pilot results.</p>
        """,
        variant="info",
        tags=["Medium priority", "BA", "Pilot"],
    )

    info_card(
        "🟡 Priority 5 — Analyse low-rating customer feedback",
        """
        <p><b>Evidence.</b> One-star reviews are approximately 11.5% of
        the total (11,424). The free-text <code>review_comment_message</code>
        column remains unused.</p>
        <p><b>Action.</b> Categorise one- and two-star reviews into
        recurring themes such as delivery delays, product quality, seller
        communication, refunds, and other service issues. Compare the
        prevalence of these themes across RJ, BA, and relevant comparison
        regions.</p>
        <p><b>Lead.</b> Customer Experience Analytics; Data Science.</p>
        <p><b>Success measures.</b> Share of low ratings by complaint
        category; recurrence of key issues; time required to resolve
        identified root causes.</p>
        """,
        variant="info",
        tags=["Medium priority", "NLP", "Qualitative"],
    )

    info_card(
        "🟢 Priority 6 — Strengthen customer-data governance",
        """
        <p><b>Evidence.</b> 99,441 <code>customer_id</code> rows collapse
        to 96,096 <code>customer_unique_id</code> values — a 3.4%
        duplication rate.</p>
        <p><b>Action.</b> Standardise customer identification, document
        the appropriate person-level identifier, and establish a
        canonical customer data structure. Validate joins and
        duplicate-handling rules before conducting customer retention,
        repeat-purchase, or lifetime-value analysis.</p>
        <p><b>Lead.</b> Data Engineering; Data Governance.</p>
        <p><b>Success measures.</b> Duplicate-record rate; unmatched
        records; reconciliation results; consistency of repeat-customer
        metrics.</p>
        """,
        variant="ok",
        tags=["Strategic", "Data model"],
    )

    info_card(
        "🟢 Priority 7 — Prioritise reliable delivery promises over unsupported speed claims",
        """
        <p><b>Evidence.</b> On-time orders average 4.29 stars; late
        orders average 2.57. The satisfaction damage is concentrated
        entirely in the <i>miss</i>, not the speed.</p>
        <p><b>Action.</b> Align customer-facing delivery estimates with
        demonstrated operational capability. Introduce consistent delay
        notifications and clear escalation procedures so that customers
        receive accurate information throughout fulfilment.</p>
        <p><b>Lead.</b> Marketing and Customer Communications;
        Logistics.</p>
        <p><b>Success measures.</b> On-time delivery against the promised
        date; delivery-related complaint rate; customer review scores;
        adherence to communication standards.</p>
        """,
        variant="ok",
        tags=["Strategic", "Brand", "Positioning"],
    )

    st.divider()
    st.markdown("### Implementation principle")
    st.markdown(
        """
        Establish baseline performance, assign accountable owners, run
        controlled pilots, measure outcomes against agreed targets, and
        scale only interventions supported by credible evidence.
        """
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB — DEPARTMENTS
# ════════════════════════════════════════════════════════════════════════════
with tab_departments:
    st.markdown(
        "## Departmental Accountability, Governance and Dashboard Handover"
    )
    st.markdown(
        "Successful implementation requires clearly defined "
        "responsibilities, coordinated execution, and a consistent "
        "framework for monitoring results. The following ownership "
        "structure is proposed based on the analytical priorities "
        "identified in the dashboard. Department names and "
        "responsibilities should be aligned with the organisation's "
        "actual structure before implementation."
    )

    info_card(
        "1. Logistics and Operations — Accountable Lead",
        """
        <p>Lead the delivery-reliability improvement programme,
        investigate regional delivery failures, review carrier
        performance, and implement escalation procedures for ageing
        orders. Coordinate corrective actions and track operational
        performance against agreed service-level targets.</p>
        """,
        variant="warn",
        tags=["Accountable lead", "Operations"],
    )

    info_card(
        "2. Regional Management — Regional Execution and Oversight",
        """
        <p>Coordinate investigations in priority states, beginning with
        Rio de Janeiro, and facilitate regional pilots. Support
        comparisons between regions and ensure that corrective actions
        reflect local operating conditions.</p>
        """,
        variant="info",
        tags=["Regional execution"],
    )

    info_card(
        "3. Customer Experience (CX) — Customer Communication and Service Recovery",
        """
        <p>Develop consistent procedures for notifying customers of
        delays, communicating revised estimated delivery dates, managing
        complaints, and applying approved service-recovery measures.
        Monitor customer feedback to determine whether operational
        changes improve the customer experience.</p>
        """,
        variant="info",
        tags=["Customer outcome"],
    )

    info_card(
        "4. Data Science and Analytics — Analytical Quality and Performance Measurement",
        """
        <p>Maintain the analytical framework, validate key performance
        indicators, assess delivery-estimation accuracy, and evaluate
        the results of operational interventions. Develop controlled
        comparisons and statistical models where necessary to distinguish
        observed associations from stronger evidence of underlying
        drivers.</p>
        """,
        variant="info",
        tags=["Analytics", "Validation"],
    )

    info_card(
        "5. Data Engineering and Data Governance — Data Integrity and Metric Standards",
        """
        <p>Maintain reliable data structures, document customer
        identifiers, validate joins, and establish standard definitions
        for orders, reviews, delivery delays, and customer-level
        measures. Ensure that dashboards use consistent calculation
        rules and reconciled source data.</p>
        """,
        variant="ok",
        tags=["Data quality", "Governance"],
    )

    info_card(
        "6. Marketing and Customer Communications — Customer-Facing Delivery Commitments",
        """
        <p>Ensure that delivery promises and customer-facing messages
        reflect realistic operational capabilities. Coordinate with
        Logistics and Customer Experience to maintain accurate
        expectations and consistent communication throughout the
        fulfilment process.</p>
        """,
        variant="ok",
        tags=["Brand", "Positioning"],
    )

    st.divider()
    st.markdown("### Dashboard handover requirements")
    st.markdown(
        """
        Before transferring the dashboard into routine business use, the
        project team should:

        - **Validate the data:** Reconcile key totals and verify
          late-delivery calculations against the source CSV files.
        - **Standardise definitions:** Document KPI formulas,
          denominators, rounding conventions, and inclusion criteria.
        - **Test functionality:** Confirm that year and state filters
          produce consistent results across the relevant KPI cards and
          charts.
        - **Document limitations:** Record incomplete reporting periods,
          review-coverage concerns, customer-identity risks, and
          external map-data dependencies.
        - **Establish ownership:** Assign a dashboard administrator and
          a business owner responsible for interpreting results and
          coordinating action.
        - **Define reporting cadence:** Agree on review frequency,
          performance thresholds, escalation procedures, and the process
          for recording corrective actions.
        - **Maintain version control:** Document code changes,
          data-source updates, and releases so that reported figures
          remain traceable.
        """
    )
    st.markdown(
        """
        **Expected outcome:** A maintainable analytical product with
        clear accountability, standardised metrics, and an agreed process
        for converting dashboard findings into operational decisions.
        """
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB — FUTURE RESEARCH
# ════════════════════════════════════════════════════════════════════════════
with tab_research:
    st.markdown("## Future Analytical Priorities and Research Opportunities")
    st.markdown(
        "The current dashboard establishes a useful baseline for "
        "understanding the relationship between delivery performance and "
        "customer satisfaction. The next phase should deepen the analysis, "
        "test alternative explanations, and determine which operational "
        "interventions deliver measurable improvements."
    )

    info_card(
        "1. Quantify the relationship between delivery delays and satisfaction",
        """
        <p>Develop multivariable regression models that account for
        relevant factors such as delivery duration, days late,
        destination state, seller characteristics, and product category
        where the required data are available. This would help determine
        whether the observed association remains after adjusting for
        potential confounding factors.</p>
        """,
        variant="info",
        tags=["Econometrics", "Causal inference", "OLS"],
    )

    info_card(
        "2. Examine delivery performance at carrier and seller level",
        """
        <p>Extend the analysis to investigate whether delays are
        concentrated among particular carriers, sellers, dispatch
        processes, or destinations. This would help operations teams
        identify specific bottlenecks rather than relying solely on
        regional averages.</p>
        """,
        variant="info",
        tags=["Attribution", "Root cause", "Carrier"],
    )

    info_card(
        "3. Investigate the November 2017 satisfaction decline",
        """
        <p>Examine the monthly score reduction alongside order volumes,
        seasonal activity, promotions, and available logistics
        indicators. The objective should be to identify plausible
        explanations and test them against supporting evidence, rather
        than assume a cause from the trend alone.</p>
        """,
        variant="info",
        tags=["Time series", "STL", "Seasonality"],
    )

    info_card(
        "4. Analyse the content of low-rated reviews",
        """
        <p>Apply text classification, topic modelling, or structured
        manual coding to one- and two-star reviews. Compare recurring
        themes across RJ, BA, and other relevant regions to distinguish
        delivery complaints from concerns about product quality, seller
        communication, refunds, and other aspects of service.</p>
        """,
        variant="info",
        tags=["NLP", "Topic modeling", "Portuguese"],
    )

    info_card(
        "5. Incorporate delivery distance and geographic variables",
        """
        <p>Where reliable location data are available, estimate route
        distances and examine the relationship between distance, transit
        time, late delivery, and review scores. This could help
        distinguish distance-related challenges from carrier or
        fulfilment inefficiencies.</p>
        """,
        variant="info",
        tags=["Geospatial", "Distance"],
    )

    info_card(
        "6. Evaluate the proposed 20-day intervention threshold",
        """
        <p>Test whether early alerts and escalation procedures for ageing
        orders improve delivery outcomes and satisfaction. A controlled
        pilot should compare performance before and after implementation,
        with a suitable comparison group where feasible.</p>
        """,
        variant="info",
        tags=["Intervention evaluation", "Pilot"],
    )

    info_card(
        "7. Assess payment-related differences in customer expectations",
        """
        <p>Explore whether payment type or instalment arrangements are
        associated with different satisfaction patterns during delivery
        delays. Any observed differences should be interpreted cautiously
        and adjusted for other relevant characteristics.</p>
        """,
        variant="info",
        tags=["Moderation", "Payments", "Behaviour"],
    )

    info_card(
        "8. Validate findings beyond the Olist marketplace",
        """
        <p>Replicate the analysis using a comparable external e-commerce
        dataset. This would help establish whether the observed 1.73-star
        difference is specific to Olist or whether a similar pattern
        appears in other marketplace settings.</p>
        """,
        variant="info",
        tags=["External validity", "Replication"],
    )

    info_card(
        "9. Strengthen customer-level analysis",
        """
        <p>Use a validated customer identifier to investigate repeat
        purchases, retention, and customer lifetime value. This would
        extend the dashboard from transaction-level satisfaction
        monitoring to longer-term customer relationship analysis.</p>
        """,
        variant="info",
        tags=["Retention", "Cohorts", "LTV"],
    )

    st.divider()
    st.markdown("### Expected research contribution")
    st.markdown(
        """
        These extensions would move the project from **descriptive
        reporting** towards stronger **explanatory analysis**,
        **intervention evaluation**, and more **generalisable business
        recommendations**.
        """
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB — EXECUTIVE REPORT (5-page document, embedded)
# ════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("# Customer Satisfaction Analysis: Executive Management Report")
    st.markdown(
        "**PORA Academy Cohort 7 | Data Analysis and AI Automation | "
        "Group 1 Capstone**"
    )

    state_display_rpt = ", ".join(s for s in selected_state if s != "All") or "All"
    st.markdown(
        f"""
        <div class="callout">
        <b>Active filter:</b> Reporting Year = <code>{selected_year}</code> · Customer States = <code>{state_display_rpt}</code>.
        KPI cards above reflect the current sidebar selection. The report
        below presents the unfiltered reference results — use the
        Reference table in the sidebar to cross-check.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 1 ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>1. Executive Overview</h2>
        <p>This report presents the principal findings, analytical
        limitations, and recommended business responses arising from the
        Olist Customer Satisfaction Dashboard. The Streamlit application
        uses Python, pandas, SQLite, Matplotlib, and Plotly to examine
        order volumes, customer ratings, monthly satisfaction trends,
        state-level differences, delivery punctuality, and delivery-time
        categories.</p>

        <p>The central finding is that <b>delivery punctuality is
        strongly associated with customer satisfaction</b>. The dashboard
        reports an average rating of <b>4.29 stars for on-time
        deliveries</b> and <b>2.57 stars for late deliveries</b>, a
        difference of <b>1.73 stars</b>. Across the full dataset, the
        reference indicators include 99,441 orders, 96,478 delivered
        orders, an average review score of 4.09, 7,826 late deliveries,
        and an average delivery time of 12.56 days.</p>

        <p>The dashboard also identifies regional differences in
        delivery performance. Rio de Janeiro (RJ) has a reported
        late-delivery rate of <b>13.5%</b>, compared with
        <b>5.9% nationally</b>, while Bahia (BA) is highlighted for a
        comparatively low average review score of approximately
        <b>3.86</b>. These patterns support targeted regional
        investigation rather than a uniform intervention across all
        states.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 2 ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>2. Key Findings and Business Implications</h2>

        <h3>Delivery punctuality and customer experience</h3>
        <p>The 1.73-star difference between on-time and late deliveries
        identifies delivery reliability as a priority for further
        operational investigation. The evidence supports examining
        fulfilment processes, carrier performance, delivery promises,
        and customer communication.</p>

        <h3>Extended delivery times and satisfaction</h3>
        <p>The dashboard reports average ratings of <b>4.45, 4.36, 4.22,
        3.69, and 2.26</b> across successive delivery-time categories.
        Ratings fall below the 4.09 overall reference average for
        delivery periods exceeding 20 days. This pattern supports
        testing early-warning and escalation measures before orders
        enter prolonged delays.</p>

        <h3>Regional performance variation</h3>
        <p>RJ warrants priority investigation because of its reported
        late-delivery rate. BA should also be examined to understand its
        lower reported average rating. Regional comparisons should
        account for differences in delivery distance, order mix, seller
        composition, and infrastructure before corrective actions are
        selected.</p>

        <h3>Review distribution and monthly trends</h3>
        <p>The dashboard reports that <b>57.8%</b> of reviews are
        five-star and <b>11.5%</b> are one-star. It also identifies a
        decline in average review scores during <b>November 2017</b>.
        These findings warrant further investigation into the
        characteristics of low-rated experiences and the circumstances
        surrounding the monthly decline. The absence of November 2016
        orders should remain a genuine data gap rather than be replaced
        with an estimated score.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 3 ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>3. Recommended Management Response</h2>
        <p>The recommended approach is an <b>evidence-led
        delivery-reliability programme</b>, beginning with a diagnostic
        review of RJ and an escalation process for orders approaching
        20 days in transit.</p>

        <p>Logistics and Operations should lead the initiative,
        supported by Regional Management, Customer Experience, Data
        Science and Analytics, Data Engineering, and Marketing.
        Priorities should include investigating carrier and last-mile
        performance, improving estimated delivery dates, notifying
        customers proactively about delays, analysing low-rated review
        comments, and strengthening the consistency of analytical
        metrics.</p>

        <p>Bahia may provide an additional comparison or pilot setting
        where appropriate. Interventions should be introduced through
        controlled pilots, evaluated against established baselines, and
        expanded only when the results support doing so.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 4 ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>4. Analytical Limitations and Governance</h2>
        <p>The findings are <b>descriptive and observational</b>; they
        identify relationships but do not independently establish
        causation. The dataset also includes incomplete calendar years,
        voluntary customer reviews, repeated review records for some
        orders, and limited delivery-distance information.</p>

        <p>Before the dashboard is used for formal executive reporting,
        the team should validate the source data, reconcile calculation
        denominators, confirm state-level late-delivery rates,
        standardise rounding rules, and test filter behaviour across
        all relevant visuals. The external geographic-data dependency
        should also be documented and monitored.</p>

        <p>The reported numerical findings should be verified against
        the original CSV files before publication or use in formal
        performance commitments.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 5 ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>5. Conclusion</h2>
        <p>The supplied dashboard identifies <b>delivery reliability as
        the clearest actionable correlate of customer satisfaction</b>
        within the analysed Olist dataset. The reported gap between
        on-time and late-delivery ratings, the elevated late-delivery
        rate in RJ, and the decline in satisfaction for longer delivery
        periods provide a practical basis for prioritising operational
        investigation.</p>

        <p>The recommended response combines <b>regional logistics
        audits</b>, <b>early intervention for ageing orders</b>,
        <b>more realistic delivery estimates</b>, <b>proactive customer
        communication</b>, <b>structured review analysis</b>, and
        <b>stronger data governance</b>. The business objective is not
        simply to deliver faster, but to <b>deliver more reliably,
        communicate more accurately, and establish a measurable process
        for continuous improvement</b>.</p>

        <p style="font-size: 13px; color: #6B7280; margin-top: 24px;">
        <i>Source and interpretation note:</i> This report is based on
        the supplied Streamlit dashboard code. Numerical results are
        reproduced from the code's documented findings and should be
        validated against the underlying data before external
        publication or use as definitive performance measures.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "**Reference (filters = All):** Total Orders 99,441 · Delivered 96,478 · "
    "Avg Review 4.09 · Late 7,826 · Avg Delivery Days 12.56. "
    "Both Year and State filters apply to every KPI and every chart on "
    "the **Overview** and **Customer Satisfaction** tabs. Chart 1 (Review "
    "Score Distribution) is intentionally unfiltered — a whole-dataset "
    "structural fact. **Tabs:** Overview · Customer Satisfaction · "
    "Detail & Method · Limitations · Recommendations · Departments · "
    "Future Research · Executive Report."
)