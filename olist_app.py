# app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis & AI Automation

A complete Streamlit dashboard analysing Olist customer satisfaction,
with tabs for Overview, Customer Satisfaction, Detail & Method,
Limitations, Recommendations, and Future Research.
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
# SECTION 0 — Page config (MUST be the first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Satisfaction | Olist Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
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
# SECTION 1 — DATA LAYER
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

conn = get_conn()


# ── Filter contract ────────────────────────────────────────────────────────
def year_params(year):              return (year, year)
def multi_params(multi_json):       return (multi_json, multi_json)
def year_multi(year, multi_json):   return (year, year, multi_json, multi_json)


# ── The 27 Brazilian state codes ───────────────────────────────────────────
ALL_STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
    "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
STATES = ["All"] + ALL_STATES

st.sidebar.title("📊 Filters")
st.sidebar.markdown(
    "Slice the dashboard below. Every value resets to the unfiltered "
    "reference when set to *All*."
)

selected_year = st.sidebar.selectbox("Year", ["All", "2016", "2017", "2018"])

selected_state = st.sidebar.multiselect(
    "State",
    options=STATES,
    default=["All"],
    help='Pick one or more states. "All" (default) applies no state filter.',
)


state_json = json.dumps(
    [] if "All" in selected_state or not selected_state
    else list(selected_state)
)


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


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — KPI ROW (Q1)
# ════════════════════════════════════════════════════════════════════════════
def kpi_values(conn, year):
    """Q1 — headline KPI row. Year-filterable."""
    return pd.read_sql("""
        SELECT COUNT(DISTINCT o.order_id) AS total_orders,
               COUNT(DISTINCT CASE WHEN o.order_status='delivered'
                                    THEN o.order_id END) AS delivered_orders,
               (SELECT ROUND(AVG(CAST(review_score AS REAL)),2)
                  FROM reviews) AS avg_review_score,
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
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
    """, conn, params=year_params(year)).iloc[0]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHARTS
# ════════════════════════════════════════════════════════════════════════════

# ── Chart 1 — Q2, not filterable ────────────────────────────────────────────
def chart_review_distribution(conn):
    """Q2 — review score distribution. Whole-dataset fact, not filterable."""
    df = pd.read_sql("""
        SELECT CAST(review_score AS INTEGER) AS score,
               COUNT(*)                     AS reviews,
               ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM reviews),1) AS pct
        FROM reviews GROUP BY 1 ORDER BY 1
    """, conn)

    purple_shades = []
    for s in df["score"]:
        if s == 5:
            purple_shades.append("#6D3FB5")
        elif s == 1:
            purple_shades.append("#D64545")
        else:
            purple_shades.append("#C4B5FD")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(df["score"].astype(str), df["reviews"], color=purple_shades)
    ax.set_title("Review Score Distribution")
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
        "**Finding.** The distribution is bimodal. 57.8% of reviews are 5★ "
        "and 11.5% are 1★. Deliberately unfiltered — a whole-dataset fact."
    )


# ── Chart 2 — Q3, filterable ────────────────────────────────────────────────
def chart_monthly_score(conn, year, state_json):
    """Q3 — monthly average review score. Filterable by year and state."""
    df = pd.read_sql("""
        SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
               COUNT(*) AS reviews,
               ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
        GROUP BY month ORDER BY month
    """, conn, params=year_multi(year, state_json))

    if df.empty:
        st.info("No data for the current filter selection.")
        return

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["month"], df["avg_score"], marker="o", color=PURPLE_DARK,
            linewidth=2.5, markersize=5)
    ax.set_title("Monthly Average Review Score")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg review score (1–5)")
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
        "**Finding.** Scores hold in the 4.0–4.2 band except Nov 2017. "
        "Nov 2016 is absent — zero orders that month; the gap is real."
    )


# ── Chart 3 — Q4, filterable ────────────────────────────────────────────────
def chart_score_by_state(conn, year, state_json):
    """Q4 — avg review score by state (top 10 by order volume)."""
    df = pd.read_sql("""
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
        GROUP BY state ORDER BY orders DESC LIMIT 10
    """, conn, params=year_multi(year, state_json))

    if df.empty:
        st.info("No data for the current filter selection.")
        return

    df = df.sort_values("avg_score")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [RED if s < 4.0 else PURPLE for s in df["avg_score"]]
    ax.barh(df["state"], df["avg_score"], color=colors)
    ax.set_title("Average Review Score by Top 10 States")
    ax.set_xlabel("Avg review score (1–5)")
    ax.set_ylabel("")
    ax.set_xlim(3.5, 4.4)
    ax.axvline(4.09, color=GREY, linestyle="--", linewidth=1.2,
               label="National avg 4.09")
    ax.legend(loc="lower right", frameon=False)
    for y, (score, orders) in enumerate(zip(df["avg_score"], df["orders"])):
        ax.text(score + 0.01, y, f"{score:.2f}  (n={orders:,})",
                va="center", fontsize=10, color="#4C1D95", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Finding.** BA (3.86) and RJ (3.88) sit below 4.0 despite high "
        "volume. PR (4.18) and SP (4.17) lead. Red bars flag states below "
        "the 4.0 threshold."
    )


# ── Chart 3b — Brazil choropleth (top 10 states only, with labels) ────────
@st.cache_data(show_spinner=False)
def _load_brazil_geojson():
    """Fetch and cache the Brazil-states GeoJSON."""
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
    """
    Hard-coded approximate centroids for the 27 Brazilian states.
    Used to place text labels on the choropleth.
    """
    return {
        "AC": (-9.0238,   -70.8120),
        "AL": (-9.5713,   -36.7820),
        "AM": (-3.4168,   -65.8561),
        "AP": ( 1.4100,   -51.7700),
        "BA": (-12.5797,  -41.7007),
        "CE": (-5.4984,   -39.3206),
        "DF": (-15.7998,  -47.8645),
        "ES": (-19.1834,  -40.3089),
        "GO": (-15.8270,  -49.8362),
        "MA": (-4.9609,   -45.2744),
        "MG": (-18.5122,  -44.5550),
        "MS": (-20.7722,  -54.7852),
        "MT": (-12.6819,  -56.9211),
        "PA": (-3.7900,   -52.4800),
        "PB": (-7.2400,   -36.7820),
        "PE": (-8.8137,   -36.9541),
        "PI": (-7.7183,   -42.7289),
        "PR": (-24.8934,  -51.5300),
        "RJ": (-22.9068,  -43.1729),
        "RN": (-5.7945,   -36.5298),
        "RO": (-11.5057,  -63.5806),
        "RR": ( 2.7376,   -62.0751),
        "RS": (-30.0346,  -51.2177),
        "SC": (-27.2423,  -50.2189),
        "SE": (-10.5741,  -37.3857),
        "SP": (-23.5505,  -46.6333),
        "TO": (-10.1753,  -48.2982),
    }


def chart_brazil_map(conn, year, state_json):
    """
    Q4 rendered geographically — TOP 10 STATES ONLY, each labelled
    with its state code and average review score.
    """
    # ── 1. Fetch metrics for every state (year-filter only) ─────────────
    all_states_df = pd.read_sql("""
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
    """, conn, params=year_params(year))

    if all_states_df.empty:
        st.info("No data for the current filter selection.")
        return

    # ── 2. Which states are currently selected in the sidebar? ──────────
    selected = json.loads(state_json)
    filter_active = len(selected) > 0

    # ── 3. Restrict to the top-10 set (union with any user-selected) ────
    top10 = all_states_df["state"].tolist()
    if filter_active:
        keep_states = sorted(set(top10) | set(selected))
    else:
        keep_states = top10

    # Ensure every kept state has a row (fill zeros for missing ones)
    kept_df = pd.DataFrame({"state": keep_states}).merge(
        all_states_df[["state", "orders", "avg_score"]],
        on="state", how="left",
    )
    kept_df["orders"]    = kept_df["orders"].fillna(0).astype(int)
    kept_df["avg_score"] = kept_df["avg_score"].fillna(0)

    # ── 4. Column Plotly colours by (grey sentinel for filtered-out) ────
    if filter_active:
        kept_df["display_score"] = kept_df.apply(
            lambda row: row["avg_score"] if row["state"] in selected else -1,
            axis=1,
        )
    else:
        kept_df["display_score"] = kept_df["avg_score"]

    # ── 5. Hover text ────────────────────────────────────────────────────
    def hover_label(row):
        if filter_active and row["state"] not in selected:
            return "Not in current filter"
        if row["orders"] == 0:
            return "No orders"
        return f"{row['avg_score']:.2f} avg · {row['orders']:,} orders"

    kept_df["hover"] = kept_df.apply(hover_label, axis=1)

    # ── 6. Centroids for the labels ──────────────────────────────────────
    centroids = _state_centroids()
    kept_df["lat"] = kept_df["state"].map(
        lambda s: centroids.get(s, (None, None))[0]
    )
    kept_df["lon"] = kept_df["state"].map(
        lambda s: centroids.get(s, (None, None))[1]
    )

    # Only label states that are actually visible (not filtered out)
    if filter_active:
        label_df = kept_df[kept_df["state"].isin(selected)].dropna(
            subset=["lat", "lon"]
        )
    else:
        label_df = kept_df.dropna(subset=["lat", "lon"])

    # Text to render on the map: "SP\n4.17"
    label_df = label_df.assign(
        label=label_df.apply(
            lambda r: (f"{r['state']}\n{r['avg_score']:.2f}"
                       if r["orders"] > 0 else r["state"]),
            axis=1,
        )
    )

    geojson = _load_brazil_geojson()

    # ── 7. Colour scale ──────────────────────────────────────────────────
    color_scale = [
        [0.00, "#3F3F5A"],   # filtered-out grey
        [0.05, "#3F3F5A"],
        [0.10, "#B83232"],   # low score, red
        [0.45, "#F0A0A0"],
        [0.65, "#C4B5FD"],   # high score, purple
        [1.00, "#6D3FB5"],
    ]

    # ── 8. Build the choropleth ──────────────────────────────────────────
    fig = px.choropleth(
        kept_df,
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

    # ── 9. Overlay the labels as scatter points ──────────────────────────
    if not label_df.empty:
        fig.add_scattergeo(
            lat=label_df["lat"],
            lon=label_df["lon"],
            text=label_df["label"],
            mode="text",
            textfont=dict(
                size=11,
                color="#1F2937",
                family="sans-serif",
            ),
            hoverinfo="skip",
            showlegend=False,
        )

    # ── 10. Geography + layout ───────────────────────────────────────────
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        showcountries=False,
        showcoastlines=False,
        showland=False,
        showlakes=False,
        projection_type="mercator",
    )

    fig.update_layout(
        title=dict(
            text=("Average Review Score — Top 10 States "
                  "(Brazil) with state labels"),
            x=0.5, xanchor="center",
            font=dict(size=14, color="#4C1D95", family="sans-serif"),
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="Avg score",
            tickfont=dict(color="#4C1D95"),
            title_font=dict(color="#4C1D95"),
            thickness=14, len=0.7,
            tickvals=[-1, 3.9, 4.0, 4.1, 4.2, 4.3],
            ticktext=["Filtered out", "3.9", "4.0", "4.1", "4.2", "4.3"],
        ),
        font=dict(family="sans-serif", color="#3F3D56"),
        height=620,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**Finding.** Map shows the **top 10 states by order volume** with "
        "each state's code and average review score overlaid on the "
        "polygon. BA and RJ anchor the low end; PR, SP, MG hold the high "
        "end. RJ's low score against its wealthy Southeast neighbours "
        "signals an operational problem, not a regional one."
    )


# ── Chart 4 — Q5, filterable ────────────────────────────────────────────────
def chart_ontime_by_state(conn, year, state_json):
    """Q5 — on-time vs late delivery by state (top 5)."""
    df = pd.read_sql("""
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
    """, conn, params=year_multi(year, state_json))

    if df.empty:
        st.info("No data for the current filter selection.")
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
    ax.set_title("On-Time vs Late Deliveries by State (top 5)")
    ax.set_xlabel("State")
    ax.set_ylabel("Delivered orders")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(frameon=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Finding.** RJ's late rate is 13.5% — 2.3× the 5.9% national "
        "average. SP has the highest volume (40,501) at a healthy 5.9%. "
        "The problem is concentrated in RJ."
    )


# ── Chart 5 — Q6, ADVANCED, filterable ─────────────────────────────────────
def chart_punctuality_vs_satisfaction(conn, year, state_json):
    """Q6 — delivery punctuality vs satisfaction. Advanced feature."""
    df = pd.read_sql("""
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
    """, conn, params=year_multi(year, state_json))

    if df.empty:
        st.info("No data for the current filter selection.")
        return

    color_map = {"On Time": PURPLE, "Late": RED}
    colors = [color_map[s] for s in df["delivery_status"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(df["delivery_status"], df["avg_score"], color=colors)
    ax.set_title("Delivery Punctuality vs Satisfaction")
    ax.set_xlabel("")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(2, 4.8)
    for bar, score, n, days in zip(bars, df["avg_score"],
                                    df["orders"], df["avg_delivery_days"]):
        ax.text(bar.get_x() + bar.get_width()/2, score + 0.06,
                f"{score:.2f}\n(n={n:,}; {days}d avg)",
                ha="center", va="bottom", fontsize=11,
                color="#4C1D95", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Headline finding.** Late deliveries cost **1.73 stars** — 4.29 "
        "for on-time vs 2.57 for late. The strongest satisfaction driver "
        "in the dataset."
    )


# ── Chart 6 — Q9, filterable ────────────────────────────────────────────────
def chart_satisfaction_by_delivery_bucket(conn, year, state_json):
    """Q9 — avg review score by delivery-time bucket. Filterable."""
    df = pd.read_sql("""
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
    """, conn, params=year_multi(year, state_json))

    if df.empty:
        st.info("No data for the current filter selection.")
        return

    gradient = ["#6D3FB5", "#8B5CF6", "#A78BFA", "#F0A0A0", "#D64545"]
    colors = gradient[:len(df)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(df["delivery_bucket"], df["avg_score"], color=colors)
    ax.set_title("Satisfaction by Delivery-Time Bucket")
    ax.set_xlabel("Days from purchase to delivery")
    ax.set_ylabel("Avg review score (1–5)")
    ax.set_ylim(1, 5)
    ax.axhline(4.09, color=GREY, linestyle="--", linewidth=1,
               label="National avg 4.09")
    ax.legend(loc="lower left", frameon=False)
    for bar, score, n in zip(bars, df["avg_score"], df["orders"]):
        ax.text(bar.get_x() + bar.get_width()/2, score + 0.06,
                f"{score:.2f}\n(n={n:,})",
                ha="center", va="bottom", fontsize=10,
                color="#4C1D95", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Finding.** The score decline is smooth, not a cliff: 4.45 → 4.36 "
        "→ 4.22 → 3.69 → 2.26. Every bucket past 20 days falls below the "
        "4.09 national average."
    )


# ── Chart 7 — Q12, filterable ───────────────────────────────────────────────
def chart_lowest_scoring_states(conn, year, state_json):
    """Q12 — lowest-scoring states with min 2,000 orders. Filterable."""
    df = pd.read_sql("""
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN reviews r  ON o.order_id  = r.order_id
        WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
          AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
        GROUP BY state
        HAVING orders >= 2000
        ORDER BY avg_score ASC
        LIMIT 10
    """, conn, params=year_multi(year, state_json))

    if df.empty:
        st.info("No state meets the 2,000-order minimum with these filters.")
        return

    def shade(score):
        if score < 3.95:  return "#B83232"
        if score < 4.05:  return "#D64545"
        return "#F0A0A0"
    colors = [shade(s) for s in df["avg_score"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(df["state"][::-1], df["avg_score"][::-1], color=colors[::-1])
    ax.set_title("Lowest-Scoring States")
    ax.set_xlabel("Avg review score (1–5)")
    ax.set_ylabel("")
    ax.set_xlim(3.7, 4.2)
    ax.axvline(4.09, color=GREY, linestyle="--", linewidth=1,
               label="National avg 4.09")
    ax.legend(loc="lower right", frameon=False)
    for y, (score, n) in enumerate(zip(df["avg_score"][::-1],
                                       df["orders"][::-1])):
        ax.text(score + 0.005, y, f"{score:.2f}  (n={n:,})",
                va="center", fontsize=10,
                color="#4C1D95", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "**Finding.** Even restricting to states with 2,000+ orders, BA "
        "(3.86) and RJ (3.88) sit ~0.23 stars below the national average. "
        "Fixing RJ alone — 12,687 orders — would move the national average."
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HELPER CARD RENDERER
# ════════════════════════════════════════════════════════════════════════════
def info_card(title, body, variant="info", tags=None):
    """Render a styled card for limitations / recommendations / research."""
    tag_html = ""
    if tags:
        tag_html = "".join(
            f'<span class="tag">{t}</span>' for t in tags
        )
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
# SECTION 6 — APP BODY & TABS
# ════════════════════════════════════════════════════════════════════════════

# ── Gradient title + subtitle ──────────────────────────────────────────────
st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #6D3FB5 0%, #8B5CF6 50%, #D64545 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -0.02em;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    ">
      Customer Satisfaction Dashboard
    </h1>
    <p style='color: #6B7280; font-size: 14px;
              margin-top: 4px; text-align: center;'>
      PORA Academy Cohort 7 — Group 1 Capstone Project
    </p>
    """,
    unsafe_allow_html=True,
)


# ── Headline banner ────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #F5F0FF 0%, #EDE7FA 100%);
        border-left: 5px solid #6D3FB5;
        padding: 18px 24px;
        border-radius: 10px;
        margin: 6px 0 20px 0;
        box-shadow: 0 3px 12px rgba(109, 63, 181, 0.10);
        text-align: center;
    ">
      <p style="
          font-size: 16px;
          color: #3F3D56;
          line-height: 1.65;
          margin: 0 0 8px 0;
      ">
        <b style="color:#4C1D95;">Headline:</b>
        Customer satisfaction is overwhelmingly driven by
        <b>delivery punctuality</b> — late deliveries cost
        <span style="color:#D64545; font-size:22px;"><b>1.73 stars</b></span>
        &nbsp;·&nbsp; <b>4.29</b> on-time vs <b>2.57</b> late.
      </p>
      <p style="
          font-size: 15px;
          color: #3F3D56;
          line-height: 1.6;
          margin: 0;
      ">
        Only <b>8.1%</b> of delivered orders are late — but they run
        <b>9.55 days</b> past estimate on average, and the problem is
        concentrated in <b style="color:#D64545;">RJ</b> (13.5% late, 2.3×
        the national rate).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

k = kpi_values(conn, selected_year)

# ── SIX TABS ───────────────────────────────────────────────────────────────
(
    tab_overview,
    tab_theme,
    tab_detail,
    tab_limitations,
    tab_recommendations,
    tab_research,
) = st.tabs(
    [
        "🏠 Overview",
        "😊 Customer Satisfaction",
        "🔍 Detail & Method",
        "⚠️ Limitations",
        "✅ Recommendations",
        "🔬 Future Research",
    ]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders", f"{int(k['total_orders']):,}")
    c2.metric("Delivered Orders", f"{int(k['delivered_orders']):,}")
    c3.metric(
        "Avg Review Score",
        f"{k['avg_review_score']:.2f}",
        delta=(f"{k['avg_review_score'] - 4.09:+.2f} vs All"
               if selected_year != "All" else None),
    )
    c4.metric(
        "Late Deliveries",
        f"{int(k['late_deliveries']):,}",
        delta=(f"{100*k['late_deliveries']/k['delivered_orders']:.1f}% "
               f"of delivered"
               if k['delivered_orders'] > 0 else None),
        delta_color="inverse",
    )
    c5.metric("Avg Delivery Days", f"{k['avg_delivery_days']:.2f}")

    if selected_year != "All":
        st.info(
            f"Filtered to **{selected_year}**. Toggle Year = All to "
            "reproduce the unfiltered reference table in the sidebar."
        )

    st.divider()
    chart_review_distribution(conn)

    st.divider()
    chart_punctuality_vs_satisfaction(conn, selected_year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SATISFACTION
# ════════════════════════════════════════════════════════════════════════════
with tab_theme:
    chart_monthly_score(conn, selected_year, state_json)
    st.divider()
    chart_score_by_state(conn, selected_year, state_json)
    st.divider()
    chart_brazil_map(conn, selected_year, state_json)
    st.divider()
    chart_ontime_by_state(conn, selected_year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DETAIL & METHOD
# ════════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.markdown(
        """
        <div class="info-card">
          <h4>Three insights</h4>
          <ol style="margin-bottom:0; color:#3F3D56; line-height:1.75;">
            <li><b>Late deliveries cost 1.73 stars.</b> On-time = 4.29;
                late = 2.57. Only 8.1% of orders, but the effect is the
                strongest driver in the dataset.</li>
            <li><b>RJ is the operational outlier.</b> 13.5% late rate —
                2.3× the national average — with 12,687 orders of leverage.</li>
            <li><b>The decline is smooth, not a cliff.</b> Scores drop from
                4.45 to 2.26 across five delivery-time buckets; every
                bucket past 20 days falls below the national average.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_satisfaction_by_delivery_bucket(conn, selected_year, state_json)
    st.divider()
    chart_lowest_scoring_states(conn, selected_year, state_json)

    st.divider()
    st.markdown("## Cleaning decisions & GMV definition")

    st.markdown("""
### Cleaning decisions — what we changed, and what we didn't

**We did not drop the fan-out.** `reviews` has 99,224 rows across 98,673
distinct orders (543 orders carry 2 reviews, 4 carry 3). We did not
deduplicate; instead, every order count on this dashboard uses
`COUNT(DISTINCT o.order_id)`. The reviews table is left untouched, and
the query changes shape around it.

**We did not filter "delivered-only" from Total Orders.** The brief's
reference table (Total Orders = 99,441) counts every order that exists,
including cancelled and in-transit. We reproduce that number. Any
delivered-only metric on this dashboard is a *separate, labelled* number
(delivered orders = 96,478), not a substitution.

**We did not join product categories.** Group 1's focus is satisfaction
by *delivery* and *state*, not by product. The 623-product / R$185,049.76
join-loss documented in the Phase 2c brief would apply to a category-based
chart; we have none. This is a scope decision, not an oversight.

**We did not load geolocation.** `olist_geolocation_dataset.csv` is
1,000,163 rows (~58 MB). No query on this dashboard needs it. The brief
permits this exclusion (p.4). The Brazil choropleth uses a small public
GeoJSON (~600 KB) fetched once and cached.

### GMV definition

This dashboard does not compute GMV. Group 1's scope is satisfaction by
delivery and state, and none of our charts or KPIs require it. If we were
to add a revenue view, we would use the brief's canonical definition —
`SUM(price + freight_value)` across **all orders** — because GMV means
*gross merchandise value*: everything that flowed through the platform,
including orders later cancelled or still in transit. Total GMV under that
definition is **R$ 15,843,553.24**. Adding a `WHERE order_status =
'delivered'` clause yields **R$ 15,419,773.75** — a legitimate *delivered
GMV* metric, but a different question, and we would label it as such. The
−2.7% gap between the two is the value of orders that never reached the
customer's door.
""")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIMITATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_limitations:
    st.markdown("## Project Limitations")
    st.markdown(
        "Every analytical project has boundaries. Naming them honestly is "
        "part of the deliverable — it tells future analysts exactly where "
        "to pick up."
    )

    info_card(
        "1. Partial-year coverage — 2016 and 2018 are incomplete",
        """
        <p><b>2016</b> covers only September, October and December.
        November 2016 is <b>empty</b> — a genuine, unfillable gap, not
        missing data.</p>
        <p><b>2018</b> covers 1 Jan through 17 Oct only.</p>
        <p><b>Why it matters.</b> Year-over-year comparisons involving
        2016 (329 orders across 3 active months) or 2018 (54,011 orders
        across 10.5 months) are unreliable unless explicitly qualified.
        Only 2017 (45,101 orders across 12 months) is a clean full year.</p>
        """,
        variant="warn",
        tags=["Time coverage", "Seasonality", "YoY risk"],
    )

    info_card(
        "2. Reviews fan-out — 551 duplicate review rows",
        """
        <p>The <code>order_reviews</code> table has 99,224 rows across
        only 98,673 distinct orders. That is a <b>551-row excess</b>
        (543 orders carry 2 reviews, 4 carry 3).</p>
        <p><b>Why it matters.</b> A naive <code>JOIN</code> would double-
        count those orders, inflating order totals by up to 0.6%. We
        neutralised this on every order-count query with
        <code>COUNT(DISTINCT o.order_id)</code> — but any downstream
        analyst who copies a raw join will silently reproduce the error.</p>
        """,
        variant="warn",
        tags=["Data quality", "Join integrity"],
    )

    info_card(
        "3. Two customer identifiers, only one is stable",
        """
        <p>The <code>customers</code> table has <b>99,441 rows</b> but
        only <b>96,096 distinct <code>customer_unique_id</code></b>. A
        single real person can appear across multiple
        <code>customer_id</code> values.</p>
        <p><b>Why it matters.</b> Our state-based analysis groups by
        <code>customer_state</code>, which is stable per row — fine.
        But any <i>repeat-purchase</i> or <i>lifetime-value</i> analysis
        must use <code>customer_unique_id</code>, or it will over-count
        loyal customers.</p>
        """,
        variant="warn",
        tags=["Identity resolution", "LTV"],
    )

    info_card(
        "4. Correlation, not causation",
        """
        <p>The headline "late deliveries cost 1.73 stars" is a strong
        <b>association</b>, not a proven causal effect. Confounders we
        have not controlled for:</p>
        <ul>
          <li>Product category — heavy or fragile goods may both ship late
              and attract low reviews.</li>
          <li>Distance — far states receive later parcels and may have
              different expectations.</li>
          <li>Seller quality — a seller with both fulfilment and product
              problems would confound the delivery-score link.</li>
        </ul>
        <p>See <b>Future Research</b> for the regression design that
        would isolate the causal effect.</p>
        """,
        variant="info",
        tags=["Causality", "Confounders"],
    )

    info_card(
        "5. No geolocation or distance variable",
        """
        <p>We deliberately excluded
        <code>olist_geolocation_dataset.csv</code> (1.0M rows, ~58 MB).
        Without lat/long or a computed delivery distance, we cannot
        separate "this delivery was late because the route is long" from
        "this delivery was late because the carrier failed."</p>
        """,
        variant="info",
        tags=["Scope", "Distance"],
    )

    info_card(
        "6. Reviews exist only for a fraction of orders",
        """
        <p>99,224 reviews vs 99,441 orders means <b>~217 orders have no
        review at all</b>, and many cancelled / unavailable orders are
        systematically under-reviewed. Scores therefore reflect the
        experience of <i>reviewers</i>, not all customers — a mild
        selection bias, typical of voluntary-feedback systems.</p>
        """,
        variant="info",
        tags=["Selection bias", "Coverage"],
    )

    info_card(
        "7. One marketplace, one country, one time period",
        """
        <p>Findings are specific to Olist (Brazil, 2016–2018). The 1.73-
        star gap between on-time and late deliveries is likely a general
        e-commerce pattern, but the <b>magnitude</b> and the <b>RJ
        concentration</b> may not transfer to other marketplaces, other
        logistics networks, or other years.</p>
        """,
        variant="info",
        tags=["External validity"],
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_recommendations:
    st.markdown("## Recommendations")
    st.markdown(
        "Each recommendation is tagged with the evidence that supports it "
        "and the group best positioned to act. Priority is graded "
        "🔴 High · 🟡 Medium · 🟢 Strategic."
    )

    info_card(
        "🔴 R1 — Fix RJ logistics first (highest ROI)",
        """
        <p><b>Evidence.</b> RJ carries 12,687 orders at a 13.5% late rate
        — 2.3× the 5.9% national average. RJ's average score (3.87) sits
        0.22 stars below SP (4.17) despite being a wealthy Southeast
        state.</p>
        <p><b>Action.</b> Audit the RJ carrier mix, warehouse proximity,
        and last-mile routes. A pilot that lifts RJ's late rate to the
        national 5.9% would remove roughly <b>950 late deliveries per
        year</b> and — applying the 1.73-star effect — lift RJ's average
        score by an estimated <b>+0.06 stars</b>.</p>
        <p><b>Owner.</b> Logistics / Operations.</p>
        """,
        variant="warn",
        tags=["High priority", "Operations", "RJ"],
    )

    info_card(
        "🔴 R2 — Treat 20 days as a hard service-level threshold",
        """
        <p><b>Evidence.</b> Delivery-time buckets degrade smoothly:
        4.45 → 4.36 → 4.22 → 3.69 → 2.26 stars. Every bucket past 20
        days falls below the 4.09 national average; the drop from
        "11-20 days" to "21-30 days" is the single largest step
        (−0.53 stars).</p>
        <p><b>Action.</b> Set a service-level agreement (SLA) that flags
        any order approaching day 20 and triggers proactive customer
        comms, expedited routing, or partial refund. Even shaving
        one bucket off the tail is measurable.</p>
        <p><b>Owner.</b> Logistics + Customer Experience.</p>
        """,
        variant="warn",
        tags=["High priority", "SLA", "Threshold"],
    )

    info_card(
        "🟡 R3 — Calibrate the estimated-delivery-date model",
        """
        <p><b>Evidence.</b> When orders <i>are</i> late, they run
        <b>9.55 days past estimate</b> on average, with a worst case of
        <b>189 days</b>. That is not a small slippage — it is a systematic
        optimism in the original estimate.</p>
        <p><b>Action.</b> Re-train the EDD (Estimated Delivery Date)
        model with the last 12 months of realised transit times, segmented
        by carrier and by destination state. Under-promising and over-
        delivering is a cheaper satisfaction lever than faster shipping.</p>
        <p><b>Owner.</b> Data Science + Logistics.</p>
        """,
        variant="info",
        tags=["Medium priority", "Forecasting", "Customer comms"],
    )

    info_card(
        "🟡 R4 — Investigate Bahia (BA) as the second-weakest market",
        """
        <p><b>Evidence.</b> BA scores 3.86 — the lowest of any state with
        2,000+ orders — and appears in the bottom quartile on every
        satisfaction cut.</p>
        <p><b>Action.</b> Run the same RJ playbook on BA: carrier mix,
        transit time distribution, review-text themes. BA's smaller volume
        (3,340 orders) makes it a faster, lower-risk pilot for any
        process change you want to prove before rolling out to RJ.</p>
        <p><b>Owner.</b> Operations / Regional.</p>
        """,
        variant="info",
        tags=["Medium priority", "BA", "Pilot"],
    )

    info_card(
        "🟡 R5 — Sentiment-analyse review_comment_message",
        """
        <p><b>Evidence.</b> The <code>order_reviews</code> table contains
        a free-text <code>review_comment_message</code> column that we
        have not used. 1★ reviews are ~11.5% of the total (11,424) — a
        rich, labelled corpus.</p>
        <p><b>Action.</b> Cluster the 1★ and 2★ comments by topic
        (delivery, product quality, seller comms, refund). Late delivery
        is currently a <i>hypothesis</i>; the review text is the
        <i>evidence</i> that names the actual failure mode.</p>
        <p><b>Owner.</b> Data Science / CX Analytics.</p>
        """,
        variant="info",
        tags=["Medium priority", "NLP", "Qualitative"],
    )

    info_card(
        "🟢 R6 — Adopt customer_unique_id as the master customer key",
        """
        <p><b>Evidence.</b> 99,441 <code>customer_id</code> rows collapse
        to 96,096 <code>customer_unique_id</code> values — a 3.4%
        duplication. Any future cohort, retention, or LTV work must key
        on <code>customer_unique_id</code>.</p>
        <p><b>Action.</b> Document this in the data dictionary and build
        a canonical <code>dim_customer</code> view before any repeat-
        purchase work begins.</p>
        <p><b>Owner.</b> Data Engineering.</p>
        """,
        variant="ok",
        tags=["Strategic", "Data model"],
    )

    info_card(
        "🟢 R7 — Shift the marketing promise to delivery certainty",
        """
        <p><b>Evidence.</b> On-time deliveries average 4.29 stars;
        late deliveries average 2.57. The on-time experience is already
        excellent — the damage is entirely concentrated in the miss.</p>
        <p><b>Action.</b> Reframe the customer promise from "fast
        delivery" to "delivered when we say" and back it with the R2 SLA.
        Certainty is cheaper to promise than speed, and the data says
        certainty is what drives the score.</p>
        <p><b>Owner.</b> Marketing + Logistics.</p>
        """,
        variant="ok",
        tags=["Strategic", "Brand", "Positioning"],
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — FUTURE RESEARCH
# ════════════════════════════════════════════════════════════════════════════
with tab_research:
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
        <p><b>Method.</b> OLS with review_score as the dependent variable.
        Regressors: <code>days_late</code> (continuous), destination-state
        fixed effects, product-category fixed effects, seller fixed
        effects, order value decile. Cluster standard errors by seller.</p>
        <p><b>Data.</b> All 5 loaded tables, plus optional geolocation for
        a computed great-circle distance.</p>
        <p><b>Expected output.</b> A coefficient on <code>days_late</code>
        with a confidence interval — the "true" per-day star penalty,
        separated from confounders.</p>
        """,
        variant="info",
        tags=["Econometrics", "Causal inference", "OLS"],
    )

    info_card(
        "🔬 F2 — Split lateness: seller-caused vs carrier-caused",
        """
        <p><b>Question.</b> Of the 7,826 late deliveries, how many were
        late because the seller handed over late, vs late because the
        carrier delivered late?</p>
        <p><b>Method.</b> Define two gaps:
        <code>seller_delay = order_delivered_carrier_date
        − order_approved_at</code>, and
        <code>carrier_delay = order_delivered_customer_date
        − order_delivered_carrier_date</code>. Compare each against a
        state-and-category-specific benchmark to identify who is over
        the line.</p>
        <p><b>Data.</b> <code>orders</code> table only — the
        <code>order_delivered_carrier_date</code> column is the key.</p>
        <p><b>Expected output.</b> A two-way attribution table: what
        share of lateness is seller-side vs carrier-side, split by state.
        This directly answers "who do we fire / who do we coach?"</p>
        """,
        variant="info",
        tags=["Attribution", "Root cause", "Carrier"],
    )

    info_card(
        "🔬 F3 — Topic-model the review_comment_message corpus",
        """
        <p><b>Question.</b> What are the actual themes inside 1★ and 2★
        reviews? Is "late delivery" the dominant theme, or a proxy for
        something else (product damage, wrong item, seller silence)?</p>
        <p><b>Method.</b> BERTopic or LDA over the
        <code>review_comment_message</code> corpus, restricted to
        Portuguese. Join topic labels back to delivery_status and to
        customer_state.</p>
        <p><b>Data.</b> <code>order_reviews</code> joined to
        <code>orders</code>.</p>
        <p><b>Expected output.</b> A topic × state heatmap showing which
        themes dominate where — RJ's complaint profile may look nothing
        like BA's.</p>
        """,
        variant="info",
        tags=["NLP", "Topic modeling", "Portuguese"],
    )

    info_card(
        "🔬 F4 — Model seasonality and the Nov 2017 dip",
        """
        <p><b>Question.</b> Nov 2017 is an anomaly — scores fall, volume
        spikes. Is this a Black-Friday effect, a logistics failure, or a
        data artefact?</p>
        <p><b>Method.</b> Decompose the monthly score series with STL
        (Seasonal-Trend decomposition using LOESS). Cross-reference the
        residual with <code>order_items</code> volume and with carrier
        performance in the same month.</p>
        <p><b>Data.</b> <code>orders</code>, <code>order_items</code>,
        <code>reviews</code>.</p>
        <p><b>Expected output.</b> A normalised seasonal curve, plus a
        clean read on whether Nov 2017 is repeatable (and therefore
        plannable) or a one-off.</p>
        """,
        variant="info",
        tags=["Time series", "STL", "Seasonality"],
    )

    info_card(
        "🔬 F5 — Cohort retention using customer_unique_id",
        """
        <p><b>Question.</b> Do customers who experience one late delivery
        come back? And do they come back at the same rate as customers
        who never experienced one?</p>
        <p><b>Method.</b> Build acquisition-month cohorts keyed on
        <code>customer_unique_id</code>. Flag each customer's first-order
        delivery_status. Compare 30/60/90-day return rates between the
        late-first-order and on-time-first-order groups.</p>
        <p><b>Data.</b> <code>orders</code> + <code>customers</code> +
        <code>reviews</code>.</p>
        <p><b>Expected output.</b> A survival curve of repeat purchase
        by first-delivery outcome. If the late group churns faster, the
        1.73-star cost is only the visible half of the damage.</p>
        """,
        variant="info",
        tags=["Retention", "Cohorts", "LTV"],
    )

    info_card(
        "🔬 F6 — Compute real delivery distance and its star cost",
        """
        <p><b>Question.</b> How much of the 1.73-star gap is explained
        by raw distance, before we blame the carrier?</p>
        <p><b>Method.</b> Load
        <code>olist_geolocation_dataset.csv</code> (1M rows). Aggregate
        to zip-code-prefix centroid. Compute great-circle distance from
        the seller's zip to the customer's zip. Include distance as a
        regressor alongside lateness in the F1 model.</p>
        <p><b>Data.</b> All 5 loaded tables + geolocation.</p>
        <p><b>Expected output.</b> Distance-adjusted lateness effect —
        the version of the 1.73 number that would survive a rigorous
        peer review.</p>
        """,
        variant="info",
        tags=["Geospatial", "Haversine", "Distance"],
    )

    info_card(
        "🔬 F7 — Payment instalments vs expectations",
        """
        <p><b>Question.</b> Customers who pay in 10 instalments may
        tolerate delays differently from customers who paid upfront.
        Does <code>payment_installments</code> moderate the lateness →
        score relationship?</p>
        <p><b>Method.</b> Add <code>payment_installments</code> and
        <code>payment_type</code> as interaction terms in the F1
        regression.</p>
        <p><b>Data.</b> <code>payments</code> joined to
        <code>orders</code> + <code>reviews</code>.</p>
        <p><b>Expected output.</b> A moderator analysis. If instalment
        payers are more forgiving, that changes who gets which
        retention offer.</p>
        """,
        variant="info",
        tags=["Moderation", "Payments", "Behaviour"],
    )

    info_card(
        "🔬 F8 — External validation on a second marketplace",
        """
        <p><b>Question.</b> Is 1.73 stars a universal e-commerce effect,
        or an Olist-specific number?</p>
        <p><b>Method.</b> Replicate the headline analysis on a public
        dataset from a different marketplace (e.g. Amazon reviews with
        delivery metadata, or a non-Brazilian logistics dataset).</p>
        <p><b>Data.</b> External — beyond the Olist corpus.</p>
        <p><b>Expected output.</b> A generalisable (or bounded) claim.
        Without this, we can only report what Olist's data says.</p>
        """,
        variant="info",
        tags=["External validity", "Replication"],
    )


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "**Reference (filters = All):** Total Orders 99,441 · Delivered 96,478 · "
    "Avg Review 4.09 · Late 7,826 · Avg Delivery Days 12.56. "
    "Every filter-driven chart responds to both Year and State. "
    "Chart 1 is intentionally unfiltered (whole-dataset fact). "
    "**Tabs:** Overview · Customer Satisfaction · Detail & Method · "
    "Limitations · Recommendations · Future Research."
)