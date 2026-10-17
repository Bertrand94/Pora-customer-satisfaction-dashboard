# olist_app.py
"""
Phase 2c Capstone — Group 1: Customer Satisfaction
PORA Academy Cohort 7 · Data Analysis & AI Automation

Complete Streamlit dashboard analysing Olist customer satisfaction.
Six curated charts, seven tabs (including the embedded 5-page summary
report), and fully connected Year + State filters across all KPIs and
visuals.
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
    "Both filters apply to every KPI card and every chart on the "
    "dashboard — including the Summary Report. Set to *All* to reproduce "
    "the unfiltered reference values."
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

if selected_year != "All" or "All" not in selected_state:
    st.sidebar.markdown(
        f"**Active filter:** Year = `{selected_year}` · "
        f"States = `{', '.join(selected_state)}`"
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — KPI ROW (fully connected to BOTH filters)
# ════════════════════════════════════════════════════════════════════════════
def kpi_values(conn, year, state_json):
    """
    Headline KPI row — fully connected to BOTH Year and State filters.
    avg_review_score is scoped by the same filters as the rest of the row.
    """
    return pd.read_sql("""
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
    """, conn, params=year_multi(year, state_json)).iloc[0]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CURATED CHARTS (6 total)
# ════════════════════════════════════════════════════════════════════════════

# ── Chart 1 — Review Score Distribution (not filterable: whole-dataset fact)
def chart_review_distribution(conn):
    """
    Whole-dataset review distribution. Deliberately unfiltered — this is
    a structural fact about the dataset that should not change when the
    user slices by year or state.
    """
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
    ax.set_title("Review Score Distribution (whole dataset)")
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
        "**Finding.** Bimodal shape — 57.8% of reviews are 5★, 11.5% are 1★. "
        "Deliberately unfiltered: a whole-dataset fact that shouldn't shift "
        "when the user slices."
    )


# ── Chart 2 — Monthly Average Review Score (filterable: Year + State)
def chart_monthly_score(conn, year, state_json):
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
        "Nov 2016 is empty — zero orders that month; the gap is real, "
        "not missing data."
    )


# ── Chart 3 — Brazil choropleth (top 10 states, labelled)
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
    """Top 10 states, labelled with state code + average score."""
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

    selected = json.loads(state_json)
    filter_active = len(selected) > 0

    top10 = all_states_df["state"].tolist()
    keep_states = sorted(set(top10) | set(selected)) if filter_active else top10

    kept_df = pd.DataFrame({"state": keep_states}).merge(
        all_states_df[["state", "orders", "avg_score"]],
        on="state", how="left",
    )
    kept_df["orders"]    = kept_df["orders"].fillna(0).astype(int)
    kept_df["avg_score"] = kept_df["avg_score"].fillna(0)

    if filter_active:
        kept_df["display_score"] = kept_df.apply(
            lambda row: row["avg_score"] if row["state"] in selected else -1,
            axis=1,
        )
    else:
        kept_df["display_score"] = kept_df["avg_score"]

    def hover_label(row):
        if filter_active and row["state"] not in selected:
            return "Not in current filter"
        if row["orders"] == 0:
            return "No orders"
        return f"{row['avg_score']:.2f} avg · {row['orders']:,} orders"

    kept_df["hover"] = kept_df.apply(hover_label, axis=1)

    centroids = _state_centroids()
    kept_df["lat"] = kept_df["state"].map(
        lambda s: centroids.get(s, (None, None))[0]
    )
    kept_df["lon"] = kept_df["state"].map(
        lambda s: centroids.get(s, (None, None))[1]
    )

    if filter_active:
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

    geojson = _load_brazil_geojson()

    color_scale = [
        [0.00, "#3F3F5A"], [0.05, "#3F3F5A"], [0.10, "#B83232"],
        [0.45, "#F0A0A0"], [0.65, "#C4B5FD"], [1.00, "#6D3FB5"],
    ]

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

    if not label_df.empty:
        fig.add_scattergeo(
            lat=label_df["lat"],
            lon=label_df["lon"],
            text=label_df["label"],
            mode="text",
            textfont=dict(
                size=11, color="#1F2937", family="sans-serif",
            ),
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
            text=("Average Review Score — Top 10 States (Brazil) "
                  "with labels"),
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
        "**Finding.** Top 10 states by order volume. BA (3.86) and RJ (3.88) "
        "anchor the low end; PR (4.18) and SP (4.17) hold the high end. "
        "RJ's low score against its wealthy Southeast neighbours signals "
        "an operational problem, not a regional one."
    )


# ── Chart 4 — On-Time vs Late by State
def chart_ontime_by_state(conn, year, state_json):
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


# ── Chart 5 — Punctuality vs Satisfaction (the headline chart)
def chart_punctuality_vs_satisfaction(conn, year, state_json):
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
        "on-time vs 2.57 late. The single strongest satisfaction driver "
        "in the dataset."
    )


# ── Chart 6 — Satisfaction by Delivery-Time Bucket
def chart_satisfaction_by_delivery_bucket(conn, year, state_json):
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
        "**Finding.** Smooth decline, not a cliff: 4.45 → 4.36 → 4.22 → "
        "3.69 → 2.26. Every bucket past 20 days falls below the 4.09 "
        "national average — the case for a 20-day SLA."
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HELPER CARD RENDERER
# ════════════════════════════════════════════════════════════════════════════
def info_card(title, body, variant="info", tags=None):
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
        concentrated in <b style="color:#D64545;">RJ</b> (13.5% late,
        2.3× the national rate).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

k = kpi_values(conn, selected_year, state_json)

# ── SEVEN TABS ─────────────────────────────────────────────────────────────
(
    tab_overview,
    tab_charts,
    tab_limitations,
    tab_recommendations,
    tab_departments,
    tab_research,
    tab_report,
) = st.tabs(
    [
        "🏠 Overview",
        "📊 Key Charts",
        "⚠️ Limitations",
        "✅ Recommendations",
        "🏢 Departments",
        "🔬 Future Research",
        "📄 Summary Report",
    ]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW (KPIs fully connected to both filters)
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders", f"{int(k['total_orders']):,}")
    c2.metric("Delivered Orders", f"{int(k['delivered_orders']):,}")
    c3.metric(
        "Avg Review Score",
        f"{k['avg_review_score']:.2f}",
        delta=(f"{k['avg_review_score'] - 4.09:+.2f} vs All"
               if selected_year != "All" or "All" not in selected_state
               else None),
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

    if selected_year != "All" or "All" not in selected_state:
        st.info(
            f"Active filter — **Year:** {selected_year} · "
            f"**States:** {', '.join(selected_state)}. "
            "Toggle to All to reproduce the reference values in the sidebar."
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
            <li><b>The decline is smooth, not a cliff.</b> Scores drop from
                4.45 to 2.26 across five delivery-time buckets; every
                bucket past 20 days falls below the national average.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    chart_punctuality_vs_satisfaction(conn, selected_year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — KEY CHARTS (5 filterable charts, curated from the original set)
# ════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.markdown(
        "Five curated, filterable charts. Redundant and low-value "
        "visuals from the original notebook were removed — every chart "
        "on this tab earns its place."
    )

    chart_review_distribution(conn)
    st.divider()
    chart_monthly_score(conn, selected_year, state_json)
    st.divider()
    chart_brazil_map(conn, selected_year, state_json)
    st.divider()
    chart_ontime_by_state(conn, selected_year, state_json)
    st.divider()
    chart_satisfaction_by_delivery_bucket(conn, selected_year, state_json)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIMITATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_limitations:
    st.markdown("## Limitations, Data Risks and Interpretation Boundaries")
    st.markdown(
        "Every analytical project has boundaries. Naming them honestly is "
        "part of the deliverable — it tells future analysts exactly where "
        "to pick up."
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
        October</b>. <b>2017 is the only complete calendar year</b>
        in this dataset.</p>
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
        distinct <code>customer_unique_id</code></b> values. Person-level
        retention or lifetime-value analysis should use the stable unique
        identifier.</p>
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
        be reconciled using distinct order IDs. The distinct-order counts
        shown in the code imply that this gap may be larger. Confirm
        with a direct anti-join before quoting a final count.</p>
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
        <p>Adding validated seller and customer postal-code centroids and
        calculating approximate distance would improve the diagnosis,
        although geographic distance would still not capture every route
        or infrastructure constraint.</p>
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
        <p>The dashboard's choropleth also fetches a public GeoJSON file
        at runtime, so map rendering depends on external availability and
        network access.</p>
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
        <p>The original notebook presents some state scores with
        slightly different rounding in different places (for example, RJ
        is shown as 3.87 in one caption and 3.88 in a chart). Establish
        one canonical query and rounding rule.</p>
        """,
        variant="warn",
        tags=["Metric consistency", "Rounding"],
    )

    st.divider()
    st.markdown("### What these limitations mean")
    st.markdown(
        """
        The dashboard is a **strong prioritisation tool**, but it should
        not be treated as a complete root-cause investigation. Its safest
        conclusion is that **lateness and lower satisfaction are strongly
        associated** and that **RJ deserves early operational attention**.
        Root cause, financial impact and causal improvement estimates
        require validation and follow-up analysis.
        """
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_recommendations:
    st.markdown("## Recommendations")
    st.markdown(
        "Each recommendation is tagged with its priority, the evidence "
        "that supports it, and the department best positioned to act. "
        "Priority is graded 🔴 High · 🟡 Medium · 🟢 Strategic."
    )

    info_card(
        "🔴 R1 — Investigate RJ delivery failures (highest ROI)",
        """
        <p><b>Evidence.</b> RJ carries 12,687 orders at a 13.5% late rate
        — 2.3× the 5.9% national average. RJ's average score (3.88) sits
        0.29 stars below PR (4.18).</p>
        <p><b>Action.</b> Audit RJ carriers, warehouse locations,
        last-mile routes, hand-off delays and performance by destination.
        Run a controlled RJ pilot before scaling.</p>
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
        carrier and destination state; back-test predicted versus actual
        dates; refresh the model periodically.</p>
        <p><b>Lead.</b> Data Science / Analytics + Logistics.</p>
        <p><b>Measures.</b> ETA error in days; share delivered by promised
        date; calibration by state/carrier.</p>
        """,
        variant="info",
        tags=["Medium priority", "Forecasting", "Customer comms"],
    )

    info_card(
        "🟡 R4 — Pilot improvement in Bahia (BA)",
        """
        <p><b>Evidence.</b> BA scores 3.86 — the lowest of any state with
        2,000+ orders — across ~3,340 orders.</p>
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
        <p><b>Evidence.</b> 99,441 <code>customer_id</code> rows collapse
        to 96,096 <code>customer_unique_id</code> values — 3.4%
        duplication.</p>
        <p><b>Action.</b> Document <code>customer_unique_id</code> as the
        stable person-level key; create a canonical customer dimension;
        validate joins before retention or lifetime-value analysis.</p>
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
        <p><b>Evidence.</b> On-time orders average 4.29 stars; late orders
        average 2.57. The damage is entirely in the <i>miss</i>, not the
        speed.</p>
        <p><b>Action.</b> Align delivery messaging with realistic
        estimates and operational service levels. Avoid promising a date
        the network cannot reliably meet.</p>
        <p><b>Lead.</b> Marketing / Customer Communications + Logistics.</p>
        <p><b>Measures.</b> On-time delivery; ETA accuracy; contacts
        about delivery; review score.</p>
        """,
        variant="ok",
        tags=["Strategic", "Brand", "Positioning"],
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEPARTMENTS (who owns what, cross-referenced to charts)
# ════════════════════════════════════════════════════════════════════════════
with tab_departments:
    st.markdown("## Departments — Ownership & Dashboard Hand-off")
    st.markdown(
        "Each department sees exactly which issue they own, which "
        "recommendation resolves it, and which dashboard chart they "
        "can open to watch the metric improve."
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
              <td>Overview — KPI row + headline banner</td>
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
        - **Operational owner:** Logistics / Operations — accountable for
          the delivery improvement plan and regional pilots.
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
        - **Days 1–30:** Validate the dashboard metrics, build an RJ
          root-cause view, define baseline KPIs.
        - **Days 31–60:** Pilot alerts and operational changes in RJ,
          while investigating BA and review comments.
        - **Days 61–90:** Compare results with the baseline, refine the
          ETA model, and decide which interventions to scale.

        *These timings are proposed for planning; they are not measured
        commitments in the supplied code.*
        """
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
        effects, order-value decile. Cluster SEs by seller.</p>
        <p><b>Data.</b> All 5 loaded tables, plus optional geolocation
        for a computed great-circle distance.</p>
        <p><b>Output.</b> Coefficient on <code>days_late</code> with a
        confidence interval — the "true" per-day star penalty.</p>
        """,
        variant="info",
        tags=["Econometrics", "Causal inference", "OLS"],
    )

    info_card(
        "🔬 F2 — Split lateness: seller-caused vs carrier-caused",
        """
        <p><b>Question.</b> Of the 7,826 late deliveries, how many were
        late because the seller handed over late vs late because the
        carrier delivered late?</p>
        <p><b>Method.</b> Define
        <code>seller_delay = order_delivered_carrier_date
        − order_approved_at</code> and
        <code>carrier_delay = order_delivered_customer_date
        − order_delivered_carrier_date</code>. Compare each against a
        state-and-category benchmark.</p>
        <p><b>Data.</b> <code>orders</code> only — the
        <code>order_delivered_carrier_date</code> column is the key.</p>
        <p><b>Output.</b> A two-way attribution table by state.</p>
        """,
        variant="info",
        tags=["Attribution", "Root cause", "Carrier"],
    )

    info_card(
        "🔬 F3 — Topic-model the review_comment_message corpus",
        """
        <p><b>Question.</b> What are the actual themes inside 1★/2★
        reviews?</p>
        <p><b>Method.</b> BERTopic or LDA over Portuguese comments; join
        labels back to delivery_status and customer_state.</p>
        <p><b>Data.</b> <code>order_reviews</code> + <code>orders</code>.</p>
        <p><b>Output.</b> Topic × state heatmap showing which themes
        dominate where.</p>
        """,
        variant="info",
        tags=["NLP", "Topic modeling", "Portuguese"],
    )

    info_card(
        "🔬 F4 — Model seasonality and the Nov 2017 dip",
        """
        <p><b>Question.</b> Is Nov 2017 a Black-Friday effect, a
        logistics failure, or a data artefact?</p>
        <p><b>Method.</b> STL decomposition of the monthly score series;
        cross-reference residual with order volume and carrier
        performance.</p>
        <p><b>Data.</b> <code>orders</code>, <code>order_items</code>,
        <code>reviews</code>.</p>
        <p><b>Output.</b> Normalised seasonal curve + repeatability
        verdict.</p>
        """,
        variant="info",
        tags=["Time series", "STL", "Seasonality"],
    )

    info_card(
        "🔬 F5 — Cohort retention using customer_unique_id",
        """
        <p><b>Question.</b> Do customers who experience one late delivery
        come back?</p>
        <p><b>Method.</b> Acquisition-month cohorts keyed on
        <code>customer_unique_id</code>; flag first-order delivery
        status; compare 30/60/90-day return rates.</p>
        <p><b>Data.</b> <code>orders</code> + <code>customers</code> +
        <code>reviews</code>.</p>
        <p><b>Output.</b> Survival curve of repeat purchase by
        first-delivery outcome.</p>
        """,
        variant="info",
        tags=["Retention", "Cohorts", "LTV"],
    )

    info_card(
        "🔬 F6 — Compute real delivery distance and its star cost",
        """
        <p><b>Question.</b> How much of the 1.73-star gap is explained by
        raw distance?</p>
        <p><b>Method.</b> Load geolocation CSV; aggregate to zip-prefix
        centroid; compute great-circle distance; add as regressor in
        F1.</p>
        <p><b>Data.</b> All 5 tables + geolocation.</p>
        <p><b>Output.</b> Distance-adjusted lateness effect — peer-review
        grade.</p>
        """,
        variant="info",
        tags=["Geospatial", "Haversine", "Distance"],
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
        <p><b>Output.</b> Moderator analysis — who is more forgiving,
        and who needs which retention offer.</p>
        """,
        variant="info",
        tags=["Moderation", "Payments", "Behaviour"],
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
        variant="info",
        tags=["External validity", "Replication"],
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — SUMMARY REPORT (the 5-page document, embedded)
# ════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("# Customer Satisfaction Analysis — Executive Summary Report")
    st.markdown(
        "**PORA Academy Cohort 7 · Data Analysis & AI Automation · "
        "Group 1 Capstone**  \n"
        "*Companion dashboard to the Olist Customer Satisfaction Streamlit "
        "application.*"
    )

    st.markdown(
        """
        <div class="callout">
        <b>Active filter:</b> Year = <code>{y}</code> · States = <code>{s}</code>.
        KPI cards above reflect the current sidebar selection. The report
        below presents the unfiltered reference results — use the
        Reference table in the sidebar to cross-check.
        </div>
        """.format(
            y=selected_year,
            s=", ".join(selected_state),
        ),
        unsafe_allow_html=True,
    )

    # ── Page 1 ──────────────────────────────────────────────────────────
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
        strongly associated with customer satisfaction</b>. The dashboard
        reports an average review score of <b>4.29 for on-time
        deliveries</b> and <b>2.57 for late deliveries</b> — a difference
        of <b>1.73 stars</b>. Overall, it lists 99,441 orders, 96,478
        delivered orders, an average review score of 4.09, 7,826 late
        deliveries and average delivery time of 12.56 days. The dashboard
        says late deliveries represent 8.1% of delivered orders and are,
        on average, 9.55 days beyond the estimated delivery date.</p>

        <p>The problem is not evenly distributed across Brazil. Rio de
        Janeiro (RJ) is highlighted as the main operational outlier, with
        a reported late-delivery rate of <b>13.5%</b>, compared with
        <b>5.9% nationally</b>, across 12,687 orders. Bahia (BA) is also
        flagged for a low average review score of approximately
        <b>3.86</b>. These patterns support prioritising regional
        logistics diagnostics rather than applying an undifferentiated
        nationwide intervention.</p>

        <h3>Key findings at a glance</h3>

        <table>
          <thead>
            <tr>
              <th>Indicator</th>
              <th>Dashboard result</th>
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
        uses the supplied code's stated results; it does not independently
        rerun the underlying CSV files. Therefore, numeric findings should
        be treated as dashboard-reported results pending data-quality
        validation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page 2 ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>2. Findings and Business Interpretation</h2>

        <h3>2.1 Delivery punctuality is the priority</h3>
        <p>The dashboard's most important comparison is <b>4.29</b>
        average stars for orders delivered on time versus <b>2.57</b> for
        late orders. This 1.73-star difference is a strong
        <i>association</i> in the observed data, but it is not proof that
        lateness alone causes the entire difference. Product quality,
        seller performance, destination distance and communication
        failures may contribute. Even with that caveat, the size of the
        gap makes delivery reliability a reasonable first operational
        priority.</p>

        <p>The delivery-time bucket analysis shows a progressive decline
        in average scores: <b>4.45</b> for 0–5 days, <b>4.36</b> for
        6–10 days, <b>4.22</b> for 11–20 days, <b>3.69</b> for 21–30
        days, and <b>2.26</b> for the longest bucket. The code
        specifically highlights the fall between 11–20 and 21–30 days.
        This supports an early-warning threshold before an order reaches
        20 days in transit, rather than waiting until a customer has
        already experienced a severe delay.</p>

        <h3>2.2 Regional performance varies</h3>
        <p>The state-level analysis identifies BA and RJ as low-scoring
        states, while Paraná (PR) and São Paulo (SP) are among the
        stronger performers. The code reports RJ's late rate at
        <b>13.5%</b>, about 2.3 times the stated national rate of
        <b>5.9%</b>. It recommends examining carrier mix, warehouse
        proximity and last-mile routes in RJ. BA, with a reported score
        of 3.86 and approximately 3,340 orders, is proposed as a second,
        smaller pilot market.</p>

        <p>Regional comparisons are useful for prioritising investigation,
        not for assigning blame. A state may differ in distance, order mix,
        seller composition, delivery promises or infrastructure. The
        dashboard does not yet adjust for all these factors. The
        operations team should compare like-for-like orders and examine
        seller hand-off time separately from carrier transit time before
        selecting corrective actions.</p>

        <h3>2.3 Trends and review distribution</h3>
        <p>The review distribution is described as bimodal:
        <b>57.8%</b> of reviews are five-star and <b>11.5%</b> are
        one-star. This suggests that many customers report positive
        experiences while a meaningful minority report serious
        dissatisfaction. The dashboard also calls attention to
        <b>November 2017</b>, when average monthly review score dips. It
        notes that November 2016 has no orders and that the absence
        should not be treated as a value to impute.</p>

        <p>The November 2017 dip should be investigated alongside
        order-volume spikes, seasonal promotions and carrier performance.
        The current dashboard identifies the anomaly but does not
        establish its cause. The analysis should therefore lead to a
        targeted investigation rather than a definitive claim that a
        particular campaign or logistics event caused the decline.</p>

        <h3>2.4 Overall interpretation</h3>
        <p>The best-supported management response is to improve
        <b>reliability</b>, <b>accuracy of delivery promises</b> and
        <b>communication during delays</b>. Faster delivery may help,
        but the code's strategic recommendation is to promise delivery
        dates that can be met consistently. Customer Experience should
        support Operations by notifying customers early, explaining
        revised delivery estimates and applying a consistent
        service-recovery policy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page 3 ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>3. Recommendations and Departmental Ownership</h2>

        <p>The actions below translate the code's recommendations into an
        implementation plan. The departments are proposed operational
        owners based on the responsibilities implied by the dashboard;
        the code does not identify named staff or confirm the
        organisation's internal structure.</p>

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
              <td>Audit carriers, warehouse locations, last-mile routes, hand-off delays and performance by destination. Run a controlled RJ pilot before scaling.</td>
              <td>Late-delivery rate; on-time rate; median and 90th-percentile transit time; average review score.</td>
            </tr>
            <tr>
              <td>2. Trigger intervention before day 20 (High)</td>
              <td>Logistics / Operations + Customer Experience (CX)</td>
              <td>Create ageing-order alerts; escalate at-risk parcels; provide revised ETAs; consider expedited routing or policy-approved compensation.</td>
              <td>Orders over 20 days; alert-to-resolution time; complaint rate; score for delayed orders.</td>
            </tr>
            <tr>
              <td>3. Recalibrate estimated delivery dates (Medium)</td>
              <td>Data Science / Analytics + Logistics</td>
              <td>Use realised transit history segmented by carrier and destination state; back-test predicted versus actual dates; refresh the model periodically.</td>
              <td>ETA error in days; share delivered by promised date; calibration by state/carrier.</td>
            </tr>
            <tr>
              <td>4. Pilot improvement in Bahia (Medium)</td>
              <td>Regional Operations + Logistics + CX</td>
              <td>Apply the RJ diagnostic checklist to BA; review carrier and transit distributions; read low-score comments; test changes on a limited scale.</td>
              <td>BA average score; late rate; order volume; customer complaints before/after.</td>
            </tr>
            <tr>
              <td>5. Analyse low-star review text (Medium)</td>
              <td>Customer Experience Analytics + Data Science</td>
              <td>Cluster 1- and 2-star comments into delivery, product quality, seller communication, refund and other themes; compare RJ and BA.</td>
              <td>Share of low reviews by topic; repeat complaint themes; time to close root causes.</td>
            </tr>
            <tr>
              <td>6. Standardise customer identity for future analytics (Strategic)</td>
              <td>Data Engineering / Data Governance</td>
              <td>Document customer_unique_id as the stable person-level key; create a canonical customer dimension; validate joins before retention or lifetime-value analysis.</td>
              <td>Duplicate rate; unmatched records; reconciliation checks; repeat-customer metric consistency.</td>
            </tr>
            <tr>
              <td>7. Promise reliability, not unsupported speed (Strategic)</td>
              <td>Marketing / Customer Communications + Logistics</td>
              <td>Align delivery messaging with realistic estimates and operational service levels. Avoid promising a date the network cannot reliably meet.</td>
              <td>On-time delivery; ETA accuracy; contacts about delivery; review score.</td>
            </tr>
          </tbody>
        </table>

        <h3>Governance and accountability</h3>
        <p><b>Executive sponsor:</b> Head of Operations or equivalent, to
        resolve cross-functional blockers and review monthly results.
        <b>Operational owner:</b> Logistics / Operations, accountable for
        the delivery improvement plan and regional pilots.
        <b>Customer outcome owner:</b> Customer Experience, accountable
        for proactive updates, complaint categorisation and service
        recovery. <b>Analytical support:</b> Data Science / Analytics,
        responsible for reliable metrics, ETA evaluation and review-text
        analysis. <b>Data controls:</b> Data Engineering / Data
        Governance, responsible for identity keys, data definitions and
        join-quality checks. <b>Commercial alignment:</b> Marketing,
        responsible for ensuring customer-facing promises match the
        delivery capability.</p>

        <h3>Suggested delivery sequence</h3>
        <p><b>First 30 days:</b> validate the dashboard metrics, build an
        RJ root-cause view and define baseline KPIs.
        <b>Days 31–60:</b> pilot alerts and operational changes in RJ,
        while investigating BA and review comments.
        <b>Days 61–90:</b> compare results with the baseline, refine the
        ETA model and decide which interventions to scale. These timings
        are proposed for planning; they are not measured commitments in
        the supplied code.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page 4 ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>4. Limitations, Data Risks and Interpretation Boundaries</h2>

        <h3>4.1 Observational results do not establish causality</h3>
        <p>The code explicitly recognises that the 1.73-star difference is
        <b>correlation, not causation</b>. Product category, distance,
        seller quality and other unmeasured variables may affect both
        delivery timing and ratings. Before claiming that a specific
        intervention will increase ratings by a precise amount, use a
        controlled pilot or a regression model that controls for relevant
        factors.</p>

        <h3>4.2 Incomplete time coverage</h3>
        <p>The dashboard notes that 2016 contains only September, October
        and December, with <b>no November orders</b>, while 2018 ends on
        <b>17 October</b>. It identifies <b>2017 as the only complete
        calendar year</b> in this dataset. Year-on-year comparisons that
        treat these periods as complete years may mislead. Reports should
        state the covered dates and compare equivalent periods where
        possible.</p>

        <h3>4.3 Review-row fan-out and customer identity</h3>
        <p>The code reports <b>99,224 review rows across 98,673 distinct
        orders</b>, a difference of 551 rows. Some orders have multiple
        review records. The dashboard protects order counts by using
        <code>COUNT(DISTINCT order_id)</code>, but raw joins can still
        inflate counts or distort averages if the grain of each table is
        not handled carefully. The customers table also has 99,441 rows
        but <b>96,096 distinct customer_unique_id values</b>;
        person-level retention or lifetime-value analysis should use the
        stable unique identifier.</p>

        <h3>4.4 Review coverage and possible selection bias</h3>
        <p>The code states that reviews are voluntary and that some
        orders may not have reviews, meaning the scores represent
        <b>reviewers rather than every customer</b>. However, the code's
        estimate of approximately 217 orders without reviews appears to
        compare total order rows with review rows; because review rows
        include repeated reviews for some orders, this estimate should be
        reconciled using distinct order IDs. The distinct-order counts
        shown in the code imply that this gap may be larger. Confirm this
        with a direct anti-join before quoting a final count.</p>

        <h3>4.5 No delivery-distance variable</h3>
        <p>The large geolocation CSV is deliberately excluded.
        Consequently, the current analysis <b>cannot distinguish
        long-distance transit from carrier underperformance</b>. Adding
        validated seller and customer postal-code centroids and
        calculating approximate distance would improve the diagnosis,
        although geographic distance would still not capture every route
        or infrastructure constraint.</p>

        <h3>4.6 Dataset and generalisability boundaries</h3>
        <p>The findings concern Olist in Brazil during 2016–2018. They may
        not generalise to other countries, marketplaces, logistics
        networks or current operating conditions. The dashboard's
        choropleth also fetches a public GeoJSON file at runtime, so map
        rendering depends on external availability and network access.</p>

        <h3>4.7 Metric and implementation checks</h3>
        <p>The code includes several metrics and chart-specific query
        definitions. Before executive use, reconcile the overall and
        chart-level denominators, validate the state late-rate
        calculation, and confirm that year/state filters behave
        consistently across all visuals. The code presents some state
        scores with slightly different rounding in different places (for
        example, RJ is shown as 3.87 in one recommendation and 3.88 in a
        chart caption). Establish one canonical query and rounding rule.</p>

        <h3>What these limitations mean</h3>
        <p>The dashboard is a strong <b>prioritisation tool</b>, but it
        should not be treated as a complete root-cause investigation. Its
        safest conclusion is that <b>lateness and lower satisfaction are
        strongly associated</b> and that <b>RJ deserves early operational
        attention</b>. Root cause, financial impact and causal improvement
        estimates require validation and follow-up analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page 5 ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="report-body">
        <h2>5. Implementation, Monitoring and Conclusion</h2>

        <h3>5.1 Recommended monitoring scorecard</h3>

        <table>
          <thead>
            <tr>
              <th>KPI</th>
              <th>Definition / use</th>
              <th>Suggested owner</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>On-time delivery rate</td>
              <td>Delivered orders delivered on or before estimated date, divided by eligible delivered orders.</td>
              <td>Logistics / Operations</td>
            </tr>
            <tr>
              <td>Late-delivery rate</td>
              <td>Late delivered orders divided by eligible delivered orders; report nationally and by state/carrier.</td>
              <td>Operations Analytics</td>
            </tr>
            <tr>
              <td>Average review score</td>
              <td>Average review score with review-grain and duplicate handling documented.</td>
              <td>CX Analytics</td>
            </tr>
            <tr>
              <td>Score gap by punctuality</td>
              <td>Average score for on-time minus late deliveries; trend over time and by state.</td>
              <td>CX Analytics / Data Science</td>
            </tr>
            <tr>
              <td>Long-delay share</td>
              <td>Share of delivered orders taking more than 20 days from purchase to delivery.</td>
              <td>Logistics</td>
            </tr>
            <tr>
              <td>ETA error</td>
              <td>Actual delivery date minus estimated date; track average and distribution, not only mean.</td>
              <td>Data Science / Logistics</td>
            </tr>
            <tr>
              <td>Low-star complaint themes</td>
              <td>Proportion of 1–2 star comments assigned to delivery, product, communication, refund or other topics.</td>
              <td>Customer Experience</td>
            </tr>
            <tr>
              <td>Data quality checks</td>
              <td>Distinct orders, duplicate review rows, missing review coverage, date completeness and join reconciliation.</td>
              <td>Data Engineering</td>
            </tr>
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
          <li>Topic-model Portuguese review comments and compare complaint
              themes by state and delivery status.</li>
          <li>Investigate November 2017 with seasonal decomposition and
              order-volume/carrier-performance comparisons.</li>
          <li>Study repeat purchase using <code>customer_unique_id</code>
              and compare customers whose first delivery was late with
              those whose first delivery was on time.</li>
          <li>Add approximate delivery distance and test whether distance
              explains part of the observed lateness association.</li>
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
        Science</b>, <b>Data Engineering</b>, <b>Regional Management</b>
        and <b>Marketing</b> as supporting teams. Use <b>BA</b> as a
        smaller comparison or pilot market where appropriate. Do not
        present projected score improvements as guaranteed outcomes;
        measure changes against a validated baseline.</p>

        <h3>Conclusion</h3>
        <p>The supplied dashboard indicates that <b>delivery reliability
        is the clearest actionable correlate of customer satisfaction</b>
        in the Olist data. On-time deliveries average 4.29 stars, while
        late deliveries average 2.57. RJ has a substantially higher
        reported late-delivery rate than the national rate, and
        satisfaction declines sharply when delivery exceeds 20 days. These
        findings justify immediate investigation and targeted operational
        pilots.</p>

        <p>The recommended response combines <b>regional logistics
        audits</b>, <b>proactive customer communication</b>,
        <b>realistic delivery estimates</b>, <b>review-text analysis</b>
        and <b>stronger data governance</b>. The key limitation is that
        the dashboard identifies patterns, not definitive causes.
        Validating the counts and denominators, separating seller and
        carrier delays, and measuring pilot outcomes will turn the current
        analysis into a more reliable management tool.</p>

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
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "**Reference (filters = All):** Total Orders 99,441 · Delivered 96,478 · "
    "Avg Review 4.09 · Late 7,826 · Avg Delivery Days 12.56. "
    "Both Year and State filters apply to every KPI and every chart on "
    "the **Overview** and **Key Charts** tabs. Chart 1 (Review Score "
    "Distribution) is intentionally unfiltered — a whole-dataset fact. "
    "**Tabs:** Overview · Key Charts · Limitations · Recommendations · "
    "Departments · Future Research · Summary Report."
)