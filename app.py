# =====================================================================
# Olist E-Commerce Customer Satisfaction Dashboard
# Capstone Project — Group 1
#
# To run:
#   1. pip install streamlit pandas plotly
#   2. Ensure the folder 'data' (next to this file) contains the 4 CSVs:
#         olist_orders_dataset.csv
#         olist_customers_dataset.csv
#         olist_order_items_dataset.csv
#         olist_order_reviews_dataset.csv
#   3. python -m streamlit run app.py
# =====================================================================

from __future__ import annotations

import json
import sqlite3
import traceback
from pathlib import Path
from typing import Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Olist Customer Satisfaction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CUSTOM CSS FOR MODERN UI (Based on reference images)
# =====================================================================
st.markdown("""
<style>
    /* General Background */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] > label {
        color: #555555;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] > div {
        color: #2c3e50;
        font-weight: 700;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #666;
    }
    .stTabs [aria-selected="true"] {
        color: #6c5ce7 !important;
        border-bottom: 3px solid #6c5ce7 !important;
    }
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# PATHS & CONSTANTS
# =====================================================================

DATA_DIR: Final[Path] = Path(__file__).parent / "data"
DB_PATH:  Final[Path] = Path(__file__).parent / "olist_dashboard.db"

HART_FIGSIZE: Final[tuple[float, float]] = (8.0, 4.8)
PARTIAL_YEARS: Final[frozenset[str]] = frozenset({"2016", "2018"})
MIN_REVIEWS_PER_MONTH: Final[int] = 30
DATA_CACHE_TTL: Final[int] = 3600

TABLES: Final[dict[str, str]] = {
    "orders":      "olist_orders_dataset.csv",
    "customers":   "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "reviews":     "olist_order_reviews_dataset.csv",
}

REQUIRED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "orders": (
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ),
    "customers":   ("customer_id", "customer_unique_id", "customer_state"),
    "order_items": ("order_id", "price", "freight_value"),
    "reviews":     ("review_id", "order_id", "review_score"),
}

DATE_COLUMNS: Final[tuple[str, ...]] = (
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date", "review_creation_date",
    "review_answer_timestamp", "shipping_limit_date",
)

# =====================================================================
# DATABASE SETUP
# =====================================================================

def clean_dataset(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLUMNS[table_name] if c not in df.columns]
    if missing:
        raise ValueError(f"Table '{table_name}' is missing required columns: {missing}.")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x).replace("", pd.NA)

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "customer_state" in df.columns:
        df["customer_state"] = df["customer_state"].str.upper()
    if table_name == "reviews" and "review_score" in df.columns:
        df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")

    return df

def setup_database() -> None:
    if DB_PATH.exists():
        return

    st.info("Database not found. Building from CSV files...")
    if not DATA_DIR.exists():
        st.error(f"Data directory not found at '{DATA_DIR}'.")
        st.stop()

    conn = sqlite3.connect(DB_PATH)
    try:
        for table_name, filename in TABLES.items():
            file_path = DATA_DIR / filename
            if not file_path.exists():
                st.error(f"Missing file: {filename}")
                st.stop()
            
            df = pd.read_csv(file_path)
            cleaned_df = clean_dataset(table_name, df)
            cleaned_df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()
    st.success("Database built successfully!")

# Ensure DB is built before proceeding
try:
    setup_database()
except Exception as exc:
    st.error(f"Database build FAILED: {exc}")
    st.stop()

@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# =====================================================================
# FILTER CONTRACT HELPERS
# =====================================================================

def year_params(year: str) -> tuple:
    return (year, year)

def multi_params(multi_json: str) -> tuple:
    return (multi_json, multi_json)

def year_multi(year: str, multi_json: str) -> tuple:
    return (year, year, multi_json, multi_json)

# =====================================================================
# QUERY FUNCTIONS
# =====================================================================

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def get_kpis(_conn, year: str) -> pd.DataFrame:
    # Q1
    query = """
    SELECT COUNT(DISTINCT o.order_id) AS total_orders,
           COUNT(DISTINCT CASE WHEN o.order_status='delivered' THEN o.order_id END) AS delivered_orders,
           (SELECT ROUND(AVG(CAST(review_score AS REAL)),2) FROM reviews) AS avg_review_score,
           SUM(CASE WHEN o.order_status='delivered' AND o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_deliveries,
           ROUND(AVG(CASE WHEN o.order_status='delivered' AND o.order_delivered_customer_date IS NOT NULL THEN julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp) END),2) AS avg_delivery_days
    FROM orders o
    WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
    """
    return pd.read_sql_query(query, _conn, params=year_params(year))

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def get_review_score_distribution(_conn) -> pd.DataFrame:
    # Q2
    query = """
    SELECT CAST(review_score AS INTEGER) AS score,
           COUNT(*) AS reviews,
           ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM reviews),1) AS pct
    FROM reviews GROUP BY 1 ORDER BY 1
    """
    return pd.read_sql_query(query, _conn)

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def get_monthly_avg_score(_conn, year: str, min_reviews: int) -> pd.DataFrame:
    # Q3
    query = """
    SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
           COUNT(*) AS reviews,
           ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
    FROM orders o JOIN reviews r ON o.order_id = r.order_id
    WHERE (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
    GROUP BY month HAVING COUNT(*) >= ? ORDER BY month
    """
    return pd.read_sql_query(query, _conn, params=year_params(year) + (min_reviews,))

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def get_avg_score_by_state(_conn, state_json: str) -> pd.DataFrame:
    # Q4
    query = """
    SELECT c.customer_state AS state,
           COUNT(DISTINCT o.order_id) AS orders,
           ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN reviews r  ON o.order_id  = r.order_id
    WHERE (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
    GROUP BY state ORDER BY orders DESC LIMIT 10
    """
    return pd.read_sql_query(query, _conn, params=multi_params(state_json))

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def get_delivery_by_state(_conn, state_json: str) -> pd.DataFrame:
    # Q5
    query = """
    SELECT c.customer_state AS state,
           COUNT(DISTINCT o.order_id) AS delivered,
           SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS late,
           ROUND(100.0*SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END)/COUNT(*),1) AS late_pct
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status='delivered' AND o.order_delivered_customer_date IS NOT NULL
      AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
    GROUP BY state ORDER BY delivered DESC LIMIT 5
    """
    return pd.read_sql_query(query, _conn, params=multi_params(state_json))

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def get_punctuality_vs_satisfaction(_conn, state_json: str) -> pd.DataFrame:
    # Q6 (Advanced Feature)
    query = """
    SELECT CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN 'On Time' ELSE 'Late' END AS delivery_status,
           COUNT(DISTINCT o.order_id) AS orders,
           ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score,
           ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)),2) AS avg_delivery_days
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN reviews r  ON o.order_id  = r.order_id
    WHERE o.order_status='delivered' AND o.order_delivered_customer_date IS NOT NULL
      AND (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
    GROUP BY 1 ORDER BY avg_score DESC
    """
    return pd.read_sql_query(query, _conn, params=multi_params(state_json))

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def get_satisfaction_by_bucket(_conn, year: str) -> pd.DataFrame:
    # Q9 (Bonus)
    query = """
    SELECT CASE
             WHEN julianday(o.order_delivered_customer_date)-julianday(o.order_purchase_timestamp) <= 5  THEN '0-5 days'
             WHEN julianday(o.order_delivered_customer_date)-julianday(o.order_purchase_timestamp) <= 10 THEN '6-10 days'
             WHEN julianday(o.order_delivered_customer_date)-julianday(o.order_purchase_timestamp) <= 20 THEN '11-20 days'
             WHEN julianday(o.order_delivered_customer_date)-julianday(o.order_purchase_timestamp) <= 30 THEN '21-30 days'
             ELSE '30+ days' END AS delivery_bucket,
           COUNT(DISTINCT o.order_id) AS orders,
           ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
    FROM orders o JOIN reviews r ON o.order_id = r.order_id
    WHERE o.order_status='delivered' AND o.order_delivered_customer_date IS NOT NULL
      AND (? = 'All' OR strftime('%Y', o.order_purchase_timestamp) = ?)
    GROUP BY 1 ORDER BY 1
    """
    return pd.read_sql_query(query, _conn, params=year_params(year))

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def get_lowest_scoring_states(_conn, state_json: str) -> pd.DataFrame:
    # Q12 (Bonus)
    query = """
    SELECT c.customer_state AS state,
           COUNT(DISTINCT o.order_id) AS orders,
           ROUND(AVG(CAST(r.review_score AS REAL)),3) AS avg_score
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN reviews r  ON o.order_id  = r.order_id
    WHERE (? = '[]' OR c.customer_state IN (SELECT value FROM json_each(?)))
    GROUP BY state HAVING orders >= 2000
    ORDER BY avg_score ASC LIMIT 10
    """
    return pd.read_sql_query(query, _conn, params=multi_params(state_json))

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def get_states(_conn) -> list[str]:
    df = pd.read_sql_query("SELECT DISTINCT customer_state FROM customers ORDER BY customer_state", _conn)
    return df["customer_state"].tolist()

# =====================================================================
# CHART FUNCTIONS
# =====================================================================

def _apply_hart_size(fig: go.Figure) -> None:
    fig.update_layout(
        width=HART_FIGSIZE[0] * 100,
        height=HART_FIGSIZE[1] * 100,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

def plot_review_score_distribution(df: pd.DataFrame) -> None:
    fig = px.bar(
        df, x="score", y="reviews", text="reviews",
        title="Distribution of Review Scores",
        labels={"score": "Review Score", "reviews": "Number of Reviews"},
        color_discrete_sequence=["#6c5ce7"],
    )
    fig.update_traces(
        texttemplate="%{text:,}<br>(%{customdata[0]}%)",
        customdata=df["pct"],
        textposition="outside",
    )
    fig.update_layout(title_x=0.5, xaxis=dict(tickmode="linear"), showlegend=False)
    _apply_hart_size(fig)
    st.plotly_chart(fig, use_container_width=True)

def plot_monthly_avg_score(df: pd.DataFrame, year: str) -> None:
    if df.empty:
        st.info(f"No months with at least {MIN_REVIEWS_PER_MONTH} reviews for the selected period.")
        return
    fig = px.line(
        df, x="month", y="avg_score", markers=True,
        title="Monthly Average Review Score",
        labels={"month": "Month", "avg_score": "Average Review Score"},
        color_discrete_sequence=["#6c5ce7"],
    )
    if year == "All":
        for py in PARTIAL_YEARS:
            months = df.loc[df["month"].str.startswith(py), "month"]
            if months.empty:
                continue
            fig.add_vrect(
                x0=months.min(), x1=months.max(),
                fillcolor="orange", opacity=0.1, line_width=0,
                annotation_text=f"{py} (partial)",
                annotation_position="top left",
            )
    fig.update_layout(title_x=0.5, yaxis_range=[1, 5])
    _apply_hart_size(fig)
    st.plotly_chart(fig, use_container_width=True)

def plot_avg_score_by_state(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available for the selected states.")
        return
    fig = px.bar(
        df.sort_values("avg_score", ascending=True),
        x="avg_score", y="state", orientation="h", text="avg_score",
        title="Average Review Score by State (Top 10 by Orders)",
        labels={"avg_score": "Average Review Score", "state": "State"},
        color="avg_score", color_continuous_scale="Purples",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="inside")
    fig.update_layout(title_x=0.5, xaxis_range=[1, 5], coloraxis_showscale=False)
    _apply_hart_size(fig)
    st.plotly_chart(fig, use_container_width=True)

def plot_delivery_by_state(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available for the selected states.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="On Time", x=df["state"],
        y=df["delivered"] - df["late"], marker_color="#00b894",
    ))
    fig.add_trace(go.Bar(
        name="Late", x=df["state"],
        y=df["late"], marker_color="#d63031",
    ))
    y2_max = max(float(df["late_pct"].max()) * 1.5, 5.0)
    fig.add_trace(go.Scatter(
        x=df["state"], y=df["late_pct"], name="Late %",
        mode="lines+markers+text",
        text=df["late_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="top center", yaxis="y2",
        line=dict(color="#2d3436", dash="dot"),
    ))
    fig.update_layout(
        barmode="stack",
        title_text="On-Time vs. Late Deliveries by State (Top 5)",
        title_x=0.5, xaxis_title="State", yaxis_title="Number of Orders",
        legend_title="Status",
        yaxis2=dict(title="Late Delivery %", overlaying="y", side="right", range=[0, y2_max], showgrid=False),
    )
    _apply_hart_size(fig)
    st.plotly_chart(fig, use_container_width=True)

def plot_punctuality_vs_satisfaction(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available for the selected states.")
        return
    fig = px.bar(
        df, x="delivery_status", y="avg_score", text="avg_score",
        color="delivery_status",
        title="Delivery Punctuality vs. Customer Satisfaction",
        labels={"delivery_status": "Delivery Status", "avg_score": "Average Review Score"},
        color_discrete_map={"On Time": "#00b894", "Late": "#d63031"},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(title_x=0.5, yaxis_range=[1, 5], showlegend=False)
    _apply_hart_size(fig)
    st.plotly_chart(fig, use_container_width=True)

def plot_satisfaction_by_bucket(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available for the selected period.")
        return
    fig = px.bar(
        df, x="delivery_bucket", y="avg_score", text="avg_score",
        title="Satisfaction by Delivery Time Bucket",
        labels={"delivery_bucket": "Delivery Time", "avg_score": "Average Review Score"},
        color="avg_score", color_continuous_scale="RdYlGn",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(title_x=0.5, yaxis_range=[1, 5], coloraxis_showscale=False)
    _apply_hart_size(fig)
    st.plotly_chart(fig, use_container_width=True)

def plot_lowest_scoring_states(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data available for the selected states.")
        return
    fig = px.bar(
        df, x="avg_score", y="state", orientation="h", text="avg_score",
        title="Lowest-Scoring States (min. 2,000 orders)",
        labels={"avg_score": "Average Review Score", "state": "State"},
        color="avg_score", color_continuous_scale="YlOrRd_r",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="inside")
    fig.update_layout(title_x=0.5, xaxis_range=[3, 4.5], coloraxis_showscale=False)
    _apply_hart_size(fig)
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# MAIN APP
# =====================================================================

def main() -> None:
    conn = get_db_connection()

    # --- Sidebar ---
    st.sidebar.header("Filter Options")
    STATES = get_states(conn)

    selected_year = st.sidebar.selectbox("Year", ["All", "2016", "2017", "2018"])
    if selected_year in PARTIAL_YEARS:
        st.sidebar.caption(f"⚠️ {selected_year} is a partial year — data is incomplete.")

    selected_state = st.sidebar.multiselect("State", STATES, default=STATES)
    state_json = json.dumps([] if not selected_state or "All" in selected_state else list(selected_state))

    st.sidebar.markdown("---")
    st.sidebar.caption("Built by Group 1. Data via Olist.")

    # --- Header ---
    st.title("🛍️ Olist Customer Satisfaction Dashboard")
    st.markdown("##### Customer Loyalty & Experience Drivers")
    st.markdown("---")

    # --- Tabs ---
    tab_overview, tab_theme, tab_detail = st.tabs(
        ["Overview", "Customer Satisfaction", "Detail & Method"]
    )

    # ============ OVERVIEW ============
    with tab_overview:
        st.header("Headline Performance Indicators")
        kpis = get_kpis(conn, selected_year)

        if kpis.empty or pd.isna(kpis.iloc[0]["total_orders"]):
            st.warning("No orders match the current filters.")
        else:
            k = kpis.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Orders", f"{int(k['total_orders']):,}")
            c2.metric("Delivered Orders", f"{int(k['delivered_orders']):,}")
            c3.metric("Avg Review Score", f"{k['avg_review_score']:.2f} / 5.0")
            c4.metric("Late Deliveries", f"{int(k['late_deliveries']):,}")
            c5.metric("Avg Delivery Days", f"{k['avg_delivery_days']:.2f} days")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Chart 1: Overall Review Score Distribution")
        st.caption("A whole-dataset fact — not affected by sidebar filters.")
        plot_review_score_distribution(get_review_score_distribution(conn))

        st.info(
            "**Key Insight:** The overall distribution is heavily skewed towards high scores (4 and 5 stars). "
            "This indicates generally high customer satisfaction, but it is crucial to investigate "
            "the drivers behind the lower scores. The **Customer Satisfaction** tab provides deeper insights."
        )

    # ============ CUSTOMER SATISFACTION ============
    with tab_theme:
        st.header("Customer Satisfaction Analysis")

        st.subheader("Chart 2: Monthly Average Review Score")
        st.caption(
            f"Filterable by year. Months with fewer than {MIN_REVIEWS_PER_MONTH} reviews are hidden. "
            "Look for the November 2017 dip, a known data anomaly. Shaded bands mark partial years."
        )
        plot_monthly_avg_score(get_monthly_avg_score(conn, selected_year, MIN_REVIEWS_PER_MONTH), selected_year)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Chart 3: Avg. Review Score by State")
            st.caption("Filterable by state. Top 10 states by order volume.")
            plot_avg_score_by_state(get_avg_score_by_state(conn, state_json))

        with col2:
            st.subheader("Chart 4: Delivery Punctuality by State")
            st.caption("Filterable by state. Top 5 states by delivery volume.")
            plot_delivery_by_state(get_delivery_by_state(conn, state_json))

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Chart 6 (Bonus): Satisfaction by Delivery Time")
            st.caption("Filterable by year. Shows a smoother decline in satisfaction.")
            plot_satisfaction_by_bucket(get_satisfaction_by_bucket(conn, selected_year))
        
        with col4:
            st.subheader("Chart 7 (Bonus): Lowest-Scoring States")
            st.caption("Filterable by state. Minimum 2,000 orders.")
            plot_lowest_scoring_states(get_lowest_scoring_states(conn, state_json))

    # ============ DETAIL & METHOD ============
    with tab_detail:
        st.header("Advanced Feature: Delivery Punctuality vs. Satisfaction")
        st.markdown(
            "This is the core insight of our analysis. We test the hypothesis that **late deliveries are the single most "
            "significant driver of customer dissatisfaction**."
        )

        punctuality_df = get_punctuality_vs_satisfaction(conn, state_json)

        if not punctuality_df.empty:
            on_time = punctuality_df.loc[punctuality_df["delivery_status"] == "On Time", "avg_score"]
            late = punctuality_df.loc[punctuality_df["delivery_status"] == "Late", "avg_score"]
            if not on_time.empty and not late.empty:
                diff = float(on_time.iloc[0]) - float(late.iloc[0])
                st.subheader(f"The Headline Story: Late deliveries cost you {diff:.2f} stars.")
                st.caption(f"On-time orders average **{float(on_time.iloc[0]):.2f}**; late ones average **{float(late.iloc[0]):.2f}**.")

        plot_punctuality_vs_satisfaction(punctuality_df)

        st.markdown("---")
        st.subheader("Data Limitations & Methodology")
        st.info(
            """
            **Data Limitations:**
            1. **Fan-out Tables:** The `reviews` table can have multiple rows per order. We consistently use `COUNT(DISTINCT order_id)` to avoid inflating order counts.
            2. **Partial Data (2016 & 2018):** The dataset is incomplete for 2016 and 2018. 2016 only has orders for Sep, Oct, and Dec. This is why monthly charts show gaps. We do not interpolate or smooth over these real data gaps.
            3. **GMV Definition:** We are not analyzing GMV in this dashboard. If we did, we would clearly state whether we are using 'all orders' GMV (15.8M) or 'delivered orders only' GMV (15.4M).
            4. **Customer Counts:** We count unique customers using the `customer_unique_id` field, which gives **96,096** unique customers, not the 99,441 `customer_id` rows.
            5. **Review Fan-out:** There are **99,224** review rows for **98,673** distinct orders. This discrepancy of 551 rows is why `COUNT(DISTINCT order_id)` is critical for order-based metrics.
            """
        )

if __name__ == "__main__":
    main()