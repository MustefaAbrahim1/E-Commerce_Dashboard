import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import threading
import time
from sqlalchemy import create_engine,text
import streamlit as st

# Establish database connection
conn = psycopg2.connect(
    dbname="examdb",   # database name
    user="postgres",   # PostgreSQL username
    password="musabr61",   # PostgreSQL password
    host="localhost",  # PostgreSQL server address
    port="5432"        # PostgreSQL port
)

def fetch_data(query):
    """Fetches data from the database using the given query."""
    return pd.read_sql_query(query, conn)

# Streamlit App Title
st.title("Enhanced E-Commerce Dashboard")

# Sidebar: Cascading Filters
st.sidebar.header("Filters")
# Fetch vendor list
query_vendors = "SELECT DISTINCT vendor_id FROM products WHERE approved = 'APPROVED'"
vendors = fetch_data(query_vendors)
vendor_list = vendors['vendor_id'].tolist()
selected_vendor = st.sidebar.selectbox("Select Vendor", options=["All"] + vendor_list)

# Fetch products based on selected vendor
if selected_vendor != "All":
    query_products = f"SELECT DISTINCT id AS product_id FROM products WHERE vendor_id = {selected_vendor} AND approved = 'APPROVED'"
else:
    query_products = "SELECT DISTINCT id AS product_id FROM products WHERE approved = 'APPROVED'"
products = fetch_data(query_products)
product_list = products['product_id'].tolist()
selected_product = st.sidebar.selectbox("Select Product", options=["All"] + product_list)

# Multi-select widget for comparing performance metrics
metrics_options = ["Revenue", "Conversion Rate", "User Retention"]
selected_metrics = st.sidebar.multiselect("Select Metrics to Compare", options=metrics_options)

# Tabbed Navigation
tab1, tab2, tab3, tab4 = st.tabs(["Heatmap Analysis", "Order Trend Forecasting", "Group vs Individual Deals", "Performance Metrics"])

# Tab 1: Heatmap Analysis
with tab1:
    st.header("Heatmap Analysis: Vendor Contribution")

    # Filtered heatmap query
    heatmap_query = """
    SELECT 
        p.vendor_id,
        p.status,
        COUNT(p.id) AS product_count
    FROM products p
    WHERE p.approved = 'APPROVED'
    """
    if selected_vendor != "All":
        heatmap_query += f" AND p.vendor_id = {selected_vendor}"
    heatmap_query += " GROUP BY p.vendor_id, p.status"

    heatmap_data = fetch_data(heatmap_query)
    pivot_heatmap = heatmap_data.pivot(index='vendor_id', columns='status', values='product_count').fillna(0)

    # Generate the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_heatmap, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Product Count'}, ax=ax)
    ax.set_title('Product Status Contribution by Vendor')
    ax.set_xlabel('Status')
    ax.set_ylabel('Vendor ID')
    st.pyplot(fig)

# Tab 2: Order Trend Forecasting
with tab2:
    st.header("Order Trend Forecasting")

    # Fetch data for orders over time
    query_timeseries = """
    SELECT DATE_TRUNC('month', created_at) AS order_month, 
           SUM(total_amount) AS total_sales
    FROM orders
    WHERE status = 'COMPLETED'
    GROUP BY order_month
    ORDER BY order_month
    """
    ts_data = fetch_data(query_timeseries)
    ts_data['order_month'] = pd.to_datetime(ts_data['order_month'])
    ts_data.set_index('order_month', inplace=True)

    # Fit ARIMA Model
    model = ARIMA(ts_data['total_sales'], order=(2, 1, 2))
    results = model.fit()

    # Forecast the next 12 months
    forecast = results.get_forecast(steps=12)
    forecast_index = pd.date_range(start=ts_data.index[-1], periods=12, freq='M')
    forecast_values = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # Plot Time-Series Data and Forecast
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts_data['total_sales'], label="Observed Sales")
    ax.plot(forecast_index, forecast_values, label="Forecasted Sales", linestyle="--")
    ax.fill_between(forecast_index, 
                    forecast_conf_int.iloc[:, 0], 
                    forecast_conf_int.iloc[:, 1], 
                    color='gray', alpha=0.2, label="Confidence Interval")
    ax.set_title("Time-Series Forecast for Total Sales")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.legend()
    st.pyplot(fig)

# Tab 3: Group vs Individual Deals
with tab3:
    st.header("Group vs Individual Deals Analysis")

    # Fetch data for average order amounts
    query_group_deals = """
    SELECT 
        CASE 
            WHEN gc.group_id IS NOT NULL THEN 'Group Deal' 
            ELSE 'Individual Deal' 
        END AS deal_type,
        AVG(o.total_amount) AS avg_amount
    FROM orders o
    LEFT JOIN groups_carts gc ON o.groups_carts_id = gc.id
    WHERE o.status = 'COMPLETED'
    GROUP BY deal_type
    """
    deal_data = fetch_data(query_group_deals)

    # Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(deal_data['deal_type'], deal_data['avg_amount'], color=['#4CAF50', '#FFC107'])
    ax.set_title("Average Order Amount: Group Deals vs Individual Deals")
    ax.set_xlabel("Deal Type")
    ax.set_ylabel("Average Total Amount")
    st.pyplot(fig)

# Tab 4: Performance Metrics
def fetch_data(query):
    """Fetches data from the database using the given query."""
    return pd.read_sql_query(query, conn)

# Performance Metrics Section
st.header("Performance Metrics Comparison")

# Revenue
st.subheader("Revenue by Vendor")
query_revenue = """
SELECT 
    p.vendor_id, 
    SUM(o.total_amount) AS revenue
FROM orders o
JOIN personal_cart_items pci ON o.personal_cart_id = pci.cart_id
JOIN products p ON pci.product_id = p.id
WHERE o.status = 'COMPLETED'
GROUP BY p.vendor_id
"""
revenue_data = fetch_data(query_revenue)
st.dataframe(revenue_data)

# Visualization: Revenue
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(revenue_data['vendor_id'], revenue_data['revenue'], color='skyblue')
ax.set_title("Revenue by Vendor")
ax.set_xlabel("Vendor ID")
ax.set_ylabel("Revenue")
plt.xticks(rotation=90)
st.pyplot(fig)

# Conversion Rate
st.subheader("Conversion Rate")
query_conversion_rate = """
WITH total_users AS (
    SELECT COUNT(DISTINCT cart_id) AS user_count
    FROM personal_cart_items
),
completed_orders AS (
    SELECT COUNT(DISTINCT o.personal_cart_id) AS completed_user_count
    FROM orders o
    WHERE o.status = 'COMPLETED'
)
SELECT 
    (CAST(completed_user_count AS FLOAT) / user_count) * 100 AS conversion_rate
FROM total_users, completed_orders
"""
conversion_rate_data = fetch_data(query_conversion_rate)
st.write(f"Conversion Rate: {conversion_rate_data['conversion_rate'][0]:.2f}%")

# User Retention
st.subheader("User Retention Rate")
query_user_retention = """
WITH user_orders AS (
    SELECT 
        pci.cart_id AS user_id, 
        COUNT(o.id) AS order_count
    FROM orders o
    JOIN personal_cart_items pci ON o.personal_cart_id = pci.cart_id
    WHERE o.status = 'COMPLETED'
    GROUP BY pci.cart_id
),
returning_users AS (
    SELECT COUNT(*) AS returning_user_count
    FROM user_orders
    WHERE order_count > 1
),
total_users AS (
    SELECT COUNT(*) AS total_user_count
    FROM user_orders
)
SELECT 
    (CAST(returning_user_count AS FLOAT) / total_user_count) * 100 AS retention_rate
FROM returning_users, total_users
"""
user_retention_data = fetch_data(query_user_retention)
st.write(f"User Retention Rate: {user_retention_data['retention_rate'][0]:.2f}%")

# ----- Real-Time Data Updates -----

# Placeholder for real-time updates
latest_order_data = None

def fetch_new_orders():
    """
    Fetch new orders every 5 minutes.
    """
    global latest_order_data
    while True:
        # Query to get the total number of orders for the last 5 minutes
        query_orders = """
        SELECT COUNT(id) AS new_orders
        FROM orders
        WHERE created_at > NOW() - INTERVAL '5 minutes'
        """
        order_data = fetch_data(query_orders)
        latest_order_data = order_data['new_orders'][0] if not order_data.empty else 0
        st.session_state.last_order_count = latest_order_data

        # Check for anomaly (significant drop in orders)
        check_for_anomaly()

        time.sleep(300)  # Sleep for 5 minutes (300 seconds)

def check_for_anomaly():
    """
    Checks for significant drop in the number of orders.
    """
    # Get previous order count from session state or set to 0 if not available
    prev_order_count = st.session_state.get("last_order_count", 0)

    if prev_order_count > 0:
        # If the drop in orders is greater than 50% (this threshold can be adjusted)
        if latest_order_data < prev_order_count * 0.5:
            st.session_state.anomaly_detected = True
            st.session_state.prev_order_count = prev_order_count
        else:
            st.session_state.anomaly_detected = False
            st.session_state.prev_order_count = latest_order_data
    else:
        st.session_state.anomaly_detected = False

def display_order_metrics():
    """
    Displays updated order metrics.
    """
    # Display the current order count
    st.subheader(f"New Orders in Last 5 Minutes: {latest_order_data}")
    
    if 'anomaly_detected' in st.session_state and st.session_state.anomaly_detected:
        st.warning("⚠️ Anomaly Detected: Sudden Drop in Orders!")
    
    # Display previous order count and the change
    if 'prev_order_count' in st.session_state:
        prev_count = st.session_state.prev_order_count
        if prev_count > 0:
            change = ((latest_order_data - prev_count) / prev_count) * 100
            st.write(f"Change in Orders: {change:.2f}% since last check.")
        else:
            st.write("No previous order count available for comparison.")

# Run the background process for fetching new orders in a separate thread
if 'last_order_count' not in st.session_state:
    st.session_state.last_order_count = 0

if 'anomaly_detected' not in st.session_state:
    st.session_state.anomaly_detected = False

# Start background thread for fetching new orders
thread = threading.Thread(target=fetch_new_orders, daemon=True)
thread.start()

# Display the order metrics in the main app
st.title("Real-Time E-Commerce Dashboard")

# Display the order metrics section
display_order_metrics()

# part 5: user segmentation Load data from the database

# Define the connection string
conn_url = "postgresql://postgres:musabr61@localhost:5432/examdb"
engine = create_engine(conn_url)
@st.cache_data
def load_user_segments():
    with engine.connect() as conn:
        query = "SELECT * FROM user_segments"
        data = pd.read_sql(text(query), conn)
    return data

# Load user segments
st.title("User Segmentation Dashboard")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Overview", "User Insights"])

if page == "Overview":
    st.header("Segmented Users Overview")
    
    # Load data
    user_segments = load_user_segments()
    
    if not user_segments.empty:
        # Show data summary
        st.write("## Segmentation Summary")
        st.write(user_segments.head())  # Display the first few rows
        
        # Show the distribution of users in each segment
        st.write("## User Distribution by Segment")
        segment_counts = user_segments["segment"].value_counts()
        st.bar_chart(segment_counts)
        
    else:
        st.warning("No data available in the `user_segments` table.")

elif page == "User Insights":
    st.header("Detailed User Insights")
    
    # Load data
    user_segments = load_user_segments()
    
    if not user_segments.empty:
        # User selection for insights
        user_id = st.selectbox("Select a User ID:", user_segments["user_id"].unique())
        user_data = user_segments[user_segments["user_id"] == user_id]
        
        if not user_data.empty:
            st.write(f"Details for User ID: {user_id}")
            st.json(user_data.iloc[0].to_dict())
        else:
            st.warning("No data found for the selected User ID.")
    else:
        st.warning("No data available in the `user_segments` table.")