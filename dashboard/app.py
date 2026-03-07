"""
NextBuy - Streamlit Dashboard Application
Owner: Pornraksa Suksawaeng
Run command: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Page configuration --------------------------------------------------------------
st.set_page_config(
    page_title="NextBuy Dashboard",
    page_icon="cart",
    layout="wide",
)

# Data and model loading -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL1_PATH = os.path.join(BASE_DIR, '..', 'models', 'model1.joblib')
MODEL2_PATH = os.path.join(BASE_DIR, '..', 'models', 'model2.joblib')

@st.cache_data(show_spinner='Loading data...')
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'cleaned_data.csv'))

@st.cache_resource(show_spinner=False)
def load_models1():
    if os.path.exists(MODEL1_PATH):
        return joblib.load(MODEL1_PATH)
    return None

@st.cache_resource(show_spinner=False)
def load_models2():
    if os.path.exists(MODEL2_PATH):
        return joblib.load(MODEL2_PATH)
    return None

# Load data ------------------------------------------------------------------------
try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'cleaned_data.csv' is in the 'data' directory.")
    st.stop()

# Load models ------------------------------------------------------------------------
model1 = load_models1()
model2 = load_models2()

# Sidebar filters -----------------------------------------------------------------------------
st.title("NextBuy Dashboard")
st.sidebar.caption("EPITECH B1 - Data Science Project - 2026")
st.sidebar.divider()
st.sidebar.subheader("Filters")

departments = ['All'] + sorted(df['department'].dropna().unique().tolist())
selected_department = st.sidebar.selectbox("Select Department", departments)

if selected_department != 'All':
    aisle_pool = df[df['department'] == selected_department]['aisle'].dropna().unique().tolist()
else:
    aisle_pool = df['aisle'].dropna().unique().tolist()

aisles = ['All'] + sorted(aisle_pool)
selected_aisle = st.sidebar.selectbox("Select Aisle", aisles)

st.sidebar.divider()
st.sidebar.caption('14M rows - 5 datasets - 12 analytics - 2 ML models')

# Filter data based on selections --------------------------------------------------------------
filtered_df = df.copy()
if selected_department != 'All':
    filtered_df = filtered_df[filtered_df['department'] == selected_department]
if selected_aisle != 'All':
    filtered_df = filtered_df[filtered_df['aisle'] == selected_aisle]

df_reorder = filtered_df[filtered_df['is_first_order'] == 0]

# Header ------------------------------------------------------------------------------
st.title("NextBuy - Customer Purchase Analysis")
st.caption("Analyzing customer purchase patterns and predicting next purchases")
st.divider()

# KPis cards -------------------------------------------------------------------------
total_orders = filtered_df['order_id'].nunique()
total_products = filtered_df['product_name'].nunique()
avg_cart_size = filtered_df.groupby('order_id')['product_id'].count().mean()
reorder_rate = df_reorder['reordered'].mean() if len(df_reorder) > 0 else 0
total_product = filtered_df['product_name'].value_counts().index[0] if len(filtered_df) > 0 else 'N/A'

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Unique Products", f"{total_products:,}")
col3.metric("Avg Cart Size", f"{avg_cart_size:.2f}")
col4.metric("Reorder Rate", f"{reorder_rate:.2%}")
col5.metric("Most Purchased Product", total_product)

st.caption(f'Top product: **{total_product}**')
st.divider()

# Charts -----------------------------------------------------------------------------
st.subheader("Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["Best Sellers", "Order Heatmap", "Reorder Rate by Department"])

# Q1 - Best Sellers
with tab1:
    top_n = st.slider("Number of products to display", min_value=5, max_value=30, value=10)

    total_products = (filtered_df
        .groupby('product_name')['order_id']
        .count()
        .nlargest(top_n)
        .reset_index()
        .rename(columns={'order_id': 'orders'})
        .sort_values('orders')
    )

    fig1 = go.Figure(go.Bar(
        x=total_products['orders'],
        y=total_products['product_name'],
        orientation='h',
        marker_color='steelblue',
        text=total_products['orders'],
        textposition='outside'
    ))

    fig1.update_layout(
        title=f"Top {top_n} Best-Selling Products",
        xaxis_title="Number of Orders",
        height=max(400, top_n * 30)
    )

    st.plotly_chart(fig1, use_container_width=True)

# Q2 - Order Heatmap
with tab2:
    order_dedup = filtered_df.drop_duplicates(subset=['order_id'])
    pivot = order_dedup.pivot_table(
        index='order_dow',
        columns='order_hour_of_day',
        values='order_id',
        aggfunc='count'
    )

    day_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    fig2 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{hour}:00" for hour in pivot.columns],
        y=[day_labels[day] for day in pivot.index],
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Orders: %{z}<extra></extra>'
    ))
    st.plotly_chart(fig2, use_container_width=True)
