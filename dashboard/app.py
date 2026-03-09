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
from dotenv import load_dotenv
load_dotenv()

USE_S3 = os.getenv('USE_S3', 'False').lower() == 'true'
S3_BUCKET = os.getenv('S3_BUCKET', '')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL1_PATH = os.path.join(BASE_DIR, '..', 'models', 'model1.joblib')
MODEL2_PATH = os.path.join(BASE_DIR, '..', 'models', 'model2.joblib')

@st.cache_data(show_spinner='Loading data...')
def load_data():
    if USE_S3:
        import s3fs
        return pd.read_csv(f's3://{S3_BUCKET}/cleaned_data.csv', storage_options={'anon': True})
    return pd.read_csv(os.path.join(DATA_DIR, 'cleaned_data.csv'))

@st.cache_resource(show_spinner=False)
def load_models1():
    if USE_S3:
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(f's3://{S3_BUCKET}/model1.joblib', 'rb') as f:
            return joblib.load(f)
    if os.path.exists(MODEL1_PATH):
        return joblib.load(MODEL1_PATH)
    return None

@st.cache_resource(show_spinner=False)
def load_models2():
    if USE_S3:
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(f's3://{S3_BUCKET}/model2.joblib', 'rb') as f:
            return joblib.load(f)
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
        colorbar=dict(title='Number of Orders'),
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Orders: %{z}<extra></extra>'
    ))
    st.plotly_chart(fig2, use_container_width=True)

# Q6 - Reorder Rate by Department
with tab3:
    if len(df_reorder) == 0:
        st.info("No reorder data available for the selected filters.")
    else:
        reorder_department = (df_reorder
            .groupby('department')['reordered'] 
            .mean()
            .reset_index()
            .rename(columns={'reordered': 'reorder_rate'})
        )
        avg_cart_size = reorder_department['reorder_rate'].mean()

        fig3 = go.Figure(go.Bar(
            x=reorder_department['reorder_rate'],
            y=reorder_department['department'],
            orientation='h',
            marker_color='steelblue',
            text=reorder_department['reorder_rate'].apply(lambda x: f"{x:.2%}"),
            textposition='outside'
        ))

        fig3.add_vline(
            x=avg_cart_size,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Avg Reorder Rate: {avg_cart_size:.2%}",
            annotation_position="top right"
        )

        fig3.update_layout(
            title=f"Reorder Rate by Department",
            xaxis_title="Reorder Rate",
            yaxis_title="Department",
            height=max(400, len(reorder_department) * 30)
        )

        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ML Predictions ----------------------------------------------------------------------
st.subheader("Machine Learning Predictions")
panel1, panel2 = st.columns(2)

# Machine Learning Model 1 - Next Product Prediction
with panel1:
    st.markdown("Next Product Prediction")
    st.caption("Predicting the next product a customer is likely to purchase based on their order history and other features.")

    if model1 is None:
        st.warning("Model 1 not found. Please ensure 'model1.joblib' is in the 'models' directory.")
    else:
        st.success("Model 1 loaded successfully!")
        st.info("This model predicts the next product a customer is likely to purchase based on their order history and other features.")

        # TODO: Add two columns with sliders for user input features (e.g., total_orders, avg_cart_size, reorder_rate, etc.)

        # TODO: Add a button to trigger the prediction and display the predicted product and its probability.

        # TODO: Add exception handling for the prediction process and display error messages if the prediction fails.

# Machine Learning Model 2 - Cart Size Prediction
with panel2:
    st.markdown("Cart Size Prediction")
    st.caption("Predicting the cart size a customer is likely to have in their next order based on their order history and other features.")

    if model2 is None:
        st.warning("Model 2 not found. Please ensure 'model2.joblib' is in the 'models' directory.")
    else:
        st.success("Model 2 loaded successfully!")
        st.info("This model predicts the cart size (number of products) a customer is likely to have in their next order based on their order history and other features.")
    
        # TODO: Add two columns with sliders for user input features (e.g., avg reorder rate, order hour of day, days of week, days since last order, etc.)

        # TODO: Add a button to trigger the prediction and display the predicted cart size and its confidence interval.

        # TODO: Add exception handling for the prediction process and display error messages if the prediction fails.

# DISCUSS THE ML PREDICTION WITH LEO AND THE FEATURE IMPORTANCE
# DISCUSS THE ML CART SIZE PREDICTION WITH MATHIS AND THE FEATURE IMPORTANCE
# MAYBE ADD A NEW VIEW FOR BETTER UX

# FOR THE DEPLOYMENT, WE CAN USE STREAMLIT CLOUD
# OR FOR LEARNING PURPOSE, WE CAN USE SNOWFLAKE TO HOST THE DATA AND THE MODELS, AND THEN CONNECT THE DASHBOARD TO SNOWFLAKE TO FETCH THE DATA AND THE PREDICTIONS IN REAL-TIME. THIS WAY, WE CAN SHOWCASE A MORE REALISTIC END-TO-END PIPELINE.
# ALSO AN OPPORTUNITY TO LEARN ABOUT SNOWFLAKE AND HOW TO INTEGRATE IT WITH PYTHON AND STREAMLIT.
# ALSO AN OPPORTUNITY TO LEARN AWS OR OTHER CLOUD PROVIDERS TO HOST THE DASHBOARD AND THE MODELS, AND THEN CONNECT EVERYTHING TOGETHER. THIS WAY, WE CAN SHOWCASE A MORE SCALABLE AND PRODUCTION-READY SOLUTION.
