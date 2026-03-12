"""
NextBuy ML Models Page
Reorder classifier (Léo) + Cart size regressor (Mathis)
Owner: Pornraksa Suksawaeng + Léo Bellard + Mathis Monnin
Run: streamlit run dashboard/pages/01_Dashboard.py
"""

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

# Page configuration ----------------------------------------------------------
st.set_page_config(
    page_title="NextBuy - ML Models",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment variables and paths ------------------------------------------------------
load_dotenv()

USE_S3 = os.getenv('USE_S3', 'False').lower() == 'true'
S3_BUCKET = os.getenv('S3_BUCKET', '')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL1_PATH = os.path.join(BASE_DIR, '..', '..', 'models', 'model1.joblib')
MODEL2_PATH = os.path.join(BASE_DIR, '..', '..', 'models', 'model2.joblib')

# Model loading function ------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path, s3_key):
    if USE_S3:
        import s3fs
        fs = s3fs.S3FileSystem()
        with fs.open(f'{S3_BUCKET}/{s3_key}', 'rb') as f:
            return joblib.load(f)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    st.error(f"Model file not found: {model_path}")
    return None

model1 = load_model(MODEL1_PATH, 'model1.joblib')
model2 = load_model(MODEL2_PATH, 'model2.joblib')

# Sidebar for model selection ------------------------------------------------------
st.sidebar.caption('EPITECH B1 - Data Science Project - 2026')
st.sidebar.divider()
st.sidebar.markdown('**Model status**')
st.sidebar.markdown(f'Reorder Classifier: {'Loaded' if model1 else 'Not Found'}')
st.sidebar.markdown(f'Cart Size Regressor: {'Loaded' if model2 else 'Not Found'}')
st.sidebar.divider()
st.sidebar.caption("14M rows · 5 datasets · 12 analyses · 2 ML models")

# Header ----------------------------------------------------------------------------
st.title('ML Models')
st.caption("This page showcases the machine learning models developed for NextBuy, including a reorder classifier and a cart size regressor. Use the sidebar to check model status.")
st.divider()

# Model 1: Reorder Classifier ------------------------------------------------------
st.header('Reorder Classifier')
st.caption('XGBoost classifier · Predicts whether a customer will reorder a specific product')

if model1 is None:
    st.warning("Reorder Classifier model not found. Please ensure 'model1.joblib' is available.")
else:
    with st.expander("How this model work", expanded=False):
        st.markdown("""
        **Algorithm:** XGBoost Classifier (200 trees, max_depth=8)

        **Pipeline steps:**
        1. `SimpleImputer` — fills missing values with the median
        2. `XGBClassifier` — predicts reorder probability (0 to 1)

        **Features used:** cart position, aisle, department, order history,
        user behaviour (total orders, avg days between orders),
        product popularity, and user-product interaction history.

        **Output:** probability that the customer will reorder this product.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Order context**")
        add_to_cart_order = st.slider("Position in cart", 1, 30, 3, key="m1_cart_pos",
            help="Where this product appears in the customer's cart (1 = first item added)")
        order_number = st.slider("Customer's total orders so far", 1, 100, 10, key="m1_order_num",
            help="How many orders this customer has placed in total")
        order_dow = st.selectbox("Day of week", 
            options=[0,1,2,3,4,5,6],
            format_func=lambda x: ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'][x],
            key="m1_dow")
        order_hour_of_day = st.slider("Hour of day", 0, 23, 10, key="m1_hour")
        days_since_prior_order = st.slider("Days since last order", 0, 30, 7, key="m1_days")
    
    with col2:
        st.markdown("**Product & user history**")
        aisle_id = st.slider("Aisle ID", 1, 134, 24, key="m1_aisle",
            help="Aisle the product belongs to (1–134)")
        department_id = st.slider("Department ID", 1, 21, 4, key="m1_dept",
            help="Department the product belongs to (1–21)")
        user_id = st.number_input("User ID", min_value=1, max_value=206209, value=1, key="m1_uid",
            help="Customer identifier")
        product_id = st.number_input("Product ID", min_value=1, max_value=49688, value=196, key="m1_pid",
            help="Product identifier")
        user_total_orders = st.slider("User total orders (engineered)", 1, 100, 10, key="m1_uto",
            help="Total number of orders placed by this user")
        user_avg_days_between_orders = st.slider("Avg days between orders (engineered)", 0.0, 30.0, 7.0, step=0.5, key="m1_avgdays")
    
    col3, col4 = st.columns(2)
    with col3:
        product_order_count = st.slider("Product order count (engineered)", 1, 200000, 1000, key="m1_poc",
            help="How many times this product has been ordered across all users")
        product_avg_cart_position = st.slider("Product avg cart position (engineered)", 1.0, 30.0, 5.0, step=0.5, key="m1_pacp")
    with col4:
        product_reorder_rate = st.slider("Product reorder rate (engineered)", 0.0, 1.0, 0.6, step=0.01, key="m1_prr")
        user_product_order_count = st.slider("User-product order count (engineered)", 1, 50, 3, key="m1_upoc",
            help="How many times this specific user has ordered this specific product")
        user_product_last_order = st.slider("User-product last order number (engineered)", 1, 100, 5, key="m1_uplo",
            help="Order number when this user last bought this product")
    
    if st.button("Predict Reorder Probability", use_container_width=True, type="primary", key="btn_m1"):
        input_df = pd.DataFrame([{
            'add_to_cart_order': add_to_cart_order,
            'aisle_id': aisle_id,
            'department_id': department_id,
            'order_number': order_number,
            'order_dow': order_dow,
            'order_hour_of_day': order_hour_of_day,
            'days_since_prior_order': days_since_prior_order,
            'user_id': user_id,
            'product_id': product_id,
            'user_total_orders': user_total_orders,
            'user_avg_days_between_orders': user_avg_days_between_orders,
            'product_order_count': product_order_count,
            'product_avg_cart_position': product_avg_cart_position,
            'product_reorder_rate': product_reorder_rate,
            'user_product_order_count': user_product_order_count,
            'user_product_last_order': user_product_last_order
        }])
    
        try:
            probability = model1.predict_proba(input_df)[0][1]
            prediction = model1.predict(input_df)[0]

            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Reorder Probability", f"{probability:.2%}")
            res_col2.metric("Predicted Class", "Will Reorder" if prediction == 1 else "Won't Reorder")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(probability * 100, 1),
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "steelblue"},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightcoral'},
                        {'range': [50, 100], 'color': 'lightgreen'}
                    ],
                    'threshold': {
                        'line': {'color': "orange", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }, title={'text': "Reorder Probability Gauge"}
            ))
            fig_gauge.update_layout(height=300, margin={'t': 20, 'b': 20, 'l': 20, 'r': 20})
            st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")