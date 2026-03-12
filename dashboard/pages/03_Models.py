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
        with fs.open(f's3://{S3_BUCKET}/{s3_key}', 'rb') as f:
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
        add_to_cart_order = st.slider(
            "Position in cart", 1, 30, 3, key="m1_cart_pos",
            help="Where this product appears in the cart (1 = first item added)")
        order_number = st.slider(
            "Customer order number", 1, 100, 10, key="m1_order_num",
            help="How many orders this customer has placed in total")
        days_since_prior_order = st.slider(
            "Days since last order", 0, 30, 7, key="m1_days")
    with col2:
        order_dow = st.selectbox(
            "Day of week",
            options=[0,1,2,3,4,5,6],
            format_func=lambda x: ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'][x],
            key="m1_dow")
        order_hour_of_day = st.slider("Hour of day", 0, 23, 10, key="m1_hour")

    # Engineered features fixed to dataset averages — not meaningful to enter manually
    ENGINEERED_DEFAULTS = {
        'aisle_id':                     24,
        'department_id':                4,
        'user_id':                      1,
        'product_id':                   196,
        'user_total_orders':            17,
        'user_avg_days_between_orders': 11.1,
        'product_order_count':          1452,
        'product_avg_cart_position':    7.5,
        'product_reorder_rate':         0.60,
        'user_product_order_count':     2,
        'user_product_last_order':      5,
    }

    if st.button("Predict Reorder Probability", use_container_width=True, type="primary", key="btn_m1"):
        input_df = pd.DataFrame([{
            'add_to_cart_order':            add_to_cart_order,
            'aisle_id':                     ENGINEERED_DEFAULTS['aisle_id'],
            'department_id':                ENGINEERED_DEFAULTS['department_id'],
            'order_number':                 order_number,
            'order_dow':                    order_dow,
            'order_hour_of_day':            order_hour_of_day,
            'days_since_prior_order':       days_since_prior_order,
            'user_id':                      ENGINEERED_DEFAULTS['user_id'],
            'product_id':                   ENGINEERED_DEFAULTS['product_id'],
            'user_total_orders':            ENGINEERED_DEFAULTS['user_total_orders'],
            'user_avg_days_between_orders': ENGINEERED_DEFAULTS['user_avg_days_between_orders'],
            'product_order_count':          ENGINEERED_DEFAULTS['product_order_count'],
            'product_avg_cart_position':    ENGINEERED_DEFAULTS['product_avg_cart_position'],
            'product_reorder_rate':         ENGINEERED_DEFAULTS['product_reorder_rate'],
            'user_product_order_count':     ENGINEERED_DEFAULTS['user_product_order_count'],
            'user_product_last_order':      ENGINEERED_DEFAULTS['user_product_last_order'],
        }])
    
        try:
            prediction = model1.predict(input_df)[0]

            if prediction == 1:
                st.success("Yes — this customer will likely reorder this product.")
            else:
                st.error("No — this customer will likely not reorder this product.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.divider()

# Model 2: Cart Size Regressor ------------------------------------------------------
st.header('Cart Size Regressor')
st.caption('Random Forest regressor · Predicts how many items a customer will add to their next order')

if model2 is None:
    st.warning("Cart Size Regressor model not found. Please ensure 'model2.joblib' is available.")
else:
    with st.expander("How this model works", expanded=False):
        st.markdown("""
        **Algorithm:** Random Forest Regressor (tuned with GridSearchCV, cv=3)

        **Pipeline steps:**
        1. `StandardScaler` — normalizes all features to mean=0, std=1
        2. `RandomForestRegressor` — predicts number of items in the next order

        **Leakage prevention:** all user history features use `.shift(1).expanding().mean()`
        — each order only sees data from the user's *previous* orders, never the current one.

        **Features:**
        - **Temporal:** day of week, hour of day, days since last order, order number
        - **User basket history:** avg basket size (all past orders), avg basket size (last 3 orders),
          standard deviation of past baskets, last basket size, historical reorder rate, total orders so far

        **Output:** predicted number of products in the next order.
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Order context**")
        order_dow_m2 = st.selectbox(
            "Day of week",
            options=[0,1,2,3,4,5,6],
            format_func=lambda x: ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'][x],
            key="m2_dow")
        order_hour_of_day_m2 = st.slider("Hour of day", 0, 23, 10, key="m2_hour")
        days_since_prior_order_m2 = st.slider("Days since last order", 0, 30, 7, key="m2_days")
        order_number_m2 = st.slider(
            "Order number", 1, 100, 10, key="m2_order_num",
            help="How many orders this customer has placed in total")

    with col2:
        st.markdown("**User basket history (engineered features)**")
        user_avg_basket = st.slider(
            "Avg basket — all past orders", 1.0, 40.0, 10.0, step=0.5, key="m2_avg",
            help="Average number of items across all of this user's previous orders")
        user_avg_basket3 = st.slider(
            "Avg basket — last 3 orders", 1.0, 40.0, 10.0, step=0.5, key="m2_avg3",
            help="Average number of items across the user's 3 most recent orders")
        user_std_basket = st.slider(
            "Std of past basket sizes", 0.0, 20.0, 3.0, step=0.5, key="m2_std",
            help="How much this user's basket size varies order to order")
        user_last_basket = st.slider(
            "Last basket size", 1, 60, 10, key="m2_last",
            help="Number of items in this user's most recent order")
        user_reorder_rate = st.slider(
            "User historical reorder rate", 0.0, 1.0, 0.6, step=0.01, key="m2_rr",
            help="Proportion of products this user typically reorders")
        user_n_orders_so_far = st.slider(
            "Total orders so far", 1, 100, 10, key="m2_norders",
            help="How many orders this user has placed before this one")

    if st.button("Predict Cart Size", use_container_width=True, type="primary", key="btn_m2"):
        input_m2 = pd.DataFrame([{
            'order_dow':              order_dow_m2,
            'order_hour_of_day':      order_hour_of_day_m2,
            'days_since_prior_order': days_since_prior_order_m2,
            'order_number':           order_number_m2,
            'user_avg_basket':        user_avg_basket,
            'user_avg_basket3':       user_avg_basket3,
            'user_std_basket':        user_std_basket,
            'user_last_basket':       user_last_basket,
            'user_reorder_rate':      user_reorder_rate,
            'user_n_orders_so_far':   user_n_orders_so_far,
        }])

        try:
            predicted   = max(1, round(model2.predict(input_m2)[0]))
            dataset_avg = 10.1

            st.metric("Predicted Cart Size", f"{predicted} items")

            fig_bar = go.Figure(go.Bar(
                x=["Dataset Average", "Your Prediction"],
                y=[dataset_avg, predicted],
                marker_color=["lightgray", "steelblue"],
                text=[f"{dataset_avg:.1f}", str(predicted)],
                textposition="outside"
            ))
            fig_bar.update_layout(
                title="Predicted Cart Size vs Dataset Average",
                yaxis_title="Number of items",
                yaxis=dict(range=[0, max(dataset_avg, predicted) * 1.3]),
                height=320,
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            if predicted > dataset_avg * 1.2:
                st.info(f"Large shop — {predicted} items is well above the dataset average of {dataset_avg:.0f}.")
            elif predicted < dataset_avg * 0.8:
                st.info(f"Small top-up shop — {predicted} items is below the dataset average of {dataset_avg:.0f}.")
            else:
                st.info(f"Typical shop — close to the dataset average of {dataset_avg:.0f} items.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")