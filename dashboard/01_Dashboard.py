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
from groq import Groq
from dotenv import load_dotenv

# Page configuration --------------------------------------------------------------
st.set_page_config(
    page_title="NextBuy Dashboard",
    page_icon="cart",
    layout="wide",
)

# Data and model loading -------------------------------------------------------------
load_dotenv()

USE_S3 = os.getenv('USE_S3', 'False').lower() == 'true'
S3_BUCKET = os.getenv('S3_BUCKET', '')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

@st.cache_data(show_spinner='Loading data...', ttl=3600)
def load_data():
    if USE_S3:
        storage_options = {
            'key': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'client_kwargs': {
                'region_name': os.getenv('AWS_REGION', 'eu-west-3')
            }
        }
        try:
            return pd.read_parquet(f's3://{S3_BUCKET}/cleaned_data.parquet', storage_options=storage_options)
        except Exception:
            return pd.read_csv(f's3://{S3_BUCKET}/cleaned_data.csv', storage_options=storage_options)
    parquet_path = os.path.join(DATA_DIR, 'cleaned_data.parquet')
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.read_csv(os.path.join(DATA_DIR, 'cleaned_data.csv'))

# Load data ------------------------------------------------------------------------
try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'cleaned_data.csv' is in the 'data' directory.")
    st.stop()

# Groq AI Global Analysis ------------------------------------------------------------------------
def stream_global_analysis(summary):
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.warning("Groq API key not found. Please set the GROQ_API_KEY environment variable to enable AI analysis.")
        return
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
            You are a senior data analyst presenting to a retail business audience.
            You have metrics from 7 different analyses of the same grocery dataset.
            Your task:
            1. Find 2-3 meaningful connections between the different analyses.
            2. Identify what they collectively reveal about customer behavior and product performance.
            3. Give 3 concrete, specific business recommendations backed by the numbers.

            Be specific and use the actual numbers. Write in flowing paragraphs, not bullet points. Make it engaging and insightful.
            Keep the total response under 300 words.

            {summary}
        """

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stop=None
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    
    except Exception as e:
        st.error(f"Error during AI analysis: {str(e)}")

# Sidebar filters -----------------------------------------------------------------------------

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

if st.sidebar.button('Global AI Analysis', use_container_width=True, type='primary'):
    st.session_state['global_analysis'] = True

st.sidebar.caption('14M rows - 5 datasets - 12 analytics - 2 ML models')

# Filter data based on selections --------------------------------------------------------------
filtered_df = df.copy()
if selected_department != 'All':
    filtered_df = filtered_df[filtered_df['department'] == selected_department]
if selected_aisle != 'All':
    filtered_df = filtered_df[filtered_df['aisle'] == selected_aisle]

df_reorder = filtered_df[filtered_df['is_first_order'] == 0]

# Pre-compute is_organic once — avoids recomputing inside AI block on every button click
filtered_df = filtered_df.copy()
filtered_df['is_organic'] = filtered_df['product_name'].str.contains('Organic', case=False, na=False).astype(int)

# Header ------------------------------------------------------------------------------
st.title("NextBuy Dashboard")
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
        height=max(400, top_n * 30),
        template='plotly_white'
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
    fig2.update_layout(
        title="Order Distribution by Day of Week and Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=500,
        template='plotly_white'
    )
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
            .sort_values('reorder_rate')
        )
        avg_reorder_rate = reorder_department['reorder_rate'].mean()

        fig3 = go.Figure(go.Bar(
            x=reorder_department['reorder_rate'],
            y=reorder_department['department'],
            orientation='h',
            marker_color='steelblue',
            text=reorder_department['reorder_rate'].apply(lambda x: f"{x:.2%}"),
            textposition='outside'
        ))

        fig3.add_vline(
            x=avg_reorder_rate,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Avg Reorder Rate: {avg_reorder_rate:.2%}",
            annotation_position="top right"
        )

        fig3.update_layout(
            title=f"Reorder Rate by Department",
            xaxis_title="Reorder Rate",
            yaxis_title="Department",
            height=max(400, len(reorder_department) * 30),
            template='plotly_white'
        )

        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# Global AI Analysis ----------------------------------------------------------------------
if st.session_state.get('global_analysis', False):
    st.session_state['global_analysis'] = False
    st.divider()
    st.subheader("Global AI Analysis")
    st.info("Generating insights from the data using Groq's AI model. This may take a moment...")

    with st.spinner("Collecting data from all analyses..."):
        # Q1 - Best Sellers
        top3_p = total_products.tail(3)

        # Q2 - Order Heatmap
        day_labels_global = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        peak_idx = pivot.values.argmax()
        pr, pc = divmod(peak_idx, pivot.shape[1])
        heatmap_peak_day = day_labels_global[pivot.index[pr]]
        heatmap_peak_hour = pivot.columns[pc]
        heatmap_peak_value = int(pivot.values[pr, pc])
        heatmap_summary = f"The peak order time is on {heatmap_peak_day} at {heatmap_peak_hour}:00 with {heatmap_peak_value} orders."

        # Q6 - Reorder Rate by Department
        top_department = reorder_department.nlargest(1, 'reorder_rate').iloc[0] if len(reorder_department) > 0 else None
        low_department = reorder_department.nsmallest(1, 'reorder_rate').iloc[0] if len(reorder_department) > 0 else None
        reorder_summary = f"The department with the highest reorder rate is {top_department['department']} at {top_department['reorder_rate']:.2%} and the lowest is {low_department['department']} at {low_department['reorder_rate']:.2%}." if top_department is not None and low_department is not None else "No reorder data available."

        # Q7 - Reorder vs day since prior order
        scatter_global = (df_reorder
            .groupby('days_since_prior_order')['reordered']
            .mean()
            .reset_index()
            .rename(columns={'reordered': 'reorder_rate'})
        )

        scatter_global = scatter_global[
            (scatter_global['days_since_prior_order'] > 0) &
            (scatter_global['days_since_prior_order'] < 30)  # Focus on the first 30 days for better insights  
        ]

        peak_scatter = scatter_global.loc[scatter_global['reorder_rate'].idxmax()]
        low_scatter = scatter_global.loc[scatter_global['reorder_rate'].idxmin()]

        # Q8 - Organic proportion
        organic_global = (filtered_df
            .groupby('department')['is_organic']
            .agg(['sum', 'count'])
        )
        organic_global['organic_rate'] = organic_global['sum'] / organic_global['count']
        top_organic = organic_global.nlargest(1, 'organic_rate')
        low_organic = organic_global.nsmallest(1, 'organic_rate')

        # Q9 - First cart item
        first_cart_global = (filtered_df[filtered_df['add_to_cart_order'] == 1]
            .groupby('product_name')['order_id']
            .count()
            .nlargest(3)
            .reset_index()
            .rename(columns={'order_id': 'count'})
        )

        # Q11 - Reorder by hour
        reorder_hour_global = (df_reorder
            .groupby('order_hour_of_day')['reordered']
            .mean()
            .reset_index()
            .rename(columns={'reordered': 'reorder_rate'})
        )

        global_summary = f"""
            GLOBAL NEXTBUY ANALYSIS SUMMARY
            Filter: department={selected_department}, aisle={selected_aisle}

            Q1 — BESTSELLERS:
            Top 3 products: {top3_p.iloc[2]['product_name']} ({top3_p.iloc[2]['orders']:,} orders), {top3_p.iloc[1]['product_name']} ({top3_p.iloc[1]['orders']:,} orders), {top3_p.iloc[0]['product_name']} ({top3_p.iloc[0]['orders']:,} orders).

            Q2 — ORDER TIMING:
            Peak ordering: {heatmap_peak_day} at {heatmap_peak_hour}:00 ({heatmap_peak_value:,} orders).

            Q6 — REORDER BY DEPARTMENT:
            {"Top reorder dept: " + top_department['department'] + " (" + f"{top_department['reorder_rate']:.1%}" + "). Lowest: " + low_department['department'] + " (" + f"{low_department['reorder_rate']:.1%}" + ")." if top_department is not None else "No reorder data."}
            Average reorder rate: {reorder_rate:.1%}.

            Q7 — REORDER VS DAYS SINCE LAST ORDER:
            Peak reorder rate {peak_scatter['reorder_rate']:.1%} at day {int(peak_scatter['days_since_prior_order'])}. Drops to {low_scatter['reorder_rate']:.1%} by day {int(low_scatter['days_since_prior_order'])}.

            Q8 — ORGANIC PROPORTION:
            Most organic department: {top_organic.index[0]} ({top_organic['organic_rate'].iloc[0]:.1%}). Least organic: {low_organic.index[0]} ({low_organic['organic_rate'].iloc[0]:.1%}).

            Q9 — FIRST CART ITEM:
            Most common first item: {first_cart_global.iloc[0]['product_name']} ({first_cart_global.iloc[0]['count']:,} times). Second: {first_cart_global.iloc[1]['product_name']} ({first_cart_global.iloc[1]['count']:,} times).

            Q11 — REORDER BY HOUR:
            Peak reorder hour: {int(reorder_hour_global.iloc[0]['order_hour_of_day'])}:00 ({reorder_hour_global.iloc[0]['reorder_rate']:.1%}). Lowest: {int(reorder_hour_global.iloc[-1]['order_hour_of_day'])}:00 ({reorder_hour_global.iloc[-1]['reorder_rate']:.1%}).

        """
    st.write_stream(stream_global_analysis(global_summary))