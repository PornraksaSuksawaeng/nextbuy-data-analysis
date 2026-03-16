"""
NextBuy - Analysis Page
Deeper EDA charts not shown on the main page.
Owner: Pornraksa Suksawaeng
Run command: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page configuration --------------------------------------------------------------
st.set_page_config(
    page_title="NextBuy Analysis",
    page_icon="analysis",
    layout="wide"
)

# Data loading -------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

USE_S3 = os.getenv('USE_S3', 'False').lower() == 'true'
S3_BUCKET = os.getenv('S3_BUCKET', '')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')

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
try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'cleaned_data.csv' is in the 'data' directory.")
    st.stop()

# Sidebar filters -----------------------------------------------------------------------------
st.sidebar.caption("EPITECH B1 - Data Science Project - 2026")
st.sidebar.divider()
st.sidebar.subheader("Filters")

departments = ['All'] + sorted(df['department'].dropna().unique().tolist())
selected_dept = st.sidebar.selectbox("Department", departments)

if selected_dept != "All":
    aisle_pool = df[df["department"] == selected_dept]["aisle"].dropna().unique().tolist()
else:
    aisle_pool = df["aisle"].dropna().unique().tolist()

aisles = ['All'] + sorted(aisle_pool)
selected_aisle = st.sidebar.selectbox("Aisle", aisles)

st.sidebar.divider()
st.sidebar.caption("14M rows · 5 datasets · 12 analyses")

# Filter data based on selections -------------------------------------------------------------
filtered = df.copy()
if selected_dept != "All":
    filtered = filtered[filtered["department"] == selected_dept]
if selected_aisle != "All":
    filtered = filtered[filtered["aisle"] == selected_aisle]

# Exclude first orders for reorder rate analyses
df_reorder = filtered[filtered["is_first_order"] == 0]

# Header ------------------------------------------------------------------------
st.title("Deeper Analysis")
st.caption("EDA charts not shown on the main page")
st.divider()

# Q7 - Reorder rate vs days since prior order -------------------------------------------------------------
st.subheader("Reorder Rate vs Days Since Prior Order")
st.caption("Uses filtered data with first orders excluded since they have no prior order by definition.")

if len(df_reorder) == 0:
    st.warning("No data available for this selection. Please adjust the filters.")
else:
    scatter_data = (df_reorder
        .groupby("days_since_prior_order")["reordered"]
        .mean()
        .reset_index()
        .rename(columns={"reordered": "reorder_rate"})              
    )

    scatter_data = scatter_data[
        # NOTE: Exclude day 0 because the context of 'days since prior order' = 0 is ambiguous (could be first order or back-to-back orders) and it skews the analysis
        (scatter_data["days_since_prior_order"] > 0) &
        # NOTE: Exclude days > 30 because it regroups all orders with 30+ days into a single category, which skews the analysis
        (scatter_data["days_since_prior_order"] < 30)
    ]

    z = np.polyfit(scatter_data["days_since_prior_order"], scatter_data["reorder_rate"], 1)
    p = np.poly1d(z)

    x_line = np.linspace(
        scatter_data["days_since_prior_order"].min(),
        scatter_data["days_since_prior_order"].max(), 100
    )

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=scatter_data["days_since_prior_order"],
        y=scatter_data["reorder_rate"],
        mode='markers',
        name='Reorder Rate',
        marker=dict(size=10, color='steelblue', opacity=0.6),
        hovertemplate='Days since prior order: %{x}<br>Reorder rate: %{y:.1%}<extra></extra>'
    ))

    fig7.add_trace(go.Scatter(
        x=x_line, y=p(x_line),
        mode='lines',
        name='Trend Line',
        line=dict(color='orange', width=2, dash='dash')
    ))

    fig7.update_layout(
        title="Reorder Rate vs Days Since Prior Order",
        xaxis_title="Days Since Prior Order",
        yaxis_title="Reorder Rate",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", y=1.1, x = 0.5, xanchor='center')
    )

    st.plotly_chart(fig7, use_container_width=True)

    peak_day = scatter_data.loc[scatter_data["reorder_rate"].idxmax(),]
    low_day = scatter_data.loc[scatter_data["reorder_rate"].idxmin()]
    c1, c2 = st.columns(2)
    c1.metric("Peak Reorder Rate", f"{peak_day['reorder_rate']:.1%}", f"at day {int(peak_day['days_since_prior_order'])}")
    c2.metric("Lowest Reorder Rate", f"{low_day['reorder_rate']:.1%}", f"at day {int(low_day['days_since_prior_order'])}")
    
    st.divider()

# Q8 - Organic proportion by department -------------------------------------------------------------
st.subheader("Organic Proportion by Department")
st.caption("Shows the proportion of organic products ordered in each department.")

# is_organic computed once on full df, not on filtered — avoids recompute on every filter change
if 'is_organic' not in df.columns:
    df['is_organic'] = df['product_name'].str.contains('Organic', case=False, na=False).astype(int)
filtered['is_organic'] = filtered['is_organic'] if 'is_organic' in filtered.columns else filtered['product_name'].str.contains('Organic', case=False, na=False).astype(int)

organic_proportion = (filtered
    .groupby('department')['is_organic']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'organic_count', 'count': 'total_count'})
)

organic_proportion['organic_pct'] = organic_proportion['organic_count'] / organic_proportion['total_count']
organic_proportion['non_organic_pct'] = 1 - organic_proportion['organic_pct']
organic_proportion = organic_proportion.sort_values('organic_pct')

fig8 = go.Figure()
fig8.add_trace(go.Bar(
    y=organic_proportion.index,
    x=organic_proportion['organic_pct'],
    name='Organic',
    orientation='h',
    marker_color='steelblue',
    text=organic_proportion['organic_pct'].apply(lambda x: f"{x:.1%}"),
    textposition='inside'
))

fig8.add_trace(go.Bar(
    y=organic_proportion.index,
    x=organic_proportion['non_organic_pct'],
    name='Non-organic',
    orientation='h',
    marker_color='lightgray'
))

fig8.update_layout(
    barmode='stack',
    xaxis=dict(tickformat='.0%', title='Proportion'),
    height=460,
    # legend white transparent background and positioned above and outside the chart to avoid overlapping with the bars
    legend=dict(orientation='h', y = 1.10, x = 0.5, xanchor='center') 
)

st.plotly_chart(fig8, use_container_width=True)

top5_organic = organic_proportion.sort_values('organic_pct', ascending=False).head(5)
top5_organic.columns = ['Organic count', 'Total count', 'Organic share', 'Non-organic share']
top5_organic['Organic share'] = top5_organic['Organic share'].map("{:.1%}".format)

with st.expander("Top 5 departments by organic share"):
    st.dataframe(top5_organic, use_container_width=True, hide_index=True)

st.divider()

# Q9 - First item added to cart -------------------------------------------------------------
st.subheader("First Item Added to Cart")
st.caption("Analyzes which products are most commonly the first item added to the cart in an order.")

top_n_q9 = st.slider("Number of products to display", min_value=5, max_value=25, value=15, key="q9_n")
first_items = (filtered[filtered["add_to_cart_order"] == 1]
    .groupby("product_name")["order_id"]
    .count()
    .nlargest(top_n_q9)
    .reset_index()
    .rename(columns={"order_id": "first_item_count"})
    .sort_values("first_item_count")
)

fig9 = go.Figure(go.Bar(
    x=first_items["first_item_count"],
    y=first_items["product_name"],
    orientation="h",
    marker_color="steelblue",
    text=first_items["first_item_count"],
    textposition="outside"
))

fig9.update_layout(
    title=f"Top {top_n_q9} Products Added First to Cart",
    xaxis_title="Number of times added first",
    height=max(400, top_n_q9 * 30)
)

st.plotly_chart(fig9, use_container_width=True)

st.divider()

# Q11 - Reorder rate by hour of day -------------------------------------------------------------
st.subheader("Reorder Rate by Hour of Day")
st.caption("Analyzes how the reorder rate varies by the hour of the day when the order was placed, using only reorder instances (first orders excluded).")

if len(df_reorder) == 0:
    st.warning("No data available for this selection. Please adjust the filters.")
else:
    reorder_by_hour = (df_reorder
        .groupby("order_hour_of_day")["reordered"]
        .mean()
        .reset_index()
        .rename(columns={"reordered": "reorder_rate"})
    )

    fig11 = go.Figure()
    fig11.add_trace(go.Scatter(
        x=reorder_by_hour["order_hour_of_day"],
        y=reorder_by_hour["reorder_rate"],
        mode='lines+markers',
        name='Reorder Rate',
        line=dict(color='steelblue', width=2),
        marker=dict(size=8, color='steelblue'),
        hovertemplate='Hour: %{x}:00<br>Reorder rate: %{y:.1%}<extra></extra>'
    ))

    fig11.update_layout(
        title="Reorder Rate by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Average Reorder Rate",
        yaxis=dict(tickformat=".0%"),
        height=400,
    )

    st.plotly_chart(fig11, use_container_width=True)

    peak_hour = reorder_by_hour.loc[reorder_by_hour["reorder_rate"].idxmax(),]
    low_hour = reorder_by_hour.loc[reorder_by_hour["reorder_rate"].idxmin(),]
    c1, c2 = st.columns(2)
    c1.metric("Peak reorder hour", f"{peak_hour['reorder_rate']:.1%}", f"at {int(peak_hour['order_hour_of_day'])}h")
    c2.metric("Lowest reorder hour", f"{low_hour['reorder_rate']:.1%}", f"at {int(low_hour['order_hour_of_day'])}h")