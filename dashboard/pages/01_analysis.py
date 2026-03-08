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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')

@st.cache_data(show_spinner='Loading data...')
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'cleaned_data.csv'))
try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'cleaned_data.csv' is in the 'data' directory.")
    st.stop()

# Sidebar filters -----------------------------------------------------------------------------
st.title("NextBuy Analysis")
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
st.subheader("Q7: Reorder Rate vs Days Since Prior Order")
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
        legend=dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig7, use_container_width=True)

    peak_day = scatter_data.loc[scatter_data["reorder_rate"].idxmax(),]
    low_day = scatter_data.loc[scatter_data["reorder_rate"].idxmin()]
    c1, c2 = st.columns(2)
    c1.metric("Peak Reorder Rate", f"{peak_day['reorder_rate']:.1%}", f"at day {int(peak_day['days_since_prior_order'])}")
    c2.metric("Lowest Reorder Rate", f"{low_day['reorder_rate']:.1%}", f"at day {int(low_day['days_since_prior_order'])}")

    with st.expander("Analysis"):
        st.markdown("""
        The reorder rate is highest around 10 days since the prior order, suggesting that customers are most likely to reorder around this time frame.
        This could be due to typical consumption patterns or the time it takes for customers to realize they need to restock.
        The reorder rate drops significantly after 20 days, indicating that customers are less likely to reorder if too much time has passed since their last purchase.
        This insight can help inform marketing strategies, such as sending reminder emails or promotions around the 10-day mark to encourage reorders.
        """)
    
    st.divider()
