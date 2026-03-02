import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# ----------------------------------------------------
# LOAD MODEL & DATA
# ----------------------------------------------------
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("mall_customers.csv")

# Prepare data
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_scaled = scaler.transform(X)
df["Cluster"] = model.predict(X_scaled)

# ----------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Dashboard Overview",
    "Cluster Visualization",
    "Predict Customer"
])

# ====================================================
# PAGE 1 — DASHBOARD OVERVIEW
# ====================================================
if page == "Dashboard Overview":

    st.title("Customer Segmentation Dashboard")
    st.markdown("### Business Intelligence Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Number of Clusters", df["Cluster"].nunique())
    col3.metric("Average Income (k$)", round(df["Annual Income (k$)"].mean(), 2))

    st.markdown("---")

    st.subheader("Cluster Distribution")

    cluster_counts = df["Cluster"].value_counts().sort_index()

    fig_bar = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Number of Customers'},
        title="Customers per Cluster"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# ====================================================
# PAGE 2 — CLUSTER VISUALIZATION
# ====================================================
elif page == "Cluster Visualization":

    st.title("Cluster Visualization")

    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=df["Cluster"].astype(str),
        title="Customer Segments",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cluster Insights")

    st.write("""
    - Cluster 0 → Premium Customers (High income & high spending)
    - Cluster 1 → Budget Customers (Low income & low spending)
    - Cluster 2 → Careful Customers (High income & low spending)
    - Cluster 3 → Impulsive Customers (Low income & high spending)
    - Cluster 4 → Average Customers
    """)

# ====================================================
# PAGE 3 — PREDICTION PAGE
# ====================================================
elif page == "Predict Customer":

    st.title("Predict Customer Segment")

    income = st.slider("Annual Income (k$)", 0, 200, 50)
    spending = st.slider("Spending Score (1-100)", 0, 100, 50)

    if st.button("Predict Segment"):

        input_data = np.array([[income, spending]])
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]

        st.success(f"Predicted Cluster: {cluster}")

        if cluster == 0:
            st.info("Premium Customer")
        elif cluster == 1:
            st.info("Budget Customer")
        elif cluster == 2:
            st.info("Careful Customer")
        elif cluster == 3:
            st.info("Impulsive Customer")
        else:
            st.info("Average Customer")

        # Show highlighted visualization
        fig2 = px.scatter(
            df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color=df["Cluster"].astype(str),
            title="Customer Position in Cluster"
        )

        fig2.add_scatter(
            x=[income],
            y=[spending],
            mode="markers",
            marker=dict(size=15),
            name="New Customer"
        )

        st.plotly_chart(fig2, use_container_width=True)
