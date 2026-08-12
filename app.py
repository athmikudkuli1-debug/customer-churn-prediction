import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Customer churn analysis dashboard")

# Load dataset
file_path = "customer_churn.csv"

if not os.path.exists(file_path):
    st.error("customer_churn.csv not found. Please run 1_generate_data.py first.")
    st.stop()

df = pd.read_csv(file_path)

st.success(f"Dataset loaded successfully — {len(df)} customers")

# Dataset overview
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Churned Customers", int((df["Churn"] == "Yes").sum()))

with col3:
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    st.metric("Churn Rate", f"{churn_rate:.1f}%")

st.subheader("Customer Data")
st.dataframe(df, use_container_width=True)

# Churn chart
st.subheader("Churn Distribution")

churn_counts = df["Churn"].value_counts()
st.bar_chart(churn_counts)

# Filters
st.subheader("Filter Customers")

if "Contract" in df.columns:
    contracts = st.multiselect(
        "Select Contract Type",
        df["Contract"].unique(),
        default=list(df["Contract"].unique())
    )

    filtered_df = df[df["Contract"].isin(contracts)]

    st.write(f"Showing {len(filtered_df)} customers")
    st.dataframe(filtered_df, use_container_width=True)