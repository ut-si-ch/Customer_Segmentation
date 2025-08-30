import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

#kmeans, scaler, pivot, product_similarity = load_models_and_data()
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
    
# Load pivot table for product similarity
pivot = pd.read_pickle("product_customer_pivot.pkl")
product_similarity = pd.read_pickle("product_similarity.pkl")

st.set_page_config(page_title="🛒 Shopper Spectrum App", layout="centered")
st.title("🛍️ Shopper Spectrum: E-Commerce Intelligence")
st.markdown("---")

# Tabs for App Modules
app_mode = st.sidebar.selectbox("Choose Module", ["📦 Product Recommendation", "👥 Customer Segmentation"])
# ====================================
# Module 1: Product Recommendation
# ====================================
if app_mode == "📦 Product Recommendation":
    st.subheader("🔍 Recommend Similar Products")

    product_input = st.text_input("Enter a Product Name:")

    if st.button("Get Recommendations"):
        if product_input in pivot.index:
            product_vector = product_similarity.loc[product_input].sort_values(ascending=False)[1:6]
            st.success("Recommended Products:")
            for i, product in enumerate(product_vector.index):
                st.markdown(f"{i+1}. **{product}**")
        else:
            st.error("Product not found. Please try another product name.")
# ====================================
# Module 2: Customer Segmentation
# ====================================
elif app_mode == "👥 Customer Segmentation":
    st.subheader("🧮 Predict Customer Segment")

    recency = st.number_input("Recency (days since last purchase):", min_value=0)
    frequency = st.number_input("Frequency (total transactions):", min_value=0)
    monetary = st.number_input("Monetary (total spend):", min_value=0.0)

    if st.button("Predict Cluster"):
        input_df = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetory": [monetary]})
        scaled_input = scaler.transform(input_df)
        cluster_label = kmeans.predict(scaled_input)[0]

        segment_map = {
            0: "High-Value",
            1: "Regular",
            2: "Occasional",
            3: "At-Risk"
        }

        segment_name = segment_map.get(cluster_label, "Unknown")
        st.success(f"Customer Segment: **{segment_name}**")

st.markdown("---")
st.caption("📊 Powered by RFM Clustering & Collaborative Filtering")
