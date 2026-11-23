import streamlit as st
import requests
import numpy as np

# st.markdown("""
# <style>

# .stApp {
#     background-color: #0e1117;
#     color: white;
# }

# section[data-testid="stSidebar"] {
#     background-color: #111827;
# }

# div.stButton > button {
#     background-color: #2463eb;
#     color: white;
#     border-radius: 8px;
#     height: 3em;
# }

# div.stButton > button:hover {
#     background-color: #1d4fd8;
#     color: white;
# }

# [data-testid="metric-container"] {
#     background-color: #1f2937;
#     border-radius: 10px;
#     padding: 15px;
# }

# </style>
# """, unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")

st.markdown(
"""
This dashboard interacts with the **Fraud Detection API** to evaluate whether a
transaction is potentially fraudulent.

Model: **Random Forest Classifier**
"""
)

st.divider()

col1, col2 = st.columns([2,1])

with col1:

    st.subheader("Transaction Features")

    st.markdown("Provide the 30 transaction features used by the model.")

    features = []

    cols = st.columns(5)

    for i in range(30):
        with cols[i % 5]:
            val = st.number_input(f"Feature {i+1}", value=0.0)
            features.append(val)

with col2:

    st.subheader("Prediction")

    if st.button("Analyze Transaction"):

        payload = {"features": features}

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            prob = result["fraud_probability"]

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected")
            else:
                st.success("✅ Transaction Appears Legitimate")

            st.metric("Fraud Probability", f"{prob:.2%}")

        else:
            st.warning("API error. Check FastAPI server.")

st.divider()

st.subheader("System Architecture")

st.code(
"""
User Transaction
        ↓
Streamlit Dashboard
        ↓
FastAPI API
        ↓
RandomForest Model
        ↓
Fraud Prediction
"""
)

st.caption("Fraud Detection System — AI/ML Project")