import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model & scaler ---
model  = joblib.load('models/xgb_churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Predictor")
st.markdown("Fill in customer details to predict if they will churn.")

# --- Sidebar inputs ---
st.sidebar.header("Customer Information")

tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly        = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total          = st.sidebar.number_input("Total Charges ($)", 0, 9000, 800)
contract       = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet       = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior         = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
tech_support   = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])

# --- Build input dataframe ---
def predict():
    data = {
        'tenure': [tenure],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total],
        'SeniorCitizen': [1 if senior == 'Yes' else 0],
        'PaperlessBilling': [1 if paperless == 'Yes' else 0],
        'Contract_One year': [1 if contract == 'One year' else 0],
        'Contract_Two year': [1 if contract == 'Two year' else 0],
        'InternetService_Fiber optic': [1 if internet == 'Fiber optic' else 0],
        'InternetService_No': [1 if internet == 'No' else 0],
        'TechSupport_No internet service': [1 if tech_support == 'No internet service' else 0],
        'TechSupport_Yes': [1 if tech_support == 'Yes' else 0],
    }
    input_df = pd.DataFrame(data)

    # Add missing columns with 0
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    # Scale numeric
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    return pred, prob

# --- Predict button ---
if st.button("🔮 Predict Churn", type="primary"):
    pred, prob = predict()
    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.error(f"⚠️ HIGH RISK: Customer likely to CHURN ({prob*100:.1f}% probability)")
        else:
            st.success(f"✅ LOW RISK: Customer likely to STAY ({(1-prob)*100:.1f}% probability)")
    with col2:
        st.metric("Churn Probability", f"{prob*100:.1f}%")