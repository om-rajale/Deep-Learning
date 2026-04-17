import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras

# Page Config for a better look
st.set_page_config(page_title="ChurnAI Analytics", layout="wide")

# Custom CSS for Innovative Design (Glassmorphism effect)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 20px; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; }
    .reportview-container .main .block-container{ padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('churn_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


model, scaler = load_assets()

# --- SIDEBAR FOR INPUTS ---
st.sidebar.header("📡 Customer Profile")
with st.sidebar:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method",
                           ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# --- MAIN DASHBOARD ---
st.title("🚀 ChurnAI: Predictive Intelligence")
st.write("Analyze customer behavior and predict retention probability.")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("💳 Charges")
    # THE FIX: format="%.2f" ensures dots are used for decimals
    monthly_charges = st.number_input("Monthly Charges", value=50.00, format="%.2f")
    total_charges = st.number_input("Total Charges", value=500.00, format="%.2f")

with col2:
    st.subheader("🌐 Services")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No"])
    backup = st.selectbox("Online Backup", ["Yes", "No"])

with col3:
    st.subheader("📺 Entertainment")
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])


# Preprocessing Function
def get_prediction_data():
    binary_map = {"Yes": 1, "No": 0}
    data = {
        'gender': 1 if gender == "Female" else 0,
        'SeniorCitizen': senior,
        'Partner': binary_map[partner],
        'Dependents': binary_map[dependents],
        'tenure': tenure,
        'PhoneService': 1,  # Defaulted based on typical use case
        'MultipleLines': 0,
        'OnlineSecurity': binary_map[security],
        'OnlineBackup': binary_map[backup],
        'DeviceProtection': 0,
        'TechSupport': 0,
        'StreamingTV': binary_map[streaming_tv],
        'StreamingMovies': binary_map[streaming_movies],
        'PaperlessBilling': 1,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    # One-hot encoding logic to match 26 features
    data['InternetService_DSL'] = 1 if internet == "DSL" else 0
    data['InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0
    data['InternetService_No'] = 1 if internet == "No" else 0
    data['Contract_Month-to-month'] = 1 if contract == "Month-to-month" else 0
    data['Contract_One year'] = 1 if contract == "One year" else 0
    data['Contract_Two year'] = 1 if contract == "Two year" else 0
    data['PaymentMethod_Bank transfer (automatic)'] = 1 if "Bank" in payment else 0
    data['PaymentMethod_Credit card (automatic)'] = 1 if "Credit" in payment else 0
    data['PaymentMethod_Electronic check'] = 1 if "Electronic" in payment else 0
    data['PaymentMethod_Mailed check'] = 1 if "Mailed" in payment else 0

    df = pd.DataFrame([data])
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df


# Final Result
if st.button("RUN ANALYTICS"):
    input_df = get_prediction_data()
    prediction = model.predict(input_df)
    prob = float(prediction[0][0])

    st.divider()
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric("Churn Probability", f"{prob * 100:.1f}%")
        if prob > 0.5:
            st.error("⚠️ HIGH RISK: This customer is likely to churn.")
        else:
            st.success("✅ LOW RISK: This customer is likely to stay.")

    with res_col2:
        # Progress bar as a visual gauge
        st.write("Risk Level")
        st.progress(prob)