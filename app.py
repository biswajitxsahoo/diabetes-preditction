import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(
    page_title="Explainable Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Explainable AI (XAI) Diabetes Risk Screening Tool")
st.markdown("""
This screening tool uses a **Logistic Regression** model trained on balanced medical data (**SMOTE**) 
to prioritize high recall (catching early risk)[cite: 1, 2]. It uses **SHAP (SHapley Additive exPlanations)** 
to explain *why* each prediction was made[cite: 1, 2].
""")

# 2. Load Artifacts from models/ directory
@st.cache_resource
def load_artifacts():
    # Adjust paths if you moved them into models/, otherwise load from root
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        explainer = joblib.load('models/shap_explainer.pkl')
    except:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        explainer = joblib.load('shap_explainer.pkl')
    return model, scaler, explainer

model, scaler, explainer = load_artifacts()

THRESHOLD = 0.35

# 3. Sidebar Inputs
st.sidebar.header("📋 Patient Clinical Inputs")

def get_user_inputs():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose (mg/dL)", 40, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure (mmHg)", 40, 130, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 7, 99, 20)
    insulin = st.sidebar.slider("Insulin (mu U/ml)", 14, 846, 79)
    bmi = st.sidebar.slider("BMI", 18.0, 67.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.07, 2.42, 0.37)
    age = st.sidebar.slider("Age", 21, 81, 33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_inputs()

# 4. Display Inputs & Metrics
st.subheader("Patient Clinical Profile")
st.dataframe(input_df, use_container_width=True)

input_scaled = scaler.transform(input_df)
input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

risk_prob = model.predict_proba(input_scaled_df)[0][1]
is_diabetic = risk_prob >= THRESHOLD

st.divider()
col1, col2 = st.columns(2)

with col1:
    if is_diabetic:
        st.error("⚠️ **Screening Result: HIGH RISK (Diabetic)**")
        st.caption("Patient exceeds clinical screening threshold (>35% risk). Follow-up testing recommended.")
    else:
        st.success("✅ **Screening Result: LOW RISK (Non-Diabetic)**")
        st.caption("Patient is below clinical screening threshold (<35% risk).")

with col2:
    st.metric(label="Calculated Diabetes Risk Probability", value=f"{risk_prob * 100:.1f}%")

# 5. Dynamic SHAP Visualizations
st.divider()
st.subheader("💡 Model Interpretability & Feature Attribution (SHAP)")

tab1, tab2 = st.tabs(["🎯 Patient Local Explanation", "📊 Overall Dataset Feature Importance"])

with tab1:
    st.markdown("#### Patient-Level Risk Factors (Waterfall Plot)")
    st.caption("Red (+) increases risk towards Diabetic. Blue (-) decreases risk towards Healthy.")
    
    shap_vals = explainer(input_scaled_df)
    
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    shap.plots.waterfall(shap_vals[0], show=False)
    plt.tight_layout()
    st.pyplot(fig1)

with tab2:
    st.markdown("#### Global Feature Ranking Across Population")
    st.caption("Shows which clinical features have the strongest overall impact across all patient data.")
    
    # Calculate global mean absolute SHAP values for current model
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    # Render bar chart directly from local explainer values
    shap.plots.bar(shap_vals[0], show=False)
    plt.tight_layout()
    st.pyplot(fig2)