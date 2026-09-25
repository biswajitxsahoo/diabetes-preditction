import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="Explainable Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Explainable AI (XAI) Diabetes Risk Screening Tool")
st.markdown("""
This screening tool uses a **Logistic Regression** model trained on balanced medical data (**SMOTE**) 
to prioritize high recall (catching early risk). It uses **SHAP (SHapley Additive exPlanations)** 
to explain *why* each prediction was made.
""")

# 2. Load Pre-trained Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    return model, scaler, explainer

model, scaler, explainer = load_artifacts()

# Clinical Screening Threshold
THRESHOLD = 0.35

# 3. Sidebar Input Form
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

# 4. Display Patient Features
st.subheader("Patient Clinical Profile")
st.dataframe(input_df, use_container_width=True)

# 5. Preprocess Input & Generate Prediction
input_scaled = scaler.transform(input_df)
input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

# Get risk probability
risk_prob = model.predict_proba(input_scaled_df)[0][1]
is_diabetic = risk_prob >= THRESHOLD

# Display Results
st.divider()
col1, col2 = st.columns(2)

with col1:
    if is_diabetic:
        st.error(f"⚠️ **Screening Result: HIGH RISK (Diabetic)**")
        st.caption("Patient exceeds clinical screening threshold (>35% risk). Follow-up testing recommended.")
    else:
        st.success(f"✅ **Screening Result: LOW RISK (Non-Diabetic)**")
        st.caption("Patient is below clinical screening threshold (<35% risk).")

with col2:
    st.metric(label="Calculated Diabetes Risk Probability", value=f"{risk_prob * 100:.1f}%")

# 6. Live SHAP Explanation Section
st.divider()
st.subheader("💡 Live SHAP Explanation (Why did the model predict this?)")
st.markdown("""
* **Red bars (+)** push patient risk **higher** towards Diabetic.
* **Blue bars (-)** pull patient risk **lower** towards Healthy.
""")

# Calculate SHAP values for current input
shap_vals = explainer(input_scaled_df)

# Render Waterfall Plot
fig, ax = plt.subplots(figsize=(9, 4.5))
shap.plots.waterfall(shap_vals[0], show=False)
plt.title("Patient-Level Risk Feature Attribution", fontsize=11, fontweight='bold')
plt.tight_layout()

st.pyplot(fig)