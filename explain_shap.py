import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Load trained model, background train data, and test data
model = joblib.load('best_model.pkl')
X_train_res = joblib.load('X_train_res.pkl')
X_test = joblib.load('X_test_scaled.pkl')

print("--- GENERATING SHAP EXPLANATIONS ---\n")

# 2. Initialize SHAP Explainer for Linear / Logistic Regression model
# We use LinearExplainer for Logistic Regression with training data as background
explainer = shap.LinearExplainer(model, X_train_res)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# 3. Global Plot 1: Feature Importance Summary Bar Plot
plt.figure(figsize=(8, 5))
shap.plots.bar(shap_values, show=False)
plt.title("Global Feature Importance (SHAP)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_global_bar.png", dpi=300)
plt.close()
print("Saved 'shap_global_bar.png' (Global Feature Ranking)")

# 4. Local Plot 2: Patient-Level Waterfall Plot for First Test Patient
patient_idx = 0
plt.figure(figsize=(9, 5))
shap.plots.waterfall(shap_values[patient_idx], show=False)
plt.title(f"Local Patient #{patient_idx+1} Risk Explanation", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_local_waterfall.png", dpi=300)
plt.close()
print(f"Saved 'shap_local_waterfall.png' (Detailed breakdown for Patient #{patient_idx+1})")

# 5. Save explainer object for Streamlit UI (Stage 7)
joblib.dump(explainer, 'shap_explainer.pkl')
print("\n--- STAGE 6 COMPLETE ---")
print("Saved shap_explainer.pkl successfully!")