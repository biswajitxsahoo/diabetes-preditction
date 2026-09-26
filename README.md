# 🩺 Explainable AI (XAI) Diabetes Risk Screening Tool

An interactive machine learning and clinical screening application designed to assess diabetes risk with a focus on **high recall** for early detection. Powered by **Logistic Regression**, **SMOTE oversampling**, and **SHAP (SHapley Additive exPlanations)** for full model interpretability.

---

## 📌 Project Overview & Medical Rationale

Early detection of diabetes is critical to preventing long-term complications such as cardiovascular disease, neuropathy, and kidney damage. Traditional machine learning metrics often prioritize overall accuracy, which can lead to high **False Negative** rates—a dangerous scenario in medical screening where a diabetic patient is mistakenly marked healthy.

### Key Objectives:
- **Prioritize Clinical Recall**: Lowered the decision threshold to **35%** to achieve an **88.89% Recall rate**, ensuring at-risk patients receive necessary follow-up care.
- **Explainable AI (XAI)**: Integrated game-theoretic SHAP values to explain every prediction at both global (dataset) and local (individual patient) levels.
- **Interactive UI**: Built a user-friendly Streamlit dashboard for real-time risk assessment and feature attribution visualization.

---

## 📊 Dataset & Preprocessing Pipeline

This project uses the **Pima Indians Diabetes Dataset** containing clinical measurements from 768 female patients.

1. **Handling Missing Data**: Biologically impossible zero values in features like `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were converted to `NaN` and imputed using training set column means.
2. **Feature Scaling**: Scaled all numerical inputs using `MinMaxScaler` to prevent features with larger ranges from biasing model coefficients.
3. **Class Balancing**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data to address class imbalance without introducing data leakage.

---

## 🤖 Model Performance & Evaluation

Three classifiers were trained on SMOTE-balanced data and evaluated at a **35% decision threshold**:

| Model | Accuracy | Recall (Priority) | F1-Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** 🏆 | **69.48%** | **88.89%** | **0.6713** |
| Random Forest | 72.08% | 87.04% | 0.6861 |
| XGBoost | 74.03% | 72.22% | 0.6610 |

> **Selection Context**: Logistic Regression was selected as the core production model because it achieved the highest Recall (88.89%), missing only 6 positive cases in test evaluation.

---

## 💡 Model Interpretability (SHAP)

To bridge the gap between machine learning outputs and clinical trust, we utilize **SHAP (SHapley Additive exPlanations)**:
- **Global Importance**: Highlights overall key drivers across the population (e.g., Glucose and BMI).
- **Local Waterfall Plots**: Explains individual patient predictions step-by-step, showing which clinical values increased (red) or decreased (blue) their risk score.

---

## 📁 Repository Structure

```text
├── data/
│   └── diabetes.csv             # Pima Indians Diabetes dataset
├── explore.py                   # Data exploration & summary statistics
├── preprocess.py                # Zero handling, mean imputation & MinMaxScaler
├── smote.py                     # Synthetic oversampling for class balance
├── train_eval.py                # Model training, thresholding & recall evaluation
├── explain_shap.py              # SHAP explainer initialization & plot generation
├── app.py                       # Main Streamlit web application
├── best_model.pkl               # Production Logistic Regression model
├── scaler.pkl                   # Trained MinMaxScaler object
├── shap_explainer.pkl           # Saved SHAP LinearExplainer
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started Locally

### 1. Clone the Repository

```bash
git clone [https://github.com/biswajitxsahoo/diabetes-preditction](https://github.com/biswajitxsahoo/diabetes-preditction)
cd diabetes-preditction
```

### 2. Set Up Virtual Environment & Install Dependencies

```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

## Built With

- Python 3.10+
- Streamlit: Web dashboard interface
- Scikit-Learn: Data preprocessing and Logistic Regression
- Imbalanced-Learn (SMOTE): Class balancing
- SHAP: Model interpretability
- Matplotlib & Pandas: Data manipulation and visualization