import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# 1. Load balanced training data & original test data
X_train_res = joblib.load('X_train_res.pkl')
y_train_res = joblib.load('y_train_res.pkl')
X_test = joblib.load('X_test_scaled.pkl')
y_test = joblib.load('y_test.pkl')

# 2. Define the 3 models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42),
    "XGBoost": XGBClassifier(max_depth=4, random_state=42, eval_metric='logloss')
}

# Clinical Threshold for screening (Flag as Diabetic if risk > 35%)
THRESHOLD = 0.35

results = {}
best_model = None
best_recall = 0
best_model_name = ""

print(f"--- MODEL TRAINING & EVALUATION (Clinical Threshold: {THRESHOLD*100:.0f}%) ---\n")

for name, model in models.items():
    # Train model
    model.fit(X_train_res, y_train_res)
    
    # Get prediction probabilities for Diabetic class (1)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Apply custom threshold for healthcare screening
    preds = (probs >= THRESHOLD).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    
    results[name] = {"Accuracy": acc, "Recall": rec, "F1-Score": f1}
    
    print(f"=== {name} ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Recall:   {rec*100:.2f}% (Priority Metric)")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix (TN, FP / FN, TP):\n{cm}\n")
    
    # Track model with best recall
    if rec > best_recall:
        best_recall = rec
        best_model = model
        best_model_name = name

print(f"🏆 Best Model Selected (Highest Recall): {best_model_name} ({best_recall*100:.2f}%)")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print("Saved best_model.pkl successfully!")