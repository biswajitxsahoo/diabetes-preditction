import pandas as pd
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load scaled training data from Stage 2
X_train = joblib.load('X_train_scaled.pkl')
y_train = joblib.load('y_train.pkl')

print("--- CLASS DISTRIBUTION BEFORE SMOTE ---")
print(y_train.value_counts())

# 2. Apply SMOTE to training set ONLY
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\n--- CLASS DISTRIBUTION AFTER SMOTE ---")
print(y_train_res.value_counts())

# Convert resampled feature array back to DataFrame with column names
X_train_res_df = pd.DataFrame(X_train_res, columns=X_train.columns)

# 3. Save resampled training data
joblib.dump(X_train_res_df, 'X_train_res.pkl')
joblib.dump(y_train_res, 'y_train_res.pkl')

print("\n--- STAGE 3 COMPLETE ---")
print(f"Resampled X_train Shape: {X_train_res_df.shape}")
print("SMOTE balanced training set saved successfully!")