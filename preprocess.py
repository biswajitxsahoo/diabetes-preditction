import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Load Data
df = pd.read_csv('data/diabetes.csv')

# 2. Replace impossible zeros with NaN in the 5 clinical columns
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)

# 3. Separate Features (X) and Target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Stratified Train-Test Split (80% Train, 20% Test)
# stratify=y maintains the exact same proportion of diabetic/non-diabetic in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Impute Missing Values (NaN) using TRAIN set mean
for col in zero_cols:
    mean_val = X_train[col].mean()
    X_train[col] = X_train[col].fillna(mean_val)
    X_test[col] = X_test[col].fillna(mean_val)

# 6. Min-Max Normalization
scaler = MinMaxScaler()
# Fit on train data, transform train data
X_train_scaled = scaler.fit_transform(X_train)
# Transform test data using the fitted scaler
X_test_scaled = scaler.transform(X_test)

# Convert back to pandas DataFrames to keep column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# 7. Save preprocessed objects for later stages
joblib.dump(X_train_scaled, 'X_train_scaled.pkl')
joblib.dump(X_test_scaled, 'X_test_scaled.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("--- STAGE 2 COMPLETE ---")
print(f"X_train Shape: {X_train_scaled.shape}")
print(f"X_test Shape: {X_test_scaled.shape}")
print("Cleaned & scaled datasets saved to disk successfully!")