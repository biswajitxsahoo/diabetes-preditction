import pandas as pd
import numpy as np

# 1. Load the Pima Indians Diabetes Dataset
df = pd.read_csv('data/diabetes.csv')

# 2. Print shape (Rows, Columns)
print("--- DATASET SHAPE ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

# 3. Print first 5 rows to confirm structure
print("--- FIRST 5 ROWS ---")
print(df.head())

# 4. Check class distribution (Diabetic vs Non-Diabetic)
print("\n--- CLASS DISTRIBUTION (Outcome) ---")
print(df['Outcome'].value_counts())

# 5. Count biologically impossible zero values in 5 specific features
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\n--- ZERO COUNTS IN CLINICAL FEATURES ---")
for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    pct = (zero_count / len(df)) * 100
    print(f"{col}: {zero_count} zeros ({pct:.1f}%)")