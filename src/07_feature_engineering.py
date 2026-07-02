# ============================================================
# Credit Card Approval Prediction
# Feature Engineering & Categorical Encoding
# ============================================================

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ============================================================
# Create Output Directory
# ============================================================

os.makedirs("outputs/data", exist_ok=True)

# ============================================================
# Read Cleaned Dataset
# ============================================================

print("=" * 60)
print("Loading Cleaned Dataset...")
print("=" * 60)

credit_app = pd.read_csv("outputs/data/cleaned_data.csv")

print(f"Dataset Shape : {credit_app.shape}")

# ============================================================
# Feature Engineering
# ============================================================

print("\n" + "=" * 60)
print("Feature Engineering")
print("=" * 60)

# Convert negative values to positive
credit_app["DAYS_BIRTH"] = credit_app["DAYS_BIRTH"].abs()
credit_app["DAYS_EMPLOYED"] = credit_app["DAYS_EMPLOYED"].abs()

# Create Total Family Members feature
credit_app["TOTAL_FAMILY_MEMBERS"] = (
    credit_app["CNT_CHILDREN"] +
    credit_app["CNT_FAM_MEMBERS"]
)

print("Feature Engineering Completed Successfully.")

# ============================================================
# Handling Categorical Values
# ============================================================

print("\n" + "=" * 60)
print("Encoding Categorical Columns")
print("=" * 60)

categorical_columns = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]
# ============================================================
# Handle Missing Values
# ============================================================

print("\n" + "=" * 60)
print("Handling Missing Values")
print("=" * 60)

# Numerical Columns
numerical_columns = credit_app.select_dtypes(include=["int64", "float64"]).columns

for column in numerical_columns:
    credit_app[column] = credit_app[column].fillna(
        credit_app[column].median()
    )

# Categorical Columns
categorical_columns = credit_app.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    credit_app[column] = credit_app[column].fillna(
        credit_app[column].mode()[0]
    )

print("Missing Values Handled Successfully.")
label_encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    credit_app[column] = encoder.fit_transform(
        credit_app[column].astype(str)
    )
    label_encoders[column] = encoder

print("Categorical Encoding Completed Successfully.")
# ============================================================
# Verify Missing Values
# ============================================================

print("\nRemaining Missing Values")

print(credit_app.isnull().sum().sort_values(ascending=False))
# ============================================================
# Preview Dataset
# ============================================================

print("\nDataset Preview")

print(credit_app.head())

print("\nDataset Shape")

print(credit_app.shape)

# ============================================================
# Save Processed Dataset
# ============================================================

credit_app.to_csv(
    "outputs/data/processed_data.csv",
    index=False
)

print("\nProcessed Dataset Saved Successfully!")

print("=" * 60)
print("Feature Engineering Completed")
print("=" * 60)