# ============================================================
# Credit Card Approval Prediction
# Data Preprocessing
# ============================================================

import os
import pandas as pd

# ============================================================
# Create Output Directory
# ============================================================

os.makedirs("outputs/data", exist_ok=True)

# ============================================================
# Read Datasets
# ============================================================

print("=" * 60)
print("Reading Datasets...")
print("=" * 60)

application_df = pd.read_csv("dataset/application_record.csv")
credit_df = pd.read_csv("dataset/credit_record.csv")

print(f"Application Dataset Shape : {application_df.shape}")
print(f"Credit Dataset Shape      : {credit_df.shape}")

# ============================================================
# Remove Duplicate Records
# ============================================================

print("\n" + "=" * 60)
print("Checking Duplicate Records...")
print("=" * 60)

print(f"Application Duplicates : {application_df.duplicated().sum()}")
print(f"Credit Duplicates      : {credit_df.duplicated().sum()}")

application_df.drop_duplicates(inplace=True)
credit_df.drop_duplicates(inplace=True)

print("Duplicate Records Removed Successfully.")

# ============================================================
# Handle Missing Values
# ============================================================

print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)

print("\nApplication Dataset Missing Values:\n")
print(application_df.isnull().sum())

print("\nCredit Dataset Missing Values:\n")
print(credit_df.isnull().sum())

# Remove rows where Occupation is missing
application_df.dropna(subset=["OCCUPATION_TYPE"], inplace=True)

print("\nMissing Values Handled Successfully.")

# ============================================================
# Convert STATUS to Binary
# ============================================================

print("\n" + "=" * 60)
print("Converting STATUS to Binary")
print("=" * 60)

def to_binary(status):
    if status in ['0', 'X', 'C']:
        return 1
    else:
        return 0

credit_df["STATUS_BIN"] = credit_df["STATUS"].apply(to_binary)

# Remove original STATUS column
credit_df.drop(columns=["STATUS"], inplace=True)

print(credit_df["STATUS_BIN"].value_counts())

# ============================================================
# Merge Datasets
# ============================================================

print("\n" + "=" * 60)
print("Merging Datasets")
print("=" * 60)

final_df = pd.merge(
    application_df,
    credit_df,
    on="ID",
    how="left"
)

print(f"Merged Dataset Shape : {final_df.shape}")

# ============================================================
# Check Missing Values After Merge
# ============================================================

print("\n" + "=" * 60)
print("Missing Values After Merge")
print("=" * 60)

print(final_df.isnull().sum().sort_values(ascending=False))

# ============================================================
# Save Cleaned Dataset
# ============================================================

final_df.to_csv(
    "outputs/data/cleaned_data.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")

print("=" * 60)
print("Data Preprocessing Completed Successfully")
print("=" * 60)