# ============================================================
# Credit Card Approval Prediction
# Deployment Model
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# Create Folder
# ============================================================

os.makedirs("flask_app/model", exist_ok=True)

# ============================================================
# Read Processed Dataset
# ============================================================

df = pd.read_csv("outputs/data/processed_data.csv")

print(df.shape)

# ============================================================
# Select Only Deployment Features
# ============================================================

features = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS"
]

X = df[features]

y = df["STATUS_BIN"]

print("\nSelected Features")

print(X.columns)

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# Train Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# Accuracy
# ============================================================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy :", accuracy)

# ============================================================
# Save Model
# ============================================================

joblib.dump(
    model,
    "flask_app/model/deployment_random_forest.pkl"
)

print("\nDeployment Model Saved Successfully!")