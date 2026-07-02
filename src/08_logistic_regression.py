# ============================================================
# Credit Card Approval Prediction
# Logistic Regression Model
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Create Output Directories
# ============================================================

os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Processed Dataset...")
print("=" * 60)

df = pd.read_csv("outputs/data/processed_data.csv")
print("=" * 60)
print("MODEL FEATURES")
print("=" * 60)

print(df.columns.tolist())
print(df.shape)

# ============================================================
# Split Features and Target
# ============================================================

X = df.drop("STATUS_BIN", axis=1)
y = df["STATUS_BIN"]

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
# Model Training
# ============================================================

print("\nTraining Logistic Regression Model...")

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

print("Training Completed!")

# ============================================================
# Prediction
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/plots/logistic_confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# Save Metrics
# ============================================================

metrics = pd.DataFrame({
    "Model": ["Logistic Regression"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1]
})

metrics.to_csv(
    "outputs/metrics/logistic_metrics.csv",
    index=False
)

# ============================================================
# Save Model
# ============================================================

joblib.dump(
    model,
    "models/logistic_regression.pkl"
)

print("\nModel Saved Successfully!")

print("=" * 60)
print("Logistic Regression Completed")
print("=" * 60)