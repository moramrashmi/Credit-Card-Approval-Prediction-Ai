# ============================================================
# Credit Card Approval Prediction
# Decision Tree Classification
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Processed Dataset...")
print("=" * 60)

df = pd.read_csv("outputs/data/processed_data.csv")

print(f"Dataset Shape : {df.shape}")

# ============================================================
# Features and Target
# ============================================================

X = df.drop("STATUS_BIN", axis=1)
y = df["STATUS_BIN"]

# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# Build Decision Tree Model
# ============================================================

print("\nTraining Decision Tree Model...")

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

print("Training Completed!")

# ============================================================
# Prediction
# ============================================================

print("\nGenerating Predictions...")

y_pred = dt_model.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nModel Performance")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/plots/decision_tree_confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# Save Metrics
# ============================================================

metrics = pd.DataFrame({
    "Model": ["Decision Tree"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1]
})

metrics.to_csv(
    "outputs/metrics/decision_tree_metrics.csv",
    index=False
)

# ============================================================
# Save Model
# ============================================================

joblib.dump(
    dt_model,
    "outputs/models/decision_tree.pkl"
)

print("\nDecision Tree Model Saved Successfully!")

print("=" * 60)
print("Decision Tree Completed")
print("=" * 60)