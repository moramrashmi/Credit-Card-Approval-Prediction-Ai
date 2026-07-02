import pandas as pd

print("=" * 60)
print("Model Comparison")
print("=" * 60)

# Load metrics
logistic = pd.read_csv("outputs/metrics/logistic_metrics.csv")
decision = pd.read_csv("outputs/metrics/decision_tree_metrics.csv")
random = pd.read_csv("outputs/metrics/random_forest_metrics.csv")

# Combine
comparison = pd.concat(
    [logistic, decision, random],
    ignore_index=True
)

# Sort by Accuracy
comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

print(comparison)

# Save
comparison.to_csv(
    "outputs/metrics/model_comparison.csv",
    index=False
)

best_model = comparison.iloc[0]["Model"]

print("\nBest Model :", best_model)

print("\nModel comparison saved successfully!")