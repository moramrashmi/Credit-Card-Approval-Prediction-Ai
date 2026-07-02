import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Create Output Folder
# ============================================================

os.makedirs("outputs/plots", exist_ok=True)

# ============================================================
# Load Metrics
# ============================================================

logistic = pd.read_csv("outputs/metrics/logistic_metrics.csv")
decision = pd.read_csv("outputs/metrics/decision_tree_metrics.csv")
random_forest = pd.read_csv("outputs/metrics/random_forest_metrics.csv")

# ============================================================
# Combine Metrics
# ============================================================

comparison = pd.concat(
    [logistic, decision, random_forest],
    ignore_index=True
)

print(comparison)

# Save comparison table
comparison.to_csv(
    "outputs/metrics/model_comparison.csv",
    index=False
)

# ============================================================
# Convert to Long Format
# ============================================================

comparison_long = comparison.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
    var_name="Metric",
    value_name="Score"
)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=comparison_long,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("Model Performance Comparison")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0.0, 1.05)

# Add values on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8)

plt.tight_layout()

plt.savefig(
    "outputs/plots/model_comparison_bar_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nModel comparison graph saved successfully!")