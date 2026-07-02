import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read Dataset
app = pd.read_csv("dataset/application_record.csv")

# Select numerical columns
numeric_df = app.select_dtypes(include="number")

# Correlation matrix
corr_matrix = numeric_df.corr()

# Plot
plt.figure(figsize=(12,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()

plt.savefig(
    "outputs/plots/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Correlation heatmap saved successfully.")