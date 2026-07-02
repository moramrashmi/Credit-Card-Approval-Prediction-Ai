import pandas as pd
import os

# Read dataset
app = pd.read_csv("dataset/application_record.csv")

print("=" * 60)
print("Descriptive Statistical Analysis")
print("=" * 60)

summary = app.describe()

print(summary)

# Save summary

os.makedirs("outputs/metrics", exist_ok=True)

summary.to_csv(
    "outputs/metrics/descriptive_statistics.csv",
    index=True
)
print("\nDescriptive statistics saved successfully.")