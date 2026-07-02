import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Read Dataset
app = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")
print("=" * 60)
print("Occupation Type Distribution")
print("=" * 60)

print(app["OCCUPATION_TYPE"].value_counts())

plt.figure(figsize=(14,6))

sns.countplot(
    x="OCCUPATION_TYPE",
    data=app,
    palette="Set2",
    order=app["OCCUPATION_TYPE"].value_counts().index
)

plt.title("Distribution of Occupation Type", fontsize=16)
plt.xlabel("Occupation Type", fontsize=12)
plt.ylabel("Number of Applicants", fontsize=12)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/plots/occupation_type_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()