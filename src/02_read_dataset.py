import pandas as pd

# Read datasets
app = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")

# Display first 5 rows
print("Application Dataset")
print(app.head())

print("\nCredit Dataset")
print(credit.head())