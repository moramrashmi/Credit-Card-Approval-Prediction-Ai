import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("outputs/data/cleaned_data.csv")

categorical_columns = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    encoder.fit(df[column].astype(str))

    print("\n" + "="*50)
    print(column)
    print("="*50)

    for i, value in enumerate(encoder.classes_):
        print(f"{i} -> {value}")