import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)

# ==========================================================
# Load Deployment Model
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "model", "deployment_random_forest.pkl")
)
# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# Prediction
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Debug (optional)
    print(request.form.to_dict())

    # Convert Years -> Days
    age_years = float(request.form["AGE"])
    employment_years = float(request.form["EMPLOYMENT"])

    days_birth = age_years * 365
    days_employed = employment_years * 365

    # Prepare Input Data
    data = pd.DataFrame([{

        "CODE_GENDER": int(request.form["CODE_GENDER"]),
        "FLAG_OWN_CAR": int(request.form["FLAG_OWN_CAR"]),
        "FLAG_OWN_REALTY": int(request.form["FLAG_OWN_REALTY"]),
        "CNT_CHILDREN": int(request.form["CNT_CHILDREN"]),
        "AMT_INCOME_TOTAL": float(request.form["AMT_INCOME_TOTAL"]),
        "NAME_INCOME_TYPE": int(request.form["NAME_INCOME_TYPE"]),
        "NAME_EDUCATION_TYPE": int(request.form["NAME_EDUCATION_TYPE"]),
        "NAME_FAMILY_STATUS": int(request.form["NAME_FAMILY_STATUS"]),
        "NAME_HOUSING_TYPE": int(request.form["NAME_HOUSING_TYPE"]),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "OCCUPATION_TYPE": int(request.form["OCCUPATION_TYPE"]),
        "CNT_FAM_MEMBERS": float(request.form["CNT_FAM_MEMBERS"])

    }])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "✅ Credit Card Approved"
    else:
        result = "❌ Credit Card Rejected"

    return render_template(
        "index.html",
        prediction=result
    )


# ==========================================================
# Run Flask
# ==========================================================

app.debug = False

if __name__ == "__main__":
    app.run()