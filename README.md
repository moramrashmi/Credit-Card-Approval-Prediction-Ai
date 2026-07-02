# 💳 Credit Card Approval Prediction System

A complete Machine Learning and Flask-based web application that predicts whether a credit card application should be **Approved** or **Rejected** based on an applicant's demographic and financial information.

The project follows an end-to-end Machine Learning workflow including data preprocessing, feature engineering, model comparison, deployment model creation, and a responsive Flask web application.

---

# 📌 Project Overview

Credit card approval is an important decision-making process in the banking industry.

This project automates the approval process using Machine Learning algorithms by analyzing applicant information such as:

- Gender
- Annual Income
- Education
- Employment
- Family Status
- Housing Type
- Occupation
- Family Members
- Children

The trained model predicts whether the applicant is eligible for credit card approval.

---

# ✨ Features

- End-to-End Machine Learning Pipeline
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Logistic Regression
- Decision Tree
- Random Forest
- Model Comparison
- Flask Web Application
- Responsive User Interface
- Deployment Ready

---

# 🏗 Technical Architecture

```
Dataset
    │
    ▼
Data Analysis
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Model Training
    │
    ▼
Model Comparison
    │
    ▼
Deployment Model
    │
    ▼
Flask Application
    │
    ▼
Prediction Result
```

---

# 🧠 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Random Forest was selected as the deployment model.

---

# 📊 Dataset

The project uses two datasets:

- application_record.csv
- credit_record.csv

The datasets are merged during preprocessing to create the final training dataset.

---

# ⚙️ Technologies Used

## Programming Language

- Python

## Machine Learning

- Scikit-learn
- NumPy
- Pandas

## Visualization

- Matplotlib
- Seaborn

## Web Development

- Flask
- HTML5
- CSS3
- JavaScript

---

# 📁 Project Structure

```text
Credit Card Approval Prediction
│
├── dataset
│
├── flask_app
│   ├── model
│   ├── static
│   ├── templates
│   └── app.py
│
├── outputs
│
├── src
│
├── requirements.txt
│
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/chandu2006-git/Credit_Card_Prediction_System.git
```

Go to project folder

```bash
cd Credit_Card_Prediction_System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Flask

```bash
cd flask_app
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 🌐 Web Application

The user enters applicant information through the web interface.

The backend processes the input and passes it to the trained Random Forest deployment model.

The prediction is displayed as:

- ✅ Credit Card Approved
- ❌ Credit Card Rejected

---

# 📈 Machine Learning Workflow

1. Import Libraries
2. Read Dataset
3. Exploratory Data Analysis
4. Univariate Analysis
5. Multivariate Analysis
6. Descriptive Statistics
7. Data Cleaning
8. Feature Engineering
9. Categorical Encoding
10. Logistic Regression
11. Decision Tree
12. Random Forest
13. Model Comparison
14. Deployment Model
15. Flask Deployment

---

# 📷 Screenshots

### Home Page

(Add Screenshot)

### Prediction Form

(Add Screenshot)

### Prediction Result

(Add Screenshot)

### Model Comparison

(Add Screenshot)


---

# 🔮 Future Enhancements

- Explainable AI (SHAP)
- User Authentication
- Database Integration
- Loan Risk Analysis
- Credit Score Prediction
- Cloud Deployment with CI/CD

---

# 👨‍💻 Author

**Chandra Sekhar**

AI & Data Science Student

---

# ⭐ If you found this project useful, consider giving it a star!
