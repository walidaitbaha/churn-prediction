# 📉 Customer Churn Prediction

Predict which telecom customers are likely to churn using
Machine Learning (XGBoost + SMOTE).

## 🎯 Results
| Model | F1 Score | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 0.81 | 0.84 |
| Random Forest | 0.88 | 0.91 |
| **XGBoost + SMOTE** | **0.91** | **0.93** |

## 🔗 Live Demo
👉 [[Try the app here](https://churn-prediction-walid.streamlit.app/)](YOUR_STREAMLIT_URL)

## 📊 Key Findings
- Customers with month-to-month contracts churn 3x more
- Short tenure + high monthly charges = highest risk
- Fiber optic users churn more than DSL users

## 🛠️ Tech Stack
Python · Pandas · Scikit-learn · XGBoost · SMOTE · Streamlit

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py
