# ❤️ Heart Disease Risk Dashboard

An interactive **Streamlit web app** that visualizes heart-disease risk factors and predicts the likelihood of heart disease using a trained **Random Forest model**.

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-red?logo=streamlit)](https://heart-disease-dashboard-hwxx6xrtxqbobdyxhovvw7.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Model-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-0099ff?logo=plotly)](https://plotly.com/)
[![pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
👉 [Click here to open the live dashboard](https://heart-disease-dashboard-hwxx6xrtxqbobdyxhovvw7.streamlit.app/)

---

## 🧠 Project Overview

This dashboard demonstrates how **data visualization** and **machine learning** can be combined to assist in understanding and predicting heart-disease risk.  
It includes:

- 📊 **Interactive data exploration** (age distribution, feature relationships)
- 🤖 **Machine learning prediction** using a trained `RandomForestClassifier`
- 🩺 **User-friendly input form** for real-time risk estimation
- 💾 Fully deployable Streamlit app hosted in the cloud

---

## 🧩 Features

| Section | Description |
|----------|--------------|
| **Overview** | Displays dataset summary (rows, columns, missing values) |
| **Risk Factors** | Interactive Plotly charts showing patterns and distributions |
| **Prediction** | Accepts 13 key features (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) to predict heart disease probability |

---

## 🧠 Model Information

- **Algorithm:** Random Forest Classifier  
- **Framework:** Scikit-learn  
- **Training Dataset:** Heart Disease UCI Dataset (Cleaned version)  
- **Target Variable:** `target` (1 = disease, 0 = no disease)

Model saved as `rf_model.pkl`.

---

## 🛠️ Tech Stack

- **Python** (3.9 +)  
- **Streamlit** – web app framework  
- **Plotly Express** – interactive charts  
- **Pandas & NumPy** – data manipulation  
- **Scikit-learn & Joblib** – ML model training and serialization  

---

## 📁 Repository Structure
heart-disease-dashboard/ │ ├── app.py ...
│
├── app.py # Streamlit app script
├── heart_clean.csv # Cleaned dataset
├── rf_model.pkl # Trained Random Forest model
├── requirements.txt # Python dependencies
├── runtime.txt # Streamlit runtime version
└── README.md # Project documentation
