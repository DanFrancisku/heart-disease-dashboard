import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Disease Risk Dashboard", layout="wide")
st.title("❤️ Heart Disease Risk Dashboard")

DATA_PATH = Path("heart_clean.csv")
MODEL_PATH = Path("rf_model.pkl")

@st.cache_data
def load_data(path: Path):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_model(path: Path):
    model = joblib.load(path)
    return model

st.sidebar.header("Project Status")
has_data = DATA_PATH.exists()
has_model = MODEL_PATH.exists()
st.sidebar.write(f"Data file found: {'✅' if has_data else '❌'}")
st.sidebar.write(f"Model file found: {'✅' if has_model else '❌'}")

tab1, tab2, tab3 = st.tabs(["Overview", "Risk Factors", "Prediction"])

with tab1:
    st.subheader("Dataset Overview")
    if has_data:
        df = load_data(DATA_PATH)
        st.write(df.head())
        st.write(df.describe())
    else:
        st.info("Please add `heart_clean.csv` in the folder and rerun.")

with tab2:
    st.subheader("Risk Factor Charts")
    if has_data:
        import plotly.express as px
        df = load_data(DATA_PATH)
        fig = px.histogram(df, x="age", color="num", title="Age vs Heart Disease")
        st.plotly_chart(fig)
    else:
        st.info("Please add `heart_clean.csv` in the folder.")

with tab3:
    st.subheader("Prediction")
    if has_model:
        model = load_model(MODEL_PATH)
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["male", "female"])
        chol = st.number_input("Cholesterol", 100, 400, 200)
        trestbps = st.number_input("Resting BP", 80, 200, 120)
        thalach = st.number_input("Max heart rate", 60, 220, 150)
        sex_val = 1 if sex == "male" else 0
        X = np.array([[age, sex_val, chol, trestbps, thalach]])
        if st.button("Predict"):
            pred = model.predict(X)[0]
            st.success(f"Prediction: {'Heart Disease' if pred==1 else 'No Disease'}")
    else:
        st.info("Please add `rf_model.pkl` in the folder.")
