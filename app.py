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
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

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
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Rows", len(df))
        with c2: st.metric("Columns", df.shape[1])
        with c3: st.metric("Missing", int(df.isna().sum().sum()))
        st.markdown("#### Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("#### Summary")
        st.write(df.describe(include='all'))
    else:
        st.info("Upload `heart_clean.csv` to the repo to enable this tab.")

with tab2:
    st.subheader("Risk Factor Charts")
    if has_data:
        import plotly.express as px
        df = load_data(DATA_PATH)

        color_col = "num" if "num" in df.columns else None

        if "age" in df.columns:
            st.plotly_chart(
                px.histogram(df, x="age", color=color_col, title="Age distribution"),
                use_container_width=True
            )

        num_cols = df.select_dtypes("number").columns.tolist()
        if len(num_cols) >= 2:
            st.plotly_chart(
                px.scatter(df, x=num_cols[0], y=num_cols[1], color=color_col,
                           title=f"{num_cols[0]} vs {num_cols[1]}"),
                use_container_width=True
            )
        st.caption("Tip: If your target column is named `num`, charts will color by class.")
    else:
        st.info("Upload `heart_clean.csv` to enable charts.")

with tab3:
    st.subheader("Prediction")
    if has_model:
        model = load_model(MODEL_PATH)

        # Inputs (adjust to your model's feature order)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("age", 18, 100, 50)
            trestbps = st.number_input("trestbps (resting BP)", 70, 250, 120)
            chol = st.number_input("chol (cholesterol)", 80, 600, 200)
        with col2:
            thalach = st.number_input("thalach (max heart rate)", 60, 230, 150)
            oldpeak = st.number_input("oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)
            slope = st.selectbox("slope", [0,1,2], index=1)
        with col3:
            sex = st.selectbox("sex", ["female","male"], index=1)
            cp = st.selectbox("cp (chest pain type)", [0,1,2,3], index=0)
            exang = st.selectbox("exang (exercise-induced angina)", [0,1], index=0)

        # Make sure this matches your training pipeline order
        sex_val = 1 if sex == "male" else 0
        X = np.array([[age, sex_val, cp, trestbps, chol, thalach, exang, oldpeak, slope]], dtype=float)

        if st.button("Predict"):
            pred = int(model.predict(X)[0])
            prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
            st.success(f"Prediction: {'Disease (1)' if pred==1 else 'No Disease (0)'}")
            if prob is not None:
                st.write(f"Estimated risk probability: **{prob:.2f}**")
        st.caption("If you trained with different features/preprocessing, adjust the inputs accordingly.")
    else:
        st.info("Upload `rf_model.pkl` to the repo to enable prediction.")
