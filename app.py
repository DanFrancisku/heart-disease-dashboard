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
    # safer load so the app doesn't crash if versions differ
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        return None

st.sidebar.header("Project Status")
has_data = DATA_PATH.exists()
has_model = MODEL_PATH.exists()
st.sidebar.write(f"Data file found: {'✅' if has_data else '❌'}")
st.sidebar.write(f"Model file found: {'✅' if has_model else '❌'}")

tab1, tab2, tab3 = st.tabs(["Overview", "Risk Factors", "Prediction"])

# -------------------------
# Tab 1: Overview
# -------------------------
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

# -------------------------
# Tab 2: Risk Factors
# -------------------------
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

# -------------------------
# Tab 3: Prediction (13 features)
# -------------------------
with tab3:
    st.subheader("Prediction")
    if has_model:
        model = load_model(MODEL_PATH)

        if model is None:
            st.info("Prediction disabled until the model loads correctly.")
        else:
            st.markdown("**Enter features (must match training order):**")
            c1, c2, c3 = st.columns(3)

            # Column 1
            with c1:
                age = st.number_input("age", 18, 100, 50)
                sex = st.selectbox("sex", ["female","male"], index=1)     # 0=female, 1=male
                cp = st.selectbox("cp (chest pain type)", [0,1,2,3], index=0)
                trestbps = st.number_input("trestbps (resting BP)", 70, 250, 120)
                chol = st.number_input("chol (cholesterol)", 80, 600, 200)

            # Column 2
            with c2:
                fbs = st.selectbox("fbs (>120 mg/dl)", [0,1], index=0)     # 0/1
                restecg = st.selectbox("restecg", [0,1,2], index=0)        # 0=normal,1=st-t abn,2=lv hypertrophy
                thalach = st.number_input("thalach (max heart rate)", 60, 230, 150)
                exang = st.selectbox("exang (exercise-induced angina)", [0,1], index=0)
                oldpeak = st.number_input("oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)

            # Column 3
            with c3:
                slope = st.selectbox("slope (peak ST segment)", [0,1,2], index=1)
                ca = st.selectbox("ca (major vessels 0–3)", [0,1,2,3], index=0)
                # If your dataset used original thal codes (3,6,7), change to [3,6,7]
                thal = st.selectbox("thal", [0,1,2], index=0)              # 0=normal,1=fixed defect,2=reversible defect

            # map sex to numeric
            sex_val = 1 if sex == "male" else 0

            # IMPORTANT: EXACT training order (13 features)
            X = np.array([[age, sex_val, cp, trestbps, chol,
                           fbs, restecg, thalach, exang, oldpeak,
                           slope, ca, thal]], dtype=float)

            if st.button("Predict"):
                try:
                    pred = int(model.predict(X)[0])
                    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
                    st.success(f"Prediction: {'Disease (1)' if pred==1 else 'No Disease (0)'}")
                    if prob is not None:
                        st.write(f"Estimated risk probability: **{prob:.2f}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            st.caption("If your training used different encodings (e.g., thal = 3/6/7), adjust the dropdown values to match.")
    else:
        st.info("Upload `rf_model.pkl` to the repo to enable prediction.")
