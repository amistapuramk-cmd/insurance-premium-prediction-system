"""
Insurance Premium Prediction System - web app.

Run:  streamlit run app.py   (after running train_model.py once)
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Premium Predictor", page_icon="🏥")
st.title("🏥 Insurance Premium Prediction System")
st.write("Enter the customer details to estimate the yearly insurance premium.")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("model.pkl not found. Run `python train_model.py` first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
with col2:
    children = st.number_input("Number of children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Premium", type="primary"):
    row = pd.DataFrame([{
        "age": age, "bmi": bmi, "children": children,
        "sex": sex, "smoker": smoker, "region": region,
    }])
    premium = model.predict(row)[0]
    st.success(f"Estimated yearly premium: **{premium:,.2f}**")
