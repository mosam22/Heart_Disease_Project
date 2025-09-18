import streamlit as st
import pandas as pd
import joblib

st.title("Heart Disease Risk Predictor")

model = joblib.load(r"..\models\final_model.pkl")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0,1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 (1/0)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max HR", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1/0)", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (3,6,7)", [3,6,7])

if st.button("Predict"):
    X_input = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                             exang, oldpeak, slope, ca, thal]],
                           columns=["age","sex","cp","trestbps","chol","fbs",
                                    "restecg","thalach","exang","oldpeak",
                                    "slope","ca","thal"])
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0,1]
    st.write("Prediction:", "Heart Disease" if pred==1 else "No Heart Disease")
    st.write(f"Probability: {proba:.2f}")

