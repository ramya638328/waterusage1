import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("water_usage_model.pkl")

# Safety check
if not hasattr(model, "predict"):
    st.error("Loaded file is not a trained ML model. Please upload correct .pkl file.")
    st.stop()

st.title("💧 Household Water Usage Prediction")
st.write("Predict daily water consumption using ML")

# Inputs
people = st.number_input("Number of People", 1, 20, 4)
temperature = st.number_input("Temperature (°C)", 0, 50, 30)
day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])
tank_level = st.slider("Tank Level (%)", 0, 100, 50)

day_type_value = 1 if day_type == "Weekend" else 0

if st.button("Predict Water Usage"):
    input_data = np.array([[people, temperature, day_type_value, tank_level]])
    prediction = model.predict(input_data)
    st.success(f"💦 Estimated Water Usage: {prediction[0]:.2f} Liters")
