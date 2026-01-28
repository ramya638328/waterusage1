import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("water_usage_model.pkl")

# App title
st.title("💧 Household Water Usage Prediction")
st.write("Predict daily water consumption based on household conditions")

# User inputs
people = st.number_input(
    "Number of People in the House",
    min_value=1,
    max_value=20,
    value=4
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0,
    max_value=50,
    value=30
)

day_type = st.selectbox(
    "Day Type",
    ["Weekday", "Weekend"]
)

tank_level = st.slider(
    "Current Tank Level (%)",
    min_value=0,
    max_value=100,
    value=50
)

# Convert categorical input
day_type_value = 1 if day_type == "Weekend" else 0

# Prediction button
if st.button("Predict Water Usage"):
    input_data = np.array([
        [people, temperature, day_type_value, tank_level]
    ])
    
    prediction = model.predict(input_data)
    
    st.success(
        f"💦 Estimated Water Usage: {prediction[0]:.2f} Liters"
    )
