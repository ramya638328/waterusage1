import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Water Usage Prediction")

MODEL_FILE = "waterusage_model.pkl"

st.title("💧 Water Usage Prediction App")

# Check model file
if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found. Upload 'waterusage_model.pkl' to the GitHub repo.")
    st.stop()

# Load model
with open(MODEL_FILE, "rb") as file:
    loaded_obj = pickle.load(file)

# If PKL is dataset (wrong)
if isinstance(loaded_obj, pd.DataFrame):
    st.error("❌ PKL file contains a dataset, not a trained ML model.")
    st.stop()

# If PKL is dictionary
elif isinstance(loaded_obj, dict):
    model = loaded_obj.get("model")
    if model is None:
        st.error("❌ 'model' key not found in PKL file.")
        st.stop()

# If PKL is a trained model
else:
    model = loaded_obj

# Inputs
members = st.number_input("Number of Family Members", min_value=1)
water_today = st.number_input("Water Used Today (Liters)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)

# Prediction
if st.button("Predict Tomorrow's Usage"):
    input_data = np.array([[members, water_today, temperature]])
    prediction = model.predict(input_data)
    st.success(f"✅ Estimated Water Usage: {prediction[0]:.2f} Liters")
