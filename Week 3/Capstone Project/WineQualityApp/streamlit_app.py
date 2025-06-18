import streamlit as st
import cloudpickle
import numpy as np
import pandas as pd

# === Load the model and scaler (if used) ===
with open('model.pkl', 'rb') as f:
    model = cloudpickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = cloudpickle.load(f)

# === Optional: map encoded output to class names ===
quality_map = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent", 4: "Outstanding"}

# === Streamlit App ===
st.title("🍷 Spanish Wine Quality Classifier")

st.write("Enter wine details below to predict the quality class.")

# === User Inputs ===
price = st.number_input("Price (€)", min_value=1.0, max_value=1000.0, step=0.1)
body = st.slider("Body (1 to 5)", 1, 5, 3)
acidity = st.slider("Acidity (1 to 5)", 1, 5, 3)

wine_type = st.selectbox("Type", ["Red", "White", "Rosé"])
region = st.selectbox("Region", ["Rioja", "Ribera del Duero", "Toro", "Other"])

# === One-hot encode type and region manually ===
input_dict = {
    'price': price,
    'body': body,
    'acidity': acidity,
    'type_Red': 1 if wine_type == 'Red' else 0,
    'type_White': 1 if wine_type == 'White' else 0,
    'region_Rioja': 1 if region == 'Rioja' else 0,
    'region_Ribera del Duero': 1 if region == 'Ribera del Duero' else 0,
    'region_Toro': 1 if region == 'Toro' else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# === Scale the features if scaler was used ===
input_scaled = scaler.transform(input_df)

# === Predict ===
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    predicted_class = quality_map.get(prediction, "Unknown")
    st.success(f"Predicted Wine Quality: **{predicted_class}**")
