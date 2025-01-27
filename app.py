import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('/Users/mohersi/Desktop/ridge_model.pkl')

# Streamlit interface
st.title("Ridge Regression Model")

# Input field for user features
st.write("Enter the feature values for prediction:")
features = st.text_input("Input features (comma-separated):", "")

# Prediction logic
if st.button("Predict"):
    try:
        # Convert the comma-separated string to a list of floats
        features = [float(x) for x in features.split(",")]

        # Perform the prediction
        prediction = model.predict([features])  # Adjust if your model expects a different shape
        st.write(f"Prediction: {prediction[0]}")
    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
