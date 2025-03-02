import streamlit as st
import requests  # To send requests to the Flask API

# Flask API URL (Change this to your deployed API URL)
FLASK_API_URL = "https://ddsystem.onrender.com" 
# FLASK_API_URL = "https://your-flask-api.onrender.com/predict"  # If deployed on Render

# Streamlit UI
st.title("🩺 AI-Powered Self-Learning Medical Assistant")
st.write("Enter symptoms separated by spaces and get an AI-powered disease prediction.")

# User input
user_input = st.text_area("Enter symptoms (separated by spaces):")

if st.button("Predict Disease"):
    if user_input.strip():  # Ensure input is not empty
        # Prepare request payload
        payload = {"symptoms": user_input}

        try:
            # Send POST request to Flask API
            response = requests.post(FLASK_API_URL, json=payload)
            result = response.json()  # Get JSON response

            # Display result
            if "predicted_disease" in result:
                st.success(f"🦠 Predicted Disease: **{result['predicted_disease']}**")
            else:
                st.error("⚠️ Error: Unexpected response from API.")
        except Exception as e:
            st.error(f"🚨 API Request Failed: {e}")
    else:
        st.warning("⚠️ Please enter symptoms before predicting.")

