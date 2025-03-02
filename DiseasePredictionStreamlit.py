import streamlit as st
import requests

# Replace with your actual API URL
API_URL = "https://ddsystem.onrender.com/predict"

st.title("🩺 AI-Powered Self-Learning Medical Assistant")

user_input = st.text_input("Enter symptoms (separated by spaces):")

if st.button("Predict Disease"):
    if not user_input.strip():
        st.error("Please enter symptoms before predicting.")
    else:
        try:
            response = requests.post(API_URL, json={"symptoms": user_input})
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Disease: **{result.get('predicted_disease', 'Unknown')}**")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
