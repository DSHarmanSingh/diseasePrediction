import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load trained deep learning model & label encoder
model = tf.keras.models.load_model("DiseasePrediction_DeepLearning.h5")
label_encoder = joblib.load("LabelEncoder.pkl")
tfidf= joblib.load("tfidf.pkl")

st.title("🩺 AI-Powered Self-Learning Medical Assistant")

# User symptom input
user_input = st.text_area("Enter symptoms (comma-separated):")

if st.button("Predict Disease"):
    symptoms_vector = np.random.rand(model.input_shape[1])  # Replace with real vector
    prediction = model.predict(symptoms_vector.reshape(1, -1))
    predicted_label = np.argmax(prediction)
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Predicted Disease: **{predicted_disease}**")
