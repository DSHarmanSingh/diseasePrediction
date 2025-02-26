import streamlit as st
import numpy as np
import tensorflow as tf
import joblit

model= tf.keras.models.load_model("DiseasePrediction_DeepLearning.h5")
label_encoder=  joblib.load("LabelEncoder.pkl")
tfidf_vectorizer= joblib.load("TFIDF_Vectorizer.pkl")

st.title("AI-Powered Self Learning Medical Assistant")
user_input= st.text_area("Enter symptoms (comma-separeated):")

def preprocess_symptoms(user_input):
    input_vector= tfidf_vectorizer.transform([user_input])
    return input_vector.toarray()

if st.button("Predicted Disease"):
    if user_input:
        symptoms_vector= preprocess_symptoms(user_input)
        prediction= model.predict(symptoms_vector)
        predicted_label= np.argmax(prediction)
        prediced_disease= label_encoder.inverse_transform([predicted_label])[0]
        st.success(f"Predicted Disease: **{predicted_disease}**")
    else:
        st.warning("Please enter symptoms to get a prediction!")
