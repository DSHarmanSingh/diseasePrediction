import os
from flask import Flask, request, jsonify
from flask_cors import  CORS
import numpy as  np
import tensorflow as tf
import joblib
from pymongo import MongoClient

#Load trained deep learning model $ label encoder
model= tf.keras.models.load_model("DiseasePrediction_DeepLearning.h5")
label_encoder= joblib.load("DiseasePrediction_LabelEncoder.pkl")
tfidf= joblib.load("DiseasePrediction_tfidf.pkl")

#Connect with MongoDB
#client= MongoClient("")
#db= client['medical ai']
#feedback_collection= db["user_feedback"]


app= Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to AI-Powered Healthcare API!"

@app.route('/predict', methods=['POST'])
def predict():
    data= request.json
    symptoms_text= data['symptoms']
    #Convert symptoms to tfidf vector
    symptoms_vector= tfidf.transform([symptoms_text]).toarray()
    prediction= model.predict(symptoms_vector)
    predicted_label= np.argmax(prediction)
    predicted_disease= label_encoder.inverse_transform([predicted_label])[0]
    return jsonify({"predicted_disease": predicted_disease})

if __name__ == '__main__':
    port= int(os.environ.get("PORT", 10000))
    app.run(debug= True, host= '0.0.0.0', port= port)
