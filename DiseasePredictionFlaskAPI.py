import numpy as np
import joblib
import tensorflow.lite as tflite
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

# Load Label Encoder & TF-IDF Vectorizer
label_encoder = joblib.load("LabelEncoder.pkl")
tfidf = joblib.load("tfidf.pkl")

# Load TF-Lite Model
interpreter = tflite.Interpreter(model_path="DiseasePrediction_DeepLearning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to AI-Powered Healthcare API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms_text = data['symptoms']

    # Convert symptoms to TF-IDF vector
    symptoms_vector = tfidf.transform([symptoms_text]).toarray().astype(np.float32)

    # Run TF-Lite model
    interpreter.set_tensor(input_details[0]['index'], symptoms_vector)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted disease
    predicted_label = np.argmax(prediction)
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

    return jsonify({"predicted_disease": predicted_disease})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
