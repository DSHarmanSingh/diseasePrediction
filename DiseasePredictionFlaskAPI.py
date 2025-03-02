from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
model = tf.keras.models.load_model("DiseasePrediction_DeepLearning.h5")
label_encoder = joblib.load("LabelEncoder.pkl")
tfidf = joblib.load("tfidf.pkl")
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to AI-Powered Healthcare API!"
# Load models


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'symptoms' not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms_text = data['symptoms'].strip()
    if not symptoms_text:
        return jsonify({"error": "Empty symptoms received"}), 400

    symptoms_vector = tfidf.transform([symptoms_text]).toarray()
    prediction = model.predict(symptoms_vector)
    predicted_label = np.argmax(prediction)
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

    return jsonify({"predicted_disease": predicted_disease})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
