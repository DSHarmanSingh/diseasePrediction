import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
#Loading model and label encoding for decoding predictions
model= load_model("DiseasePrediction_DeepLearning.h5")
label_encoder= joblib.load("DiseasePrediction_LabelEncoder.pkl")

diseases= label_encoder.classes_
Q_table = {disease: 0 for disease in diseases}


#Reward function
def get_reward(predicted_disease, correct_disease):
    if predicted_disease== correct_disease:
        return 1
    else:
        return -1

def update_q_table(predicted_disease, correct_disease):
    reward= get_reward(predicted_disease, correct_disease)
    Q_table[predicted_disease] += 0.1*(reward- Q_table[predicted_disease])
    return reward
#Integrating RL with the model prediction
def predict_disease(symptom_vector):
    prediction= model.predict(symptom_vector.reshape(1, -1))#Get prediction probabilities
    predicted_label= np.argmax(prediction)#Get highest probabilities class
    predicted_disease= label_encoder.inverse_transform([predicted_label])[0]#Converting to disease name
    return predicted_disease

def self_learning_prediction(symptom_vector, correct_disease= None):
    predicted_disease= predict_disease(symptom_vector)

    if correct_disease:
        reward= update_q_table(predicted_disease, correct_disease)
        print(f"User Feedback Received! Adjusted Model for {predicted_disease} (Reward:{reward})")
        return predicted_disease

#Running the self_learning Ai
'''sample_symptom_vector= np.random.rand(model.input_shape[1])#random test vector

predicted_disease= self_learning_prediction(sample_symptom_vector)
print(f"Predicted Disease: {predicted_disease}")

true_disease= "Malaria"
self_learning_prediction(sample_symptom_vector, correct_disease= true_disease)'''
