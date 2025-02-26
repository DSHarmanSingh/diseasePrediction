import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


df_v= pd.read_csv(r"C:\Users\harma\OneDrive\Desktop\datasets\diseaseSymptomsVectorized.csv")
X= df_v.drop(columns= ["disease"])
y= df_v["disease"]
#Encode target variable
label_encoder= LabelEncoder()
y_encoded= label_encoder.fit_transform(y)

#Save label encoder for decoding prediction later
joblib.dump(label_encoder, "LabelEncoder.pkl")

X_train, X_test, y_train, y_test= train_test_split(X, y_encoded, test_size= 0.2, random_state= 42, stratify= y_encoded)

#Building deep learning model
model= Sequential([Dense(512, activation= 'relu', input_shape= (X_train.shape[1],)), Dropout(0, 3), 
                   Dense(256, activation= 'relu'), Dropout(0, 3), 
                   Dense(128, activation= 'relu'), 
                   Dense(len(np.unique(y_encoded)), activation= 'softmax')])
#Compiling the model
model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])

#Training the model
model.fit(X_train, y_train, validation_dataq= (X_test, y_test), epochs= 30, batch_size= 32)

#Evaluate the model
test_loss, test_acc= model.evaluate(X_test, y_test)
print(f" Model Accuracy: {test_acc:.4f}")

#Save the trained model
model.save("DiseasePrediction_DeepLearning.h5")
print("Model saved as 'DiseasePrediction_DeepLearning.h5'")
