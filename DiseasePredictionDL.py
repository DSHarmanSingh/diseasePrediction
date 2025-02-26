import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df= pd.read_csv(r"/kaggle/input/diseaseandsymptoms/DiseaseAndSymptoms.csv")
#Standardizing symptoms names by removing spaces with underscore, and lower casing,
df.columns= df.columns.str.strip().str.lower().str.replace(' ', '_')
for col in df.columns:
    df[col]= df[col].astype(str).str.strip().str.lower().str.replace(' ', '_')
    
#Handling Missing values
#Droping columns of symptoms which have too many missing values
threshold= len(df)* 0.1#Drop if more than 90% missing 
df= df.dropna(thresh= threshold, axis=1)
#Now filling the missing values with "no_symtoms"
df.fillna("no_symptom", inplace= True)
#Converting symptoms into lists per disease
symptom_columns= [col for col in df.columns if "symptom" in col]#for identifying symptoms columsn
df["symptoms_list"]= df[symptom_columns].apply(lambda row: " ".join(set(row.values) - {"no_symptom"}), axis=1)
#drop all odl symptom columns
df= df[["disease", "symptoms_list"]]
df.to_csv(r"/kaggle/working/diseaseSymptomsCleaned.csv", index=False)
tfidf= TfidfVectorizer(max_features= 500)#Limiting features for efficiency
X_tfidf= tfidf.fit_transform(df["symptoms_list"])
tfidf_df= pd.DataFrame(X_tfidf.toarray(), columns= tfidf.get_feature_names_out())
df_final= pd.concat([df[["disease"]], tfidf_df], axis= 1)
df_final.to_csv(r"/kaggle/working/diseaseSymptomsVectorized.csv", index=False)


import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


df_v= pd.read_csv(r"/kaggle/input/diseaseprecaution/Disease precaution.csv")
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
