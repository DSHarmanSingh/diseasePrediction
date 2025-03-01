import pandas as  pd
import sklearn.preprocessing import LabelEncoder
import joblib
df_v= pd.read_csv(r"C:\Users\harma\OneDrive\Desktop\datasets\diseaseSymptomsVectorized.csv")
X= df_v.drop(columns= ["disease"])
y= df_v["disease"]
#Encode target variable
label_encoder= LabelEncoder()
y_encoded= label_encoder.fit_transform(y)

#Save label encoder for decoding prediction later
joblib.dump(label_encoder, "DiseasePrediction_LabelEncoder.pkl")
