import numpy as np
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df= pd.read_csv(r"C:\Users\harma\OneDrive\Desktop\datasets\DiseaseAndSymptoms.csv")
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

tfidf= TfidfVectorizer(max_features= 500)#Limiting features for efficiency
X_tfidf= tfidf.fit_transform(df["symptoms_list"])
tfidf_df= pd.DataFrame(X_tfidf.toarray(), columns= tfidf.get_feature_names_out())
df_final= pd.concat([df[["disease"]], tfidf_df], axis= 1)

joblib.dump(tfidf, "DiseasePrediction_tfidf.pkl")
