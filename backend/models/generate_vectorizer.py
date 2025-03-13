import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

dataset_path = os.path.join(os.path.dirname(__file__), "cleaned_fraud_medical_dataset.csv")
df = pd.read_csv(dataset_path)

text_features = ["Diagnosis", "Hospital Name", "Treatment Details"]  
df["combined_text"] = df[text_features].astype(str).apply(" ".join, axis=1)


vectorizer = TfidfVectorizer()
vectorizer.fit(df["combined_text"])

save_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
joblib.dump(vectorizer, save_path)

print(f"Vectorizer trained on actual data and saved at: {save_path}")
