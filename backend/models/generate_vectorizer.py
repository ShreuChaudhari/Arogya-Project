import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Load actual training data
dataset_path = os.path.join(os.path.dirname(__file__), "cleaned_fraud_medical_dataset.csv")
df = pd.read_csv(dataset_path)

# Combine text features (modify based on dataset structure)
text_features = ["Diagnosis", "Hospital Name", "Treatment Details"]  # Adjust based on actual dataset
df["combined_text"] = df[text_features].astype(str).apply(" ".join, axis=1)

# Train vectorizer on real data
vectorizer = TfidfVectorizer()
vectorizer.fit(df["combined_text"])

# Save the vectorizer
save_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
joblib.dump(vectorizer, save_path)

print(f"Vectorizer trained on actual data and saved at: {save_path}")
