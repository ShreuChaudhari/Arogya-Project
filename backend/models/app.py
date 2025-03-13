from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import re

app = FastAPI()

# Load model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

if not hasattr(vectorizer, "vocabulary_"):
    raise RuntimeError("Vectorizer does not have a vocabulary. Ensure it was saved properly.")

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF."""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text("text") + "\n"
    return text.strip()

def extract_invoice_fields(text):
    """Extract structured fields from invoice text using regex."""
    
    fields = {
        "Claim Amount": 0,
        "Diagnosis": "",
        "Hospital Name": "",
        "Treatment Details": ""
    }

    # Example regex patterns (modify as per invoice format)
    claim_amount_match = re.search(r"Claim Amount:\s*\$?(\d+[\.\d+]*)", text)
    diagnosis_match = re.search(r"Diagnosis:\s*(.*)", text)
    hospital_match = re.search(r"Hospital Name:\s*(.*)", text)
    treatment_match = re.search(r"Treatment Details:\s*(.*)", text)

    if claim_amount_match:
        fields["Claim Amount"] = float(claim_amount_match.group(1))
    
    if diagnosis_match:
        fields["Diagnosis"] = diagnosis_match.group(1)

    if hospital_match:
        fields["Hospital Name"] = hospital_match.group(1)

    if treatment_match:
        fields["Treatment Details"] = treatment_match.group(1)

    return fields

@app.post("/predict/")
async def predict_fraud(file: UploadFile = File(...)):
    try:
        # Extract text from PDF
        pdf_bytes = await file.read()
        extracted_text = extract_text_from_pdf(pdf_bytes)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text found in the PDF.")

        # Extract structured fields
        invoice_data = extract_invoice_fields(extracted_text)

        # Convert structured data to DataFrame
        df_invoice = pd.DataFrame([invoice_data])

        # Separate numerical and text data
        text_features = ["Diagnosis", "Hospital Name", "Treatment Details"]
        num_features = ["Claim Amount"]

        # Vectorize text fields
        combined_text = df_invoice[text_features].astype(str).apply(" ".join, axis=1)
        vectorized_text = vectorizer.transform(combined_text)

        # Convert vectorized text into DataFrame
        vectorized_df = pd.DataFrame(vectorized_text.toarray(), columns=vectorizer.get_feature_names_out())

        # Add numerical fields
        for num_feature in num_features:
            vectorized_df[num_feature] = df_invoice[num_feature]

        # Ensure model features are correctly aligned
        model_features = model.feature_names_in_
        vectorized_df = vectorized_df.reindex(columns=model_features, fill_value=0)

        # Predict fraud
        prediction = model.predict(vectorized_df)

        return {
            "filename": file.filename,
            "prediction": "Fraudulent" if prediction[0] == 1 else "Valid"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
