📜 README.md
md
Copy
Edit
# 🏥 Fraud Detection API

This is a **FastAPI-based Fraud Detection API** that predicts whether a given **medical invoice** is fraudulent or valid using a **trained machine learning model**.

## 🚀 Features
- 📄 **Upload a medical invoice** in PDF format.
- 🔍 **Extract invoice text** using `PyMuPDF` (OCR support can be added).
- 🤖 **Predict fraudulence** using a pre-trained **Random Forest model**.
- 🖥️ **Swagger UI Integration** for easy API testing.
- 📂 **Fast & Secure** with automatic file cleanup.

---

## 🏗️ Model Overview
We use a **Random Forest Classifier** trained on a dataset of **medical invoices**, with features such as:
- **Patient Name**
- **Claim Amount**
- **Diagnosis**
- **Date of Service**

The model was trained using **Scikit-Learn** and saved as `fraud_model.pkl` for deployment.

### **🔧 How the Model Works**
1. The invoice **PDF is uploaded**.
2. We **extract key details** (OCR/text extraction).
3. The extracted data is **converted into a DataFrame**.
4. The model **predicts** whether the invoice is **Fraudulent** or **Valid**.

---

## 🛠️ Installation & Setup
### **1️⃣ Install Dependencies**
Make sure you have Python installed, then run:
```sh
pip install -r requirements.txt
2️⃣ Run the API
sh
Copy
Edit
uvicorn main:app --reload
This will start the FastAPI server on http://127.0.0.1:8000.

3️⃣ Access Swagger UI
Visit:

arduino
Copy
Edit
http://127.0.0.1:8000/docs
to test the API with Swagger.

📤 API Endpoints
Method	Endpoint	Description
POST	/predict	Upload a PDF invoice & get prediction
Example Request
sh
Copy
Edit
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@fraudulent_invoice.pdf'
Example Response
json
Copy
Edit
{
  "prediction": "Fraudulent",
  "extracted_data": {
    "patient_name": "John Doe",
    "claim_amount": 5000,
    "diagnosis": "Migraine",
    "date_of_service": "2024-03-12"
  }
}
🖼️ Screenshots
📌 Swagger API (Fraud Detection)
<img width="944" alt="fraud_detection" src="https://github.com/user-attachments/assets/d9eaf72e-68d0-44a1-8ff4-45cc442ba80c" />


📌 Real Invoice (Valid Detection)

📝 Note: Place your screenshots inside the screenshots/ folder.

🏛️ Tech Stack
FastAPI (Backend Framework)
Scikit-Learn (Machine Learning)
PyMuPDF (fitz) (PDF Text Extraction)
Pandas & NumPy (Data Processing)
Uvicorn (ASGI Server)
📌 Future Enhancements
✅ Integrate OCR (Tesseract) for better text extraction
✅ Improve model accuracy with deep learning
✅ Add JWT authentication for security

📜 License
This project is licensed under the MIT License.

