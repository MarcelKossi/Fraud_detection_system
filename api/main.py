from fastapi import FastAPI
import joblib
import numpy as np

from api.schemas import Transaction

app = FastAPI(title="Fraud Detection API")

# Load model and scaler
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}


@app.post("/predict")
def predict(transaction: Transaction):

    features = np.array(transaction.features).reshape(1, -1)

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    probability = model.predict_proba(features_scaled)

    return {
        "prediction": int(prediction[0]),
        "fraud_probability": float(probability[0][1])
    }