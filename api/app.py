import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import mlflow
import mlflow.xgboost

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using XGBoost + MLflow",
    version="1.0.0"
)

# ── Load model ─────────────────────────────────────────────────────
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

# ── Request schema ─────────────────────────────────────────────────
class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount: float
    Time: float

# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "fraud-detection-xgboost"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data = transaction.model_dump()

        # Scale Amount and Time
        amount_scaled = scaler.transform([[data['Amount']]])[0][0]
        time_scaled   = scaler.transform([[data['Time']]])[0][0]

        # Build feature vector
        features = [data[f"V{i}"] for i in range(1, 29)]
        features += [amount_scaled, time_scaled]
        features = np.array(features).reshape(1, -1)

        # Predict
        prob  = model.predict_proba(features)[0][1]
        pred  = int(prob >= 0.5)
        alert = "FRAUD" if pred == 1 else "LEGIT"

        return {
            "prediction":        pred,
            "fraud_probability": round(float(prob), 4),
            "alert":             alert,
            "risk_level":        "HIGH" if prob >= 0.8
                            else "MEDIUM" if prob >= 0.5
                            else "LOW"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/recent")
def recent_predictions():
    """Read latest predictions from Delta Lake output"""
    try:
        delta_path = "delta_output/predictions"
        if not os.path.exists(delta_path):
            return {"message": "No predictions yet — start the streaming consumer first"}
        df = pd.read_parquet(delta_path)
        return df.tail(20).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
