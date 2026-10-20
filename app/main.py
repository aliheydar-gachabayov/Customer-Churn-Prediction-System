
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

model = joblib.load(MODEL_PATH)


class CustomerData(BaseModel):
    tenure: int = Field(ge=0, le=72)
    MonthlyCharges: float = Field(ge=0)

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]


@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    input_data = pd.DataFrame([customer.model_dump()])

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    prediction = int(probability >= 0.5)

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "risk_level": "High" if prediction else "Low"
    }
