
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator


# ==========================================
# 1. PATHS AND MODEL LOADING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]


# ==========================================
# 2. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Customer Churn Prediction API",
    version="2.0.0"
)


# ==========================================
# 3. INPUT VALIDATION
# ==========================================

class CustomerData(BaseModel):

    gender: Literal["Female", "Male"]

    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0, le=72)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service"
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0, allow_inf_nan=False)

    TotalCharges: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_service_combinations(self):

        # Phone service validation
        if self.PhoneService == "No":
            if self.MultipleLines != "No phone service":
                raise ValueError(
                    "MultipleLines must be 'No phone service' "
                    "when PhoneService is 'No'."
                )

        elif self.MultipleLines == "No phone service":
            raise ValueError(
                "MultipleLines cannot be 'No phone service' "
                "when PhoneService is enabled."
            )

        # Internet service validation
        internet_dependent_features = [
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies
        ]

        if self.InternetService == "No":

            if any(
                value != "No internet service"
                for value in internet_dependent_features
            ):
                raise ValueError(
                    "All internet-dependent services must be "
                    "'No internet service' when InternetService is 'No'."
                )

        else:

            if any(
                value == "No internet service"
                for value in internet_dependent_features
            ):
                raise ValueError(
                    "'No internet service' is only valid when "
                    "InternetService is 'No'."
                )

        return self


# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_churn(customer: CustomerData):

    input_data = pd.DataFrame([
        customer.model_dump()
    ])

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    prediction = int(probability >= threshold)

    return {
        "churn_probability": round(probability, 4),
        "threshold": threshold,
        "prediction": prediction,
        "risk_level": "High" if prediction else "Low"
    }
