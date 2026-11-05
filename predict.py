
import pandas as pd
import joblib

from pathlib import Path


# ==========================================
# 1. LOAD MODEL ARTIFACT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]


# ==========================================
# 2. CREATE CUSTOMER DATA
# ==========================================

customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0
}])


# ==========================================
# 3. MAKE PREDICTION
# ==========================================

probability = model.predict_proba(customer)[0][1]

prediction = int(probability >= threshold)


# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

print("\nCUSTOMER CHURN PREDICTION")
print("=" * 35)

print(f"Churn probability: {probability:.1%}")
print(f"Decision threshold: {threshold}")

if prediction == 1:
    print("Prediction: Likely to churn")
else:
    print("Prediction: Unlikely to churn")
