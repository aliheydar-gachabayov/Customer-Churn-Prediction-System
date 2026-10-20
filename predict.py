
import pandas as pd
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"

model = joblib.load(MODEL_PATH)

customer = pd.DataFrame([{
    "tenure": 12,
    "MonthlyCharges": 75.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic"
}])

probability = model.predict_proba(customer)[0][1]

print(f"\nChurn probability: {probability:.1%}")

if probability >= 0.5:
    print("Prediction: Likely to churn")
else:
    print("Prediction: Unlikely to churn")
