
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["ChurnBinary"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


print("\n========== COMPLETE CATEGORICAL ANALYSIS ==========")

for feature in categorical_features:

    print(f"\n{'=' * 50}")
    print(f"FEATURE: {feature}")
    print("=" * 50)

    summary = (
        df.groupby(feature, observed=True)
        .agg(
            customers=("ChurnBinary", "size"),
            churn_rate=("ChurnBinary", "mean")
        )
    )

    summary["churn_rate"] *= 100

    print(summary.round(2))