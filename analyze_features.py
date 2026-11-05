
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype("string").str.strip(),
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["ChurnBinary"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if df["ChurnBinary"].isna().any():
    raise ValueError("Unexpected values found in Churn column.")


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

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


# ============================================================
# OVERALL CHURN BASELINE
# ============================================================

overall_churn_rate = df["ChurnBinary"].mean() * 100

print("\n========== CUSTOMER CHURN ANALYSIS ==========")

print(f"\nTotal customers: {len(df):,}")
print(f"Customers who churned: {df['ChurnBinary'].sum():,}")
print(f"Overall churn rate: {overall_churn_rate:.2f}%")


# ============================================================
# CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n========== COMPLETE CATEGORICAL ANALYSIS ==========")

for feature in categorical_features:

    print(f"\n{'=' * 55}")
    print(f"FEATURE: {feature}")
    print("=" * 55)

    summary = (
        df.groupby(feature, observed=True)
        .agg(
            customers=("ChurnBinary", "size"),
            churn_rate=("ChurnBinary", "mean")
        )
    )

    summary["churn_rate"] *= 100

    summary["difference_from_average"] = (
        summary["churn_rate"] - overall_churn_rate
    )

    summary = summary.sort_values(
        "churn_rate",
        ascending=False
    )

    print(summary.round(2).to_string())


print("\n========== ANALYSIS COMPLETED ==========")
