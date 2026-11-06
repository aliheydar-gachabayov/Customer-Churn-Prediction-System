
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Set up project paths

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# 2. Load the dataset

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in Churn column.")

print(f"Dataset shape: {df.shape}")
print(f"Number of features: {X.shape[1]}")


# 3. Split the data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 4. Prepare numerical and categorical features

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# 5. Build and train the model

# Train a simple baseline model for comparison

baseline_model = DummyClassifier(strategy="most_frequent")

print("\nTraining baseline model...")

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000)),
])

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)


# 6. Evaluate the model

baseline_metrics = {
    "accuracy": float(accuracy_score(y_test, baseline_predictions)),
    "precision": float(precision_score(y_test, baseline_predictions, zero_division=0)),
    "recall": float(recall_score(y_test, baseline_predictions, zero_division=0)),
    "f1": float(f1_score(y_test, baseline_predictions, zero_division=0)),
}

threshold = 0.5

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions)),
    "recall": float(recall_score(y_test, predictions)),
    "f1": float(f1_score(y_test, predictions)),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
}

print("\nBASELINE MODEL EVALUATION")
print("=" * 35)

for name, value in baseline_metrics.items():
    print(f"{name}: {value:.4f}")

print("\nMODEL EVALUATION")
print("=" * 35)

for name, value in metrics.items():
    if name == "confusion_matrix":
        continue
    print(f"{name}: {value:.4f}")

print("\nConfusion Matrix:")
print(metrics["confusion_matrix"])


# 7. Save the model

model_path = MODEL_DIR / "churn_model.joblib"

joblib.dump(
    {
        "model": model,
        "threshold": threshold,
    },
    model_path,
)

print(f"\nModel saved to: {model_path}")


# 8. Save evaluation metrics

metrics_path = REPORTS_DIR / "final_test_metrics.json"

evaluation_report = {
    "baseline": baseline_metrics,
    "logistic_regression": metrics,
}

with open(metrics_path, "w") as file:
    json.dump(evaluation_report, file, indent=4)

print(f"Metrics saved to: {metrics_path}")

print("\nTraining completed successfully!")
