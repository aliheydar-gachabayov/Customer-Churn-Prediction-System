
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# 1. LOAD DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
).fillna(0)

y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected target values detected.")


# ==========================================
# 2. DEFINE FEATURES
# ==========================================

features = [
    column for column in df.columns
    if column not in ["customerID", "Churn"]
]

X = df[features]

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    column for column in features
    if column not in numeric_features
]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ==========================================
# 4. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================
# 5. BUILD MODEL PIPELINE
# ==========================================

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])


# ==========================================
# 6. SELECT THRESHOLD USING OOF PREDICTIONS
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probabilities = cross_val_predict(
    final_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

candidate_thresholds = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
]

threshold_results = []

for candidate in candidate_thresholds:

    oof_predictions = (
        oof_probabilities >= candidate
    ).astype(int)

    score = f1_score(y_train, oof_predictions)

    threshold_results.append((candidate, score))

best_threshold, best_f1 = max(
    threshold_results,
    key=lambda result: result[1]
)

print("\nTHRESHOLD SELECTION RESULTS")

for candidate, score in threshold_results:
    print(f"Threshold: {candidate:.2f} | OOF F1: {score:.4f}")

print(f"\nSelected threshold: {best_threshold:.2f}")
print(f"Best OOF F1-score: {best_f1:.4f}")


# ==========================================
# 7. TRAIN FINAL MODEL
# ==========================================

final_model.fit(X_train, y_train)


# ==========================================
# 8. FINAL TEST EVALUATION
# ==========================================

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFINAL TEST RESULTS")
print(f"Threshold: {best_threshold:.2f}")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, test_probabilities):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_test, test_predictions):.4f}"
)

print(
    f"Recall: "
    f"{recall_score(y_test, test_predictions):.4f}"
)

print(
    f"F1-score: "
    f"{f1_score(y_test, test_predictions):.4f}"
)

print("\nClassification Report:")
print(classification_report(y_test, test_predictions))


# ==========================================
# 9. SAVE MODEL AND THRESHOLD
# ==========================================

model_path = MODEL_DIR / "churn_model.joblib"

joblib.dump(
    {
        "model": final_model,
        "threshold": float(best_threshold)
    },
    model_path
)

print(f"\nSaved model: {model_path}")
