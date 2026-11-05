
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
).fillna(0)

y = df["Churn"].map({"No": 0, "Yes": 1})

features = [
    col for col in df.columns
    if col not in ["customerID", "Churn"]
]

X = df[features]

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    col for col in features
    if col not in numeric_features
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric_features
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

# ==========================================
# 1. GENERATE OUT-OF-FOLD PROBABILITIES
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


# ==========================================
# 2. SELECT CLASSIFICATION THRESHOLD
# ==========================================

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

best_threshold = None
best_f1 = -1

print("\nTHRESHOLD SELECTION USING TRAINING DATA")

for threshold in thresholds:

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nSELECTED THRESHOLD")
print(f"Threshold: {best_threshold}")
print(f"Validation F1: {best_f1:.4f}")


# ==========================================
# 3. TRAIN FINAL MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 4. FINAL TEST EVALUATION
# ==========================================

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions
).ravel()

print("\nFINAL TEST SET EVALUATION")
print(f"Selected threshold: {best_threshold}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives: {tn}")
