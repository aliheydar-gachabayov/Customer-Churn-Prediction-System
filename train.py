
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
)

y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected target values detected.")

features = [
    column for column in df.columns
    if column not in ["customerID", "Churn"]
]

X = df[features].copy()

print(f"Dataset size: {len(df)}")
print(f"Number of features: {len(features)}")
print(f"Missing TotalCharges: {X['TotalCharges'].isna().sum()}")


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# ==========================================
# 4. PREPROCESSING
# ==========================================

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    column for column in features
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


# ==========================================
# 5. BUILD FINAL MODEL
# ==========================================

final_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    ),
])


# ==========================================
# 6. SELECT THRESHOLD USING OOF PREDICTIONS
# ==========================================

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print("\nGenerating out-of-fold training probabilities...")

oof_probabilities = cross_val_predict(
    final_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

candidate_thresholds = np.round(
    np.arange(0.10, 0.801, 0.01),
    2,
)

threshold_results = []

for threshold in candidate_thresholds:
    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_train, predictions, zero_division=0
    )

    recall = recall_score(
        y_train, predictions, zero_division=0
    )

    f1 = f1_score(
        y_train, predictions, zero_division=0
    )

    threshold_results.append({
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    })


# Select maximum F1; prefer higher threshold when tied.
best_result = max(
    threshold_results,
    key=lambda result: (
        result["f1"],
        result["threshold"],
    ),
)

best_threshold = best_result["threshold"]
best_oof_f1 = best_result["f1"]

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    REPORTS_DIR / "final_threshold_analysis.csv",
    index=False,
)

print("\nTHRESHOLD SELECTION RESULTS")
print("-" * 55)

for result in threshold_results:
    if abs(result["threshold"] - best_threshold) < 1e-9:
        marker = " <-- SELECTED"
    else:
        marker = ""

    print(
        f"Threshold: {result['threshold']:.2f} | "
        f"Precision: {result['precision']:.4f} | "
        f"Recall: {result['recall']:.4f} | "
        f"F1: {result['f1']:.4f}"
        f"{marker}"
    )

print(f"\nSelected threshold: {best_threshold:.2f}")
print(f"Best training OOF F1: {best_oof_f1:.4f}")


# ==========================================
# 7. TRAIN FINAL MODEL
# ==========================================

print("\nTraining final Logistic Regression model...")

final_model.fit(X_train, y_train)


# ==========================================
# 8. FINAL HELD-OUT TEST EVALUATION
# ==========================================

print("\nEvaluating on untouched test set...")

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()

test_metrics = {
    "model": "Logistic Regression",
    "feature_set": "All Features",
    "threshold": float(best_threshold),
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
    "average_precision": float(
        average_precision_score(y_test, test_probabilities)
    ),
    "accuracy": float(
        (test_predictions == y_test).mean()
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "specificity": float(tn / (tn + fp)),
    "f1": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "brier_score": float(
        brier_score_loss(y_test, test_probabilities)
    ),
    "log_loss": float(
        log_loss(y_test, test_probabilities)
    ),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "test_samples": int(len(y_test)),
}

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

for metric, value in test_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")


# ==========================================
# 9. SAVE MODEL AND THRESHOLD
# ==========================================

model_path = MODEL_DIR / "churn_model.joblib"

joblib.dump(
    {
        "model": final_model,
        "threshold": float(best_threshold),
    },
    model_path,
)

print(f"\nSaved model: {model_path}")


# ==========================================
# 10. SAVE TEST METRICS
# ==========================================

metrics_path = REPORTS_DIR / "final_test_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(test_metrics, file, indent=4)

print(f"Saved test metrics: {metrics_path}")

print("\nFinal training and evaluation completed.")
