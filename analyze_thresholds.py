
"""Business-oriented analysis of churn model predictions.

Uses the original held-out test split and the threshold
stored inside models/churn_model.joblib.

Run: python analyze_thresholds.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
REPORT_DIR = BASE_DIR / "reports"


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce",
    ).fillna(0)

    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected target values detected.")

    features = [
        column
        for column in df.columns
        if column not in ["customerID", "Churn"]
    ]

    X = df[features]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    return X_test, y_test


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "evaluation_split": "Stratified 20% holdout; random_state=42",
        "threshold": float(threshold),
        "test_samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics, predictions


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

def print_business_interpretation(metrics):

    print("\n========== BUSINESS INTERPRETATION ==========")

    print(
        f"\nThe model identified "
        f"{metrics['true_positives']} customers who churned."
    )

    print(
        f"It incorrectly flagged "
        f"{metrics['false_positives']} customers."
    )

    print(
        f"It missed "
        f"{metrics['false_negatives']} customers who churned."
    )

    print(
        f"\nRecall: {metrics['recall']:.2%} "
        "of actual churners were identified."
    )

    print(
        f"Precision: {metrics['precision']:.2%} "
        "of flagged customers actually churned."
    )

    print(
        f"Specificity: {metrics['specificity']:.2%} "
        "of non-churning customers were correctly identified."
    )

    print(
        "\nPredictions can help prioritize retention outreach. "
        "They do not establish why a customer churns."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run python train.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    threshold = float(artifact["threshold"])

    X_test, y_test = load_test_data()

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics, predictions = calculate_metrics(
        y_test,
        probabilities,
        threshold,
    )

    # Display results
    print("\n========== HELD-OUT BUSINESS ANALYSIS ==========")

    print(f"\nDecision threshold: {threshold:.3f}")

    for name, value in metrics.items():

        if isinstance(value, float):
            print(
                f"{name.replace('_', ' ').title()}: {value:.4f}"
            )

        elif isinstance(value, int):
            print(
                f"{name.replace('_', ' ').title()}: {value}"
            )

    print("\nCONFUSION MATRIX")

    print(f"True negatives:  {metrics['true_negatives']}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"True positives:  {metrics['true_positives']}")

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    print_business_interpretation(metrics)

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "business_test_metrics.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nResults saved to: {output_path}")
    print("\n========== ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
