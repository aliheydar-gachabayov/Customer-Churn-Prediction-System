"""Evaluate the saved churn model on the original held-out test split.

Run: python evaluate_model.py
Outputs are written to reports/ (created automatically).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
REPORT_DIR = BASE_DIR / "reports"


def load_test_data():
    """Recreate the exact stratified holdout split used by train.py."""
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce",
    ).fillna(0)

    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected target values detected.")

    features = [
        column for column in df.columns
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


def main():
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}. Run python train.py first."
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = float(artifact["threshold"])

    X_test, y_test = load_test_data()
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "evaluation_split": "stratified 20% holdout; random_state=42",
        "test_samples": int(len(y_test)),
        "decision_threshold": threshold,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ROC and precision-recall curves use probabilities, not hard predictions.
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fraction_positive, mean_predicted = calibration_curve(
        y_test, probabilities, n_bins=10, strategy="quantile"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    axes[0, 0].set(
        title="ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        recall, precision,
        label=f"Average precision = {metrics['average_precision']:.3f}",
    )
    axes[0, 1].axhline(
        y=float(y_test.mean()),
        linestyle="--",
        label="Positive-class prevalence",
    )
    axes[0, 1].set(
        title="Precision-recall curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    matrix = np.array([[tn, fp], [fn, tp]])
    image = axes[1, 0].imshow(matrix)
    axes[1, 0].set(
        title=f"Confusion matrix (threshold = {threshold:.2f})",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
    )
    for row in range(2):
        for column in range(2):
            axes[1, 0].text(
                column, row, f"{matrix[row, column]:,}",
                ha="center", va="center",
            )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(
        mean_predicted, fraction_positive,
        marker="o",
        label="Model",
    )
    axes[1, 1].plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axes[1, 1].set(
        title="Calibration curve",
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("Customer churn model — held-out test evaluation", fontsize=15)
    fig.tight_layout()
    plot_path = REPORT_DIR / "evaluation_dashboard.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("HELD-OUT TEST EVALUATION")
    print(f"Threshold:          {threshold:.3f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"Average precision:  {metrics['average_precision']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"Specificity:        {metrics['specificity']:.4f}")
    print(f"F1-score:           {metrics['f1']:.4f}")
    print(f"Brier score:        {metrics['brier_score']:.4f}")
    print(f"Log loss:           {metrics['log_loss']:.4f}")
    print(f"Confusion matrix:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved plots:   {plot_path}")
    print("\nNote: this test set is for final evaluation, not threshold tuning.")


if __name__ == "__main__":
    main()
