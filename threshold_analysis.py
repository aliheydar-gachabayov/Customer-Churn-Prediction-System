"""Analyze decision-threshold trade-offs using out-of-fold training predictions.

Run: python threshold_analysis.py

This script deliberately does not use the held-out test set to select a
threshold. It clones the saved pipeline's estimator settings, generates
out-of-fold probabilities on the training partition, and reports precision,
recall, specificity, F1, and confusion-matrix counts across thresholds.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
REPORT_DIR = BASE_DIR / "reports"
RANDOM_STATE = 42
N_SPLITS = 5


def load_training_data():
    """Recreate the training partition used by train.py."""
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

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, y_train


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}. Run python train.py first."
        )

    artifact = joblib.load(MODEL_PATH)
    saved_model = artifact["model"]
    saved_threshold = float(artifact["threshold"])

    X_train, y_train = load_training_data()
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # clone() retains hyperparameters but refits independently in each fold.
    oof_probabilities = cross_val_predict(
        clone(saved_model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.round(np.arange(0.10, 0.801, 0.01), 2)
    results = pd.DataFrame([
        calculate_metrics(y_train, oof_probabilities, threshold)
        for threshold in thresholds
    ])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "threshold_analysis.csv"
    results.to_csv(csv_path, index=False)

    # Identify the best F1 threshold on OOF training predictions only.
    best_row = results.loc[results["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results["threshold"], results["precision"], label="Precision")
    axes[0].plot(results["threshold"], results["recall"], label="Recall")
    axes[0].plot(results["threshold"], results["specificity"], label="Specificity")
    axes[0].plot(results["threshold"], results["f1"], label="F1-score", linewidth=2)
    axes[0].axvline(
        saved_threshold,
        linestyle="--",
        label=f"Saved threshold ({saved_threshold:.2f})",
    )
    axes[0].axvline(
        best_threshold,
        linestyle=":",
        label=f"OOF best F1 ({best_threshold:.2f})",
    )
    axes[0].set(
        title="Metrics by decision threshold",
        xlabel="Decision threshold",
        ylabel="Score",
        ylim=(0, 1.02),
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        results["threshold"],
        results["false_positives"],
        label="False positives",
    )
    axes[1].plot(
        results["threshold"],
        results["false_negatives"],
        label="False negatives",
    )
    axes[1].axvline(saved_threshold, linestyle="--", label="Saved threshold")
    axes[1].set(
        title="Error counts by decision threshold",
        xlabel="Decision threshold",
        ylabel="Number of customers",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Threshold analysis — out-of-fold training predictions")
    fig.tight_layout()
    plot_path = REPORT_DIR / "threshold_analysis.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    saved_row = results.iloc[
        (results["threshold"] - saved_threshold).abs().argmin()
    ]

    print("THRESHOLD ANALYSIS — OUT-OF-FOLD TRAINING PREDICTIONS")
    print("=" * 65)
    print(f"Training samples:          {len(y_train)}")
    print(f"Cross-validation folds:    {N_SPLITS}")
    print(f"Saved model threshold:     {saved_threshold:.2f}")
    print(f"Saved threshold OOF F1:    {saved_row['f1']:.4f}")
    print(f"Best OOF F1 threshold:     {best_threshold:.2f}")
    print(f"Best OOF F1-score:         {best_row['f1']:.4f}")
    print("\nMetrics at saved threshold:")
    print(saved_row.to_string())
    print("\nMetrics at best OOF F1 threshold:")
    print(best_row.to_string())
    print("\nSaved files:")
    print(f"- {csv_path}")
    print(f"- {plot_path}")
    print(
        "\nDo not treat the best OOF threshold as a final business decision "
        "without considering the relative costs of false positives and "
        "false negatives. The held-out test set remains reserved for final evaluation."
    )


if __name__ == "__main__":
    main()
