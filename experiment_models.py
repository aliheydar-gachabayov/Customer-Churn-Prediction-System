
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4

DATA_PATH = Path("data/Telco-Customer-Churn.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def build_pipeline(model, X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", model),
    ])


def select_threshold(y_true, probabilities):
    thresholds = np.arange(0.10, 0.801, 0.01)

    scores = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        scores.append((score, threshold))

    # If F1 scores tie, prefer the higher threshold.
    best_f1, best_threshold = max(scores, key=lambda item: (item[0], item[1]))

    return float(best_threshold)


def evaluate_configuration(name, feature_set, model, X, y):
    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    outer_probabilities = np.zeros(len(y), dtype=float)
    outer_predictions = np.zeros(len(y), dtype=int)
    selected_thresholds = []

    start_time = time.perf_counter()

    for fold, (train_idx, validation_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]

        pipeline = build_pipeline(clone(model), X_train)

        # Select the threshold using only inner out-of-fold predictions.
        inner_probabilities = cross_val_predict(
            clone(pipeline),
            X_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        threshold = select_threshold(
            y_train,
            inner_probabilities,
        )

        # Fit on the complete outer training fold.
        pipeline.fit(X_train, y_train)

        # Evaluate on the untouched outer validation fold.
        validation_probabilities = pipeline.predict_proba(
            X_validation
        )[:, 1]

        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        outer_probabilities[validation_idx] = validation_probabilities
        outer_predictions[validation_idx] = validation_predictions
        selected_thresholds.append(threshold)

        print(
            f"  Fold {fold}/{OUTER_FOLDS}: "
            f"threshold={threshold:.2f}"
        )

    elapsed_time = time.perf_counter() - start_time

    return {
        "feature_set": feature_set,
        "model": name,
        "roc_auc": roc_auc_score(y, outer_probabilities),
        "average_precision": average_precision_score(
            y, outer_probabilities
        ),
        "precision": precision_score(
            y, outer_predictions, zero_division=0
        ),
        "recall": recall_score(
            y, outer_predictions, zero_division=0
        ),
        "f1": f1_score(
            y, outer_predictions, zero_division=0
        ),
        "mean_threshold": np.mean(selected_thresholds),
        "threshold_std": np.std(selected_thresholds),
        "evaluation_time_seconds": elapsed_time,
    }


def plot_results(results):
    metrics = ["roc_auc", "average_precision", "f1"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, metric in zip(axes, metrics):
        pivot = results.pivot(
            index="model",
            columns="feature_set",
            values=metric,
        )

        pivot.plot(kind="bar", ax=ax)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Feature set")

    plt.tight_layout()

    output_path = REPORTS_DIR / "model_comparison_nested_cv.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved chart: {output_path}")


def main():
    X, y = load_data()

    # Keep the final test set separate from model comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Final test samples reserved: {len(X_test)}")
    print(f"Outer folds: {OUTER_FOLDS}")
    print(f"Inner folds: {INNER_FOLDS}")

    feature_sets = {
        "Original 4": [
            "tenure",
            "MonthlyCharges",
            "Contract",
            "InternetService",
        ],
        "All Features": X_train.columns.tolist(),
    }

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for feature_set_name, features in feature_sets.items():
        X_subset = X_train[features].copy()

        print(f"\n{'=' * 65}")
        print(f"FEATURE SET: {feature_set_name}")
        print(f"{'=' * 65}")

        for model_name, model in models.items():
            print(f"\nEvaluating: {model_name}")

            result = evaluate_configuration(
                name=model_name,
                feature_set=feature_set_name,
                model=model,
                X=X_subset,
                y=y_train,
            )

            results.append(result)

            print(
                f"  ROC-AUC: {result['roc_auc']:.4f}\n"
                f"  Average Precision: {result['average_precision']:.4f}\n"
                f"  Precision: {result['precision']:.4f}\n"
                f"  Recall: {result['recall']:.4f}\n"
                f"  F1: {result['f1']:.4f}\n"
                f"  Mean threshold: {result['mean_threshold']:.3f}\n"
                f"  Evaluation time: "
                f"{result['evaluation_time_seconds']:.2f}s"
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="roc_auc",
        ascending=False,
    )

    csv_path = REPORTS_DIR / "model_comparison_nested_cv.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 85)
    print("FINAL NESTED CROSS-VALIDATION RESULTS")
    print("=" * 85)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "roc_auc": "{:.4f}".format,
                "average_precision": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "mean_threshold": "{:.3f}".format,
                "threshold_std": "{:.3f}".format,
                "evaluation_time_seconds": "{:.2f}".format,
            },
        )
    )

    print(f"\nSaved results: {csv_path}")

    plot_results(results_df)

    print(
        "\nThe final held-out test set was not used for model "
        "selection or threshold tuning."
    )


if __name__ == "__main__":
    main()
