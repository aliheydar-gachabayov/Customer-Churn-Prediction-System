"""Explain the saved logistic-regression churn model with SHAP.

Run: python explain_model.py
Outputs are written to reports/ (created automatically).

SHAP values explain the classifier's raw decision score (log-odds), not
the final churn probability or the API's thresholded class label.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
REPORT_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
BACKGROUND_SIZE = 200
EXPLANATION_SIZE = 500


def load_data():
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

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test


def as_dense(matrix):
    """Convert transformed features to a NumPy array for SHAP."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def main():
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}. Run python train.py first."
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = float(artifact["threshold"])

    if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError("Expected a pipeline with preprocessor and classifier steps.")

    X_train, X_test = load_data()
    rng = np.random.RandomState(RANDOM_STATE)
    background = X_train.sample(
        n=min(BACKGROUND_SIZE, len(X_train)),
        random_state=RANDOM_STATE,
    )
    explain_rows = X_test.sample(
        n=min(EXPLANATION_SIZE, len(X_test)),
        random_state=RANDOM_STATE,
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    background_matrix = as_dense(preprocessor.transform(background))
    explanation_matrix = as_dense(preprocessor.transform(explain_rows))
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(classifier, background_matrix)
    shap_explanation = explainer(explanation_matrix)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Global importance: mean absolute SHAP value across held-out examples.
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_explanation.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    importance_path = REPORT_DIR / "shap_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    shap.plots.beeswarm(
        shap_explanation,
        max_display=20,
        show=False,
    )
    plt.title("SHAP summary — held-out customers")
    plt.tight_layout()
    summary_path = REPORT_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    shap.plots.bar(
        shap_explanation,
        max_display=20,
        show=False,
    )
    plt.title("Mean absolute SHAP importance")
    plt.tight_layout()
    bar_path = REPORT_DIR / "shap_importance.png"
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()

    # Explain one held-out customer, selected reproducibly.
    local_explanation = shap_explanation[0]
    shap.plots.waterfall(local_explanation, max_display=15, show=False)
    plt.title("SHAP explanation — one held-out customer")
    plt.tight_layout()
    waterfall_path = REPORT_DIR / "shap_waterfall.png"
    plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close()

    local_probability = float(
        model.predict_proba(explain_rows.iloc[[0]])[0, 1]
    )
    local_prediction = int(local_probability >= threshold)

    print("SHAP EXPLANATIONS")
    print(f"Explained held-out customers: {len(explain_rows)}")
    print(f"Background customers:         {len(background)}")
    print(f"Decision threshold:           {threshold:.3f}")
    print(f"Example churn probability:    {local_probability:.3f}")
    print(f"Example predicted class:      {local_prediction}")
    print("\nTop 15 features by mean absolute SHAP value:")
    print(importance.head(15).to_string(index=False))
    print("\nSaved files:")
    print(f"- {summary_path}")
    print(f"- {bar_path}")
    print(f"- {waterfall_path}")
    print(f"- {importance_path}")
    print(
        "\nInterpretation: SHAP values are in log-odds space for the "
        "positive (churn) class. Positive values push the model score "
        "toward churn; negative values push it away from churn."
    )


if __name__ == "__main__":
    main()
