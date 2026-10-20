
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService"
]

X = df[features]
y = df["Churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

def make_preprocessor():
    return ColumnTransformer([
        (
            "numeric",
            StandardScaler(),
            ["tenure", "MonthlyCharges"]
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Contract", "InternetService"]
        )
    ])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    results[name] = scores.mean()

    print(f"\n{name}")
    print(f"CV ROC-AUC: {scores.mean():.4f}")
    print(f"CV Std: {scores.std():.4f}")

best_name = max(results, key=results.get)

print(f"\nSelected model: {best_name}")

best_model = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", models[best_name])
])

best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

print("\nFinal Test Results")
print(classification_report(y_test, predictions))
print(
    f"Test ROC-AUC: "
    f"{roc_auc_score(y_test, probabilities):.4f}"
)

model_path = MODEL_DIR / "churn_model.joblib"
joblib.dump(best_model, model_path)

print(f"\nSaved model: {model_path}")
