
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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


# ==========================================
# 1. LOAD DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

# Convert TotalCharges to numeric.
# Blank values belong to customers with zero tenure.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
).fillna(0)

# Encode target
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected target values detected.")

# ==========================================
# 2. DEFINE FEATURE SETS
# ==========================================

baseline_features = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService"
]

all_features = [
    column for column in df.columns
    if column not in ["customerID", "Churn"]
]

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X = df[all_features]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================================
# 4. CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}

# ==========================================
# 6. BUILD PREPROCESSING PIPELINE
# ==========================================

def make_pipeline(features, classifier):

    numerical = [
        column for column in features
        if column in numeric_features
    ]

    categorical = [
        column for column in features
        if column not in numeric_features
    ]

    transformers = []

    if numerical:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numerical
        ))

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical
        ))

    preprocessor = ColumnTransformer(transformers)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


# ==========================================
# 7. RUN EXPERIMENTS USING CROSS-VALIDATION
# ==========================================

feature_sets = {
    "Original 4 Features": baseline_features,
    "All 19 Features": all_features
}

results = []

for feature_name, features in feature_sets.items():

    print("\n" + "=" * 60)
    print(f"FEATURE SET: {feature_name}")
    print("=" * 60)

    for model_name, classifier in models.items():

        pipeline = make_pipeline(
            features,
            classifier
        )

        scores = cross_val_score(
            pipeline,
            X_train[features],
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        results.append({
            "Feature Set": feature_name,
            "Model": model_name,
            "CV Mean": scores.mean(),
            "CV Std": scores.std()
        })

        print(f"\nModel: {model_name}")
        print(f"CV ROC-AUC: {scores.mean():.4f}")
        print(f"CV Std: {scores.std():.4f}")


# ==========================================
# 8. SELECT BEST CONFIGURATION
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV Mean",
    ascending=False
)

print("\n" + "=" * 60)
print("FINAL CROSS-VALIDATION COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

best_result = results_df.iloc[0]

best_feature_name = best_result["Feature Set"]
best_model_name = best_result["Model"]

best_features = feature_sets[best_feature_name]
best_classifier = models[best_model_name]

print("\nSelected configuration:")
print(f"Features: {best_feature_name}")
print(f"Model: {best_model_name}")


# ==========================================
# 9. FINAL TEST EVALUATION
# ==========================================

final_pipeline = make_pipeline(
    best_features,
    best_classifier
)

final_pipeline.fit(
    X_train[best_features],
    y_train
)

test_probabilities = final_pipeline.predict_proba(
    X_test[best_features]
)[:, 1]

test_predictions = final_pipeline.predict(
    X_test[best_features]
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)

print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

print(f"Test ROC-AUC: {test_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions
))
