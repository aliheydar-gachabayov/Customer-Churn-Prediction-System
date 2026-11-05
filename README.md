# Customer Churn Prediction System

An end-to-end machine learning application that predicts customer churn for a telecommunications company. The project combines a scikit-learn classification pipeline, a FastAPI REST API, an interactive web interface, automated tests, and Docker containerization.

The goal is to identify customers who are likely to leave a service and provide probability-based churn predictions.

## Features

- Compares Logistic Regression and Random Forest using stratified 5-fold cross-validation.
- Evaluates models using ROC-AUC.
- Uses 19 customer features for prediction.
- Automatically selects a classification threshold using out-of-fold predictions.
- Provides a REST API built with FastAPI.
- Validates customer input, including dependencies between subscribed services.
- Includes an interactive web interface.
- Includes automated API tests.
- Supports Docker-based deployment with a container health check.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Backend | FastAPI, Uvicorn |
| Validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest, HTTPX |
| Containerization | Docker |

## Dataset

The project uses the public [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The dataset contains telecommunications customer information, including demographics, subscribed services, contract details, and billing information.

The target variable is `Churn`, indicating whether a customer left the company.

The model uses 19 input features and excludes the customer identifier and target column.

## Machine Learning Approach

### 1. Data preprocessing

The preprocessing pipeline includes:

- Conversion of `TotalCharges` to numeric values.
- Replacement of blank `TotalCharges` values with zero before the pipeline (the dataset's blank entries correspond to zero-tenure customers).
- Handling any remaining missing numerical values using median imputation.
- Handling missing categorical values using most-frequent imputation.
- Standardization of numerical features.
- One-hot encoding of categorical features.

Preprocessing is implemented using scikit-learn pipelines and a `ColumnTransformer` to ensure transformations are fitted on training data.

### 2. Model comparison

Two classification algorithms were evaluated:

- Logistic Regression
- Random Forest

The models were compared using stratified 5-fold cross-validation and ROC-AUC.

| Model configuration | Mean CV ROC-AUC |
|---|---:|
| Logistic Regression — 19 features | **0.8462** |
| Logistic Regression — 4 features | 0.8345 |
| Random Forest — 19 features | 0.8205 |
| Random Forest — 4 features | 0.7941 |

Logistic Regression using all 19 features achieved the highest mean cross-validation ROC-AUC and was selected.

### 3. Decision threshold optimization

Instead of automatically using the default classification threshold of 0.50, the project selects a threshold using out-of-fold prediction probabilities generated from the training data.

The threshold selection process uses:

- `StratifiedKFold` with 5 folds.
- `cross_val_predict` to generate out-of-fold probabilities.
- Seven candidate thresholds: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, and 0.60.
- F1-score to select the best threshold.

The selected threshold was **0.35**, with an out-of-fold F1-score of 0.6353.

Threshold selection is performed automatically during training. The selected threshold and trained model are saved together in `models/churn_model.joblib`, allowing the API to use the same threshold during prediction.

The held-out test set is not used for threshold selection.

The out-of-fold F1-score is used to select the threshold, so it should be interpreted as a threshold-selection score rather than an independent estimate of generalization performance. The held-out test metrics provide the final evaluation.

### Threshold trade-off analysis

Run:

```bash
python threshold_analysis.py
```

This script recreates the training partition and generates out-of-fold probabilities with stratified 5-fold cross-validation. It compares precision, recall, specificity, F1, false positives, and false negatives over thresholds from 0.10 to 0.80.

Outputs:
- `reports/threshold_analysis.png`: metric and error-count curves.
- `reports/threshold_analysis.csv`: threshold-by-threshold metrics.

Threshold selection is based on out-of-fold training predictions only. The script also reports the saved threshold and the best F1 threshold for comparison. The best F1 threshold is not automatically the best business threshold: the relative costs of missed churners and unnecessary retention outreach should guide the final choice. The held-out test partition remains reserved for final evaluation.

## Model Evaluation

The selected Logistic Regression model was evaluated on a held-out test set representing 20% of the dataset.

| Metric | Test result |
|---|---:|
| ROC-AUC | **0.8421** |
| Precision | 0.5432 |
| Recall | 0.7059 |
| F1-score | 0.6140 |
| Decision threshold | 0.35 |

### Confusion Matrix

| | Predicted churn | Predicted no churn |
|---|---:|---:|
| Actual churn | 264 | 110 |
| Actual no churn | 222 | 813 |

The selected threshold prioritizes identifying more customers who are likely to churn. This increases recall but also generates more false-positive predictions.

## Application Architecture

The application consists of three main components:

1. **Frontend:** A web interface for entering customer information and displaying predictions.
2. **FastAPI backend:** Validates input, processes prediction requests, and returns results.
3. **Machine learning model:** A trained scikit-learn pipeline loaded from a Joblib artifact.

The frontend sends customer information to the `/predict` endpoint. The API validates the data, obtains the churn probability, applies the selected threshold, and returns the prediction.

## Project Structure

```text
customer-churn-ml/
│
├── app/
│   └── main.py
│
├── frontend/
│   └── index.html
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.joblib
│
├── test/
│   └── test_api.py
│
├── analyze_features.py
├── analyze_thresholds.py
├── experiment_models.py
├── inspect_data.py
├── predict.py
├── train.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

## Getting Started

### Prerequisites

- Python 3.14
- pip

### 1. Clone the repository

```bash
git clone https://github.com/aliheydar-gachabayov/Customer-Churn-Prediction-System.git

cd Customer-Churn-Prediction-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements-dev.txt
```

### 4. Run the application

```bash
uvicorn app.main:app --reload
```

Open the following URLs:

- Web application: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

The trained model artifact is included in `models/`, so retraining is not required to run the application.

### 5. Run automated tests

```bash
python -m pytest -v
```

The test suite currently contains 9 API tests covering successful predictions, response structure, numerical validation, and service-combination validation.

### 6. Reproduce model experiments and training

To compare the model configurations:

```bash
python experiment_models.py
```

To train the selected Logistic Regression model, automatically optimize its decision threshold, evaluate it, and save the artifact:

```bash
python train.py
```

The trained model and selected threshold are saved to:

```text
models/churn_model.joblib
```

## Docker

The project includes a Dockerfile and a `.dockerignore` configuration.

### Build the image

```bash
docker build -t customer-churn-api .
```

### Run the container

```bash
docker run -d -p 8000:8000 --name customer-churn-container customer-churn-api
```

### Check container status

```bash
docker ps
```

The Docker image includes a health check that periodically calls the application's `/health` endpoint.

Open http://localhost:8000 to access the application.

### Stop the container

```bash
docker rm -f customer-churn-container
```

## API Usage

### Endpoint

`POST /predict`

The endpoint accepts 19 customer attributes.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0
  }'
```

Example response (illustrative values):

```json
{
  "churn_probability": 0.775,
  "threshold": 0.35,
  "prediction": 1,
  "risk_level": "High"
}
```

### Response fields

| Field | Description |
|---|---|
| `churn_probability` | Estimated probability of customer churn |
| `threshold` | Classification threshold used by the model |
| `prediction` | `1` indicates likely churn; `0` indicates unlikely churn |
| `risk_level` | `High` or `Low` based on the classification result |

Invalid input returns HTTP 422 with validation details.

## Model evaluation and explainability

Install the development dependencies (including Matplotlib and SHAP):

```bash
python -m pip install -r requirements-dev.txt
```

### Evaluate the saved model

Run:

```bash
python evaluate_model.py
```

The script recreates the original stratified 80/20 split with `random_state=42`, loads the saved model and decision threshold, and evaluates only the held-out test partition. It reports:

- ROC-AUC and average precision.
- Precision, recall, specificity, F1, accuracy, and balanced accuracy.
- Brier score and log loss for probability quality.
- Confusion-matrix counts and a classification report.

It also generates a four-panel dashboard with ROC, precision-recall, confusion-matrix, and calibration plots.

### Explain predictions with SHAP

Run:

```bash
python explain_model.py
```

The script uses SHAP's `LinearExplainer` for the fitted Logistic Regression classifier after applying the pipeline's fitted preprocessing step. It samples a reproducible background set from the training partition and explains held-out test customers.

Generated outputs include:

- `shap_summary.png`: global SHAP beeswarm summary.
- `shap_importance.png`: global mean absolute SHAP importance.
- `shap_waterfall.png`: explanation for one held-out customer.
- `shap_feature_importance.csv`: ranked mean absolute SHAP values.

**How to interpret SHAP:** positive values move the classifier's raw score toward the churn class; negative values move it away from churn. For this linear classifier, the explanation is in log-odds space—not a direct percentage-point change in churn probability. SHAP describes the model's learned associations, not causal effects.

Both scripts write generated files to the local `reports/` directory. The held-out test data is used for final reporting and explanations, not for choosing a new threshold or tuning the model.

## Limitations

- The model uses a public historical dataset, which may not represent current customer behavior.
- Classification performance depends on the quality and representativeness of the dataset.
- The selected decision threshold prioritizes recall and produces false-positive predictions.
- The application is an educational portfolio project, not a production-ready customer retention system.
- Model performance should be monitored and periodically reevaluated before real-world deployment.

## Future Improvements

- Add more extensive model evaluation and calibration.
- Evaluate business costs associated with false positives and false negatives.
- Add explainability techniques such as SHAP.
- Introduce CI workflows for automated testing.
- Deploy the application to a cloud platform.
- Add model monitoring and data drift detection.

---

**Author:** [Aliheydar Gachabayov](https://www.linkedin.com/in/aliheydar-gachabayov)

**GitHub:** [aliheydar-gachabayov](https://github.com/aliheydar-gachabayov)
