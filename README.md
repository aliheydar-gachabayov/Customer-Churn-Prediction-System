# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to leave a service.

The project uses Python and Scikit-learn to train a Logistic Regression model. It also includes a FastAPI backend and a simple web interface where users can enter customer information and receive a churn prediction.

## Project Overview

Customer churn occurs when a customer stops using a company's services.

The goal of this project is to build a classification model that estimates the probability of customer churn based on customer information, subscribed services, contract details, and billing information.

### Main Features

- Customer churn prediction using Logistic Regression.
- Data preprocessing with Scikit-learn.
- Handling missing numerical and categorical values.
- Numerical feature scaling and categorical encoding.
- Model evaluation using standard classification metrics.
- Saving and loading the trained model.
- REST API built with FastAPI.
- Interactive web interface.
- Input validation using Pydantic.
- Automated API tests using Pytest.
- GitHub Actions workflow.

## Technologies

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas |
| Model Storage | Joblib |
| Backend | FastAPI, Uvicorn |
| Input Validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest, HTTPX |
| Version Control | Git, GitHub |
| CI | GitHub Actions |

## Dataset

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The dataset contains 7,043 customer records and 21 columns.

It includes information such as:

- Customer demographics
- Subscribed services
- Contract type
- Payment method
- Monthly charges
- Total charges

The target column is `Churn`:

- `Yes` means the customer left the company.
- `No` means the customer stayed.

The model uses 19 input features. The customer ID and target column are excluded from the input features.

Place the dataset at:

```text
data/Telco-Customer-Churn.csv
```

## Machine Learning Workflow

The training process consists of the following steps:

1. Load the customer dataset.
2. Convert `TotalCharges` into numerical values.
3. Separate input features (`X`) from the target (`y`).
4. Split the dataset into training and testing sets.
5. Preprocess numerical and categorical features.
6. Train a Logistic Regression classifier.
7. Evaluate the model on the test set.
8. Save the trained model and evaluation metrics.

### Data Preprocessing

The project uses a Scikit-learn preprocessing pipeline.

**Numerical features:**
- Missing values are replaced with the median.
- Features are standardized using `StandardScaler`.

**Categorical features:**
- Missing values are replaced with the most frequent category.
- Categories are converted into numerical features using `OneHotEncoder`.

The preprocessing steps are combined using `ColumnTransformer` and `Pipeline`.

This ensures that preprocessing is fitted on the training data and applied consistently to the test data and future predictions.

### Training and Testing

The dataset is divided into:

- 80% training data.
- 20% testing data.

The split uses stratification to preserve the proportion of churn and non-churn customers.

A random state of `42` is used to make the split reproducible.

### Classification Model

The project uses Logistic Regression for binary classification.

The model estimates the probability that a customer will churn.

A classification threshold of `0.5` is used:

- Probability greater than or equal to 0.5: predict churn.
- Probability below 0.5: predict no churn.

## Model Performance

The latest training run produced the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1-score | 60.40% |
| ROC-AUC | 0.8419 |

### Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 926 | 109 |
| Actual Churn | 165 | 209 |

The model correctly classified 926 customers who stayed and 209 customers who churned.

It also missed 165 actual churners. This indicates that improving churn detection, particularly recall, is an area for future work.

These results describe the current model and are not a guarantee of performance on new customer data.

## Project Structure

```text
customer-churn-ml/
│
├── app/
│   └── main.py
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── frontend/
│   └── index.html
│
├── models/
│   └── churn_model.joblib
│
├── reports/
│   └── final_test_metrics.json
│
├── test/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── train.py
├── predict.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── README.md
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

### 4. Train the model

Make sure the dataset is located at `data/Telco-Customer-Churn.csv`.

Run:

```bash
python train.py
```

The script trains the Logistic Regression model and saves:

```text
models/churn_model.joblib
reports/final_test_metrics.json
```

### 5. Make a prediction

Run:

```bash
python predict.py
```

The script uses an example customer and displays:

- Churn probability
- Decision threshold
- Churn prediction

### 6. Run the web application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open the following addresses:

- Web application: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

The application loads the trained model from `models/churn_model.joblib`.

You can enter customer information through the web interface and receive a prediction.

### 7. Run automated tests

Run:

```bash
python -m pytest -v
```

The test suite contains nine API tests covering:

- Homepage and health endpoints.
- Successful customer predictions.
- Prediction response structure.
- Invalid customer information.
- Phone service validation.
- Internet service validation.

GitHub Actions also runs model training and API tests automatically when changes are pushed or submitted through a pull request targeting `main`.

## API

The application provides three endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the web interface |
| GET | `/health` | Checks API health |
| POST | `/predict` | Predicts customer churn |

The prediction endpoint accepts customer information, validates the input, and returns the predicted churn probability and classification.

## Future Improvements

Some possible improvements include:

- Experimenting with additional classification algorithms.
- Improving recall while maintaining reasonable precision.
- Comparing different classification thresholds.
- Adding model explainability.
- Improving the frontend.
- Adding more prediction and integration tests.

## Disclaimer

This project is an educational machine learning application. Its predictions should not be treated as validated business decisions or as a substitute for a properly evaluated customer retention system.