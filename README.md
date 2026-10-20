# Customer Churn Prediction

A small machine learning project that predicts whether a telecom customer is likely to leave (churn). I trained a model with scikit-learn, wrapped it in a FastAPI service, and added a simple web page where you can enter a customer's details and see the predicted risk.

This is a learning project, not a production system.

## What it does

- Trains and compares two models (Logistic Regression and Random Forest) using 5-fold cross-validation.
- Picks the model with the best cross-validated ROC-AUC and evaluates it once on a held-out test set.
- Serves predictions through a REST API (`/predict`) with input validation.
- Includes a basic web form, automated API tests, and a Dockerfile.

## Dataset

I used the public [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (about 7,000 customers). About a quarter of the customers in it churned, so the classes are imbalanced.

I only used four features to keep the model and the form simple:

| Feature | Type |
|---|---|
| `tenure` (months with the company) | numeric |
| `MonthlyCharges` | numeric |
| `Contract` (month-to-month, one year, two year) | categorical |
| `InternetService` (DSL, fiber optic, none) | categorical |

## Results

| Model | CV ROC-AUC (5-fold) |
|---|---|
| Logistic Regression | [fill in from train.py output] |
| Random Forest | [fill in from train.py output] |

Selected model: **[fill in]**

Test set (20% held out): ROC-AUC **[fill in]**, plus precision/recall for churners **[fill in]**.

Run `python train.py` to reproduce these numbers.

## Project structure

```
.
├── app/
│   └── main.py          # FastAPI app
├── frontend/
│   └── index.html       # web form
├── data/                # Telco-Customer-Churn.csv
├── models/              # saved model (churn_model.joblib)
├── tests/               # API tests
├── train.py             # training and evaluation
├── predict.py           # predict for one example customer
├── Dockerfile
└── requirements.txt
```

(Adjust this to match your actual folders.)

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model (creates `models/churn_model.joblib`):

```bash
python train.py
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the web form, or http://127.0.0.1:8000/docs for the interactive API docs.

Run the tests:

```bash
pytest
```

### With Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## API example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 75.0, "Contract": "Month-to-month", "InternetService": "Fiber optic"}'
```

Response:

```json
{
  "churn_probability": 0.0,
  "prediction": 0,
  "risk_level": "Low"
}
```

(Replace the numbers with the real output you get.)

Invalid input, such as a negative tenure or an unknown contract type, returns a 422 error.

## Limitations

- Only four features are used. The dataset has about 20, so the model leaves out information that would probably help.
- Predictions use a fixed 0.5 threshold. Because churn is imbalanced, a different threshold might suit a real business better, depending on the cost of missing a churner versus contacting a loyal customer.
- The dataset is a public sample, so the results say little about real-world performance.
- `risk_level` is just High or Low from that same threshold.

## What I would improve next

- Try more features and check whether they actually improve the test score.
- Tune the decision threshold using precision/recall.
- Add a medium risk level.
- Add more tests and run them in CI.