
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    customer = {
        "tenure": 12,
        "MonthlyCharges": 75.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert result["risk_level"] in ["High", "Low"]


def test_invalid_customer():
    customer = {
        "tenure": -5,
        "MonthlyCharges": 75.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
