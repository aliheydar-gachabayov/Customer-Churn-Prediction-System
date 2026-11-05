
from fastapi.testclient import TestClient

from app.main import app, threshold


client = TestClient(app)


# ==========================================
# VALID CUSTOMER TEST DATA
# ==========================================

def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0
    }


# ==========================================
# BASIC ENDPOINT TESTS
# ==========================================

def test_home():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ==========================================
# PREDICTION TESTS
# ==========================================

def test_prediction():
    response = client.post(
        "/predict",
        json=valid_customer()
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["churn_probability"] <= 1
    assert result["threshold"] == threshold
    assert result["prediction"] in [0, 1]
    assert result["risk_level"] == (
        "High" if result["prediction"] == 1 else "Low"
    )


def test_prediction_response_structure():
    response = client.post(
        "/predict",
        json=valid_customer()
    )

    result = response.json()

    expected_fields = {
        "churn_probability",
        "threshold",
        "prediction",
        "risk_level"
    }

    assert set(result.keys()) == expected_fields


# ==========================================
# INPUT VALIDATION TESTS
# ==========================================

def test_invalid_tenure():
    customer = valid_customer()
    customer["tenure"] = -5

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422


def test_invalid_internet_service_combination():
    customer = valid_customer()

    customer["InternetService"] = "No"
    customer["OnlineSecurity"] = "Yes"

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422


def test_valid_no_internet_combination():
    customer = valid_customer()

    customer["InternetService"] = "No"

    customer["OnlineSecurity"] = "No internet service"
    customer["OnlineBackup"] = "No internet service"
    customer["DeviceProtection"] = "No internet service"
    customer["TechSupport"] = "No internet service"
    customer["StreamingTV"] = "No internet service"
    customer["StreamingMovies"] = "No internet service"

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200


def test_invalid_phone_service_combination():
    customer = valid_customer()

    customer["PhoneService"] = "No"
    customer["MultipleLines"] = "Yes"

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422


def test_valid_no_phone_service_combination():
    customer = valid_customer()

    customer["PhoneService"] = "No"
    customer["MultipleLines"] = "No phone service"

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200
