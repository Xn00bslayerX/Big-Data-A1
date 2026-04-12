import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Sample complete feature set for testing
sample_features = {
    "VendorID": 1.0,
    "passenger_count": 2.0,
    "trip_distance": 5.0,
    "RatecodeID": 1.0,
    "PULocationID": 100.0,
    "DOLocationID": 200.0,
    "payment_type": 1.0,
    "fare_amount": 20.0,
    "extra": 0.5,
    "mta_tax": 0.5,
    "tip_amount": 0.0,  # Set to 0 for prediction
    "tolls_amount": 0.0,
    "improvement_surcharge": 0.3,
    "total_amount": 21.3,
    "congestion_surcharge": 2.5,
    "Airport_fee": 0.0,
    "trip_duration_minutes": 15.0,
    "trip_speed_mph": 20.0,
    "log_trip_distance": 1.609,
    "fare_per_mile": 4.0,
    "fare_per_minute": 1.333,
    "pickup_hour": 14.0
}

# Successful single prediction with valid input
def test_single_prediction():
    response = client.post("/predict", json={"features": sample_features})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert "model_version" in data
    assert "prediction_id" in data

# Successful batch prediction with valid input
def test_batch_prediction():
    response = client.post("/predict/batch", json=[
        {"features": sample_features},
        {"features": {**sample_features, "trip_distance": 3.0}}
    ])
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for prediction in data:
        assert "prediction" in prediction
        assert isinstance(prediction["prediction"], float)
        assert "model_version" in prediction
        assert "prediction_id" in prediction

# Reject invalid inputs. Multiple cases with missing fields, bad data types, and out-of-range values
def test_invalid_inputs():
    # Missing required features
    response = client.post("/predict", json={"features": {"trip_distance": 5.0}})
    assert response.status_code == 422

    # Non-numeric feature value
    invalid_features = sample_features.copy()
    invalid_features["trip_distance"] = "far" # type: ignore
    response = client.post("/predict", json={"features": invalid_features})
    assert response.status_code == 422

    # Negative trip distance
    invalid_features = sample_features.copy()
    invalid_features["trip_distance"] = -1.0
    response = client.post("/predict", json={"features": invalid_features})
    assert response.status_code == 422

    # Unrealistically large trip distance
    invalid_features = sample_features.copy()
    invalid_features["trip_distance"] = 1000.0
    response = client.post("/predict", json={"features": invalid_features})
    assert response.status_code == 422

    # Unrealistically large fare amount
    invalid_features = sample_features.copy()
    invalid_features["fare_amount"] = 2000.0
    response = client.post("/predict", json={"features": invalid_features})
    assert response.status_code == 422

    # Invalid pickup hour
    invalid_features = sample_features.copy()
    invalid_features["pickup_hour"] = 25.0
    response = client.post("/predict", json={"features": invalid_features})
    assert response.status_code == 422

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"