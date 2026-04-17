import os
import subprocess

import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, Field, validator
from sklearn.preprocessing import StandardScaler
import uvicorn

# Try to import torch for PyTorch model support; optional for serving
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ Warning: PyTorch not available. PyTorch models cannot be loaded.")

app = FastAPI()


# PyTorch Neural Network Model for Regression (needed for unpickling even if torch isn't installed)
if TORCH_AVAILABLE:
    class TipPredictionNN(nn.Module):
        def __init__(self, input_size, hidden_size=64):
            super(TipPredictionNN, self).__init__()
            # Input layer -> First hidden layer
            self.fc1 = nn.Linear(input_size, hidden_size)
            # First hidden layer -> Second hidden layer
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            # Second hidden layer -> Output layer
            self.fc3 = nn.Linear(hidden_size, 1)
            # ReLU activation
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
else:
    # Placeholder class for unpickling if torch not available
    class TipPredictionNN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not installed. Cannot use PyTorch models.")


# Model and scaler loading
numeric_features: List[str] = []
scaler = StandardScaler()
classification_model: Optional[Any] = None
regression_model: Optional[Any] = None
global prediction_id
prediction_id = 0


# Load numeric features, scaler, and model from disk
# Use environment variables (set in docker-compose.yml) or fall back to local paths
numeric_features_path = os.environ.get(
    "NUMERIC_FEATURES_PATH", "data/processed/numeric_features.joblib"
)
scaler_path = os.environ.get("SCALER_PATH", "data/processed/scaler.joblib")
classification_model_path = os.environ.get(
    "CLASSIFICATION_MODEL_PATH", "data/processed/classification_model.joblib"
)
regression_model_path = os.environ.get(
    "REGRESSION_MODEL_PATH", "data/processed/regression_model.joblib"
)

print(f"Looking for numeric features at: {numeric_features_path}")
if os.path.exists(numeric_features_path):
    numeric_features = joblib.load(numeric_features_path)
    print(f"✓ Loaded numeric features: {numeric_features}")
else:
    print(f"✗ Warning: numeric_features.joblib not found at {numeric_features_path}")

print(f"Looking for scaler at: {scaler_path}")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f"✓ Scaler loaded from {scaler_path}")
else:
    print(f"✗ Warning: scaler.joblib not found at {scaler_path}")

print(f"Looking for classification model at: {classification_model_path}")
if os.path.exists(classification_model_path):
    classification_model = joblib.load(classification_model_path)
    print(f"✓ Classification model loaded from {classification_model_path}")
    if hasattr(classification_model, "n_features_in_"):
        print(
            f"  Classification model expects {classification_model.n_features_in_} features."
        )
else:
    print(
        f"✗ ERROR: classification_model.joblib not found at {classification_model_path}"
    )
    classification_model = None

print(f"Looking for regression model at: {regression_model_path}")
if os.path.exists(regression_model_path):
    try:
        regression_model = joblib.load(regression_model_path)
        print(f"✓ Regression model loaded from {regression_model_path}")
        if hasattr(regression_model, "n_features_in_"):
            print(f"  Regression model expects {regression_model.n_features_in_} features.")
    except Exception as e:
        print(f"✗ ERROR loading regression model: {type(e).__name__}: {str(e)[:100]}")
        print("  Continuing without regression model...")
        regression_model = None
else:
    print(f"✗ Warning: regression model not found at {regression_model_path}")
    regression_model = None

# Verify at least one model is loaded
if not classification_model and not regression_model:
    print("\n✗ CRITICAL ERROR: Neither classification nor regression model could be loaded!")
    print("  Make sure the notebook has been executed to generate the model files.")
else:
    print("\n✓ At least one model loaded successfully.")


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        example={
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
            "pickup_hour": 14.0,
        },
        description="All features required by the model. Note: tip_amount is NOT included (it's the target variable being predicted)."
    )

    @validator("features", pre=True)
    def validate_features(cls, features):
        if not isinstance(features, dict):
            raise ValueError("features must be an object with numeric values")

        if "fare_amount" in features:
            fare = features["fare_amount"]
            if not isinstance(fare, (int, float)):
                raise ValueError("fare_amount must be numeric")
            if fare <= 0:
                raise ValueError("fare_amount must be positive")
            if fare > 1000:
                raise ValueError("fare_amount is unrealistically large")

        if "trip_distance" in features:
            distance = features["trip_distance"]
            if not isinstance(distance, (int, float)):
                raise ValueError("trip_distance must be numeric")
            if distance <= 0:
                raise ValueError("trip_distance must be positive")
            if distance > 500:
                raise ValueError("trip_distance is unrealistically large")

        if "pickup_hour" in features:
            hour = features["pickup_hour"]
            if not isinstance(hour, (int, float)):
                raise ValueError("pickup_hour must be numeric")
            if hour < 0 or hour > 23:
                raise ValueError("pickup_hour must be between 0 and 23")
        return features


class TestResult(BaseModel):
    test_name: str
    passed: bool


class TestSummary(BaseModel):
    total_tests: int
    passed: int
    failed: int
    all_passed: bool
    tests: List[TestResult]


@app.get("/")
def root():
    return {
        "message": "API is working"
    }  # Simple root endpoint to verify API is up and running


@app.post("/predict")
def predict(request: PredictionRequest):

    if regression_model is None:
        print("Error: Regression model is not loaded, cannot perform prediction.")
        raise HTTPException(
            status_code=503,
            detail="Regression model not loaded - model file not found or failed to load",
        )

    # Check if model feature count matches our current features
    if hasattr(regression_model, "n_features_in_") and regression_model.n_features_in_ != len(
        numeric_features
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Model feature mismatch",
                "message": f"Model was trained with {regression_model.n_features_in_} features but current configuration uses {len(numeric_features)} features. The model needs to be retrained with the corrected feature set (excluding tip_amount to prevent data leakage).",
                "model_features": regression_model.n_features_in_,
                "current_features": len(numeric_features),
                "required_features": numeric_features,
            },
        )

    print(f"Received prediction request with features: {request.features}")

    missing_features = [f for f in numeric_features if f not in request.features]
    if missing_features:
        print(
            f"Error: Missing required features: {missing_features}. Required features are: {numeric_features}"
        )
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing_features,
                "required_features": numeric_features,
            },
        )

    input_data = []
    for feature in numeric_features:
        value = request.features[feature]
        # Validate individual feature values by using the Pydantic validator logic defined above
        PredictionRequest(features={feature: value})
        input_data.append(float(value))

    input_scaled = scaler.transform([input_data])
    prediction = regression_model.predict(input_scaled)[0]
    global prediction_id
    prediction_id += 1
    return {
        "prediction": float(prediction),
        "prediction_id": prediction_id,
    }


@app.post("/predict/batch")  # Accept up to 100 records for batch prediction
def predict_batch(requests: List[PredictionRequest]):
    if regression_model is None:
        raise HTTPException(
            status_code=503,
            detail="Regression model not loaded - model file not found or failed to load",
        )

    # Check if model feature count matches our current features
    if hasattr(regression_model, "n_features_in_") and regression_model.n_features_in_ != len(
        numeric_features
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Model feature mismatch",
                "message": f"Model was trained with {regression_model.n_features_in_} features but current configuration uses {len(numeric_features)} features. The model needs to be retrained with the corrected feature set (excluding tip_amount to prevent data leakage).",
                "model_features": regression_model.n_features_in_,
                "current_features": len(numeric_features),
                "required_features": numeric_features,
            },
        )

    if len(requests) > 100:
        raise HTTPException(
            status_code=422, detail="Batch size cannot exceed 100 records"
        )

    input_data = []
    for req in requests:
        missing_features = [f for f in numeric_features if f not in req.features]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": missing_features,
                    "required_features": numeric_features,
                },
            )
        row = []
        for feature in numeric_features:
            value = req.features[feature]
            PredictionRequest(features={feature: value})
            row.append(float(value))
        input_data.append(row)

    input_scaled = scaler.transform(input_data)
    predictions = regression_model.predict(input_scaled)
    global prediction_id
    results = []
    for pred in predictions:
        prediction_id += 1
        results.append(
            {
                "prediction": float(pred),
                "prediction_id": prediction_id,
            }
        )
    return results


@app.get("/health")
def health():
    feature_mismatch = False
    if regression_model is not None and hasattr(regression_model, "n_features_in_"):
        feature_mismatch = regression_model.n_features_in_ != len(numeric_features)

    return {
        "status": "ok",
        "regression_model_loaded": regression_model is not None,
        "classification_model_loaded": classification_model is not None,
        "scaler_fitted": bool(numeric_features),
        "feature_count_match": not feature_mismatch,
        "regression_model_features": regression_model.n_features_in_
        if regression_model and hasattr(regression_model, "n_features_in_")
        else None,
        "current_features": len(numeric_features),
        "data_leakage_fixed": "tip_amount" not in numeric_features,
    }


@app.get("/model/info")
def model_info():
    feature_mismatch = False
    if regression_model is not None and hasattr(regression_model, "n_features_in_"):
        feature_mismatch = regression_model.n_features_in_ != len(numeric_features)

    return {
        "regression_model_loaded": regression_model is not None,
        "classification_model_loaded": classification_model is not None,
        "required_features": numeric_features,
        "feature_count": len(numeric_features),
        "regression_model_feature_count": regression_model.n_features_in_
        if regression_model and hasattr(regression_model, "n_features_in_")
        else None,
        "classification_model_feature_count": classification_model.n_features_in_
        if classification_model and hasattr(classification_model, "n_features_in_")
        else None,
        "feature_mismatch": feature_mismatch,
        "data_leakage_status": "fixed"
        if "tip_amount" not in numeric_features
        else "present",
    }


@app.get("/tests/run", response_model=TestSummary)
def run_tests():
    """Run all tests and return their results"""
    try:
        result = subprocess.run(
            [
                "py",
                "-3.12",
                "-m",
                "pytest",
                "test_app.py",
                "-v",
                "--tb=short",
                "--json-report",
                "--json-report-file=test_report.json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Parse pytest output to extract test results
        output_lines = result.stdout.split("\n")
        tests = []
        passed_count = 0
        failed_count = 0

        for line in output_lines:
            if "PASSED" in line:
                # Extract test name
                test_name = (
                    line.split("::")[1].split(" ")[0] if "::" in line else "Unknown"
                )
                tests.append(TestResult(test_name=test_name, passed=True))
                passed_count += 1
            elif "FAILED" in line:
                test_name = (
                    line.split("::")[1].split(" ")[0] if "::" in line else "Unknown"
                )
                tests.append(TestResult(test_name=test_name, passed=False))
                failed_count += 1

        total = passed_count + failed_count
        all_passed = failed_count == 0 and total > 0

        # If no tests were parsed, try to get from pytest summary line
        if total == 0:
            for line in output_lines:
                if "passed" in line or "failed" in line:
                    # Try to extract counts from summary line like "5 passed in 0.45s"
                    passed_match = re.search(r"(\d+) passed", line)
                    failed_match = re.search(r"(\d+) failed", line)
                    if passed_match:
                        passed_count = int(passed_match.group(1))
                    if failed_match:
                        failed_count = int(failed_match.group(1))
                    total = passed_count + failed_count
                    all_passed = failed_count == 0 and total > 0
                    break

        return TestSummary(
            total_tests=total,
            passed=passed_count,
            failed=failed_count,
            all_passed=all_passed,
            tests=tests,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Test execution timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running tests: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    print("Running in production mode")
