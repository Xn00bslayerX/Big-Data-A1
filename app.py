import os
import subprocess
import json
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, validator
from sklearn.preprocessing import StandardScaler
import numpy as np
import polars as pl
import uvicorn

app = FastAPI()

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow_client = MlflowClient()

# All model-related variables are defined at the module level for better visibility and management. This also makes Pylance's static analysis more effective in identifying issues with model loading and usage.
model_name = "TaxiTipRandomForestRegressor"
model: Optional[Any] = None
loaded_model_uri: Optional[str] = None
loaded_model_version: Optional[str] = None
loaded_model_stage: Optional[str] = None
numeric_features: List[str] = []
scaler = StandardScaler()
global prediction_id
prediction_id = 0

processed_parquet_file = "data/processed/cleaned_trips.parquet"
if os.path.exists(processed_parquet_file):
    df = pl.read_parquet(processed_parquet_file)
    numeric_features = [
        col
        for col in df.columns
        if df[col].dtype.is_numeric() and col not in {"high_tip", "tip_amount"}  # Exclude tip_amount to prevent data leakage since we're predicting it
    ]
    scaler.fit(df[numeric_features].to_pandas())
    print (f"Scaler fitted on features: {numeric_features}")
else:
    print("Warning: Processed data not found, scaler not fitted properly.")


def load_registered_model(name: str) -> Optional[Any]:
    global loaded_model_uri, loaded_model_version, loaded_model_stage

    try:
        production_versions = mlflow_client.get_latest_versions(name, stages=["Production"])
        source_version = production_versions[0] if production_versions else None

        if source_version is None:
            all_versions = mlflow_client.get_latest_versions(name)
            if not all_versions:
                raise ValueError(f"No registered versions found for model '{name}'")
            source_version = sorted(all_versions, key=lambda v: int(v.version), reverse=True)[0]

        loaded_model_version = source_version.version
        loaded_model_stage = source_version.current_stage
        loaded_model_uri = f"models:/{name}/{loaded_model_version}"
        print (f"Loading model '{name}' version {loaded_model_version} from stage '{loaded_model_stage}' at URI: {loaded_model_uri}")
        return mlflow.sklearn.load_model(loaded_model_uri)
    except Exception as exc:
        print(f"Error loading registered model '{name}': {exc}")
        return None

model = load_registered_model(model_name)
if model is not None:
    print(f"Model '{model_name}' loaded successfully.")
    print(f"Model expects {model.n_features_in_} features.")
    print(f"Scaler fitted on {len(numeric_features)} features: {numeric_features}")
else:
    print(f"Failed to load model '{model_name}'.")


class PredictionRequest(BaseModel):
    features: Dict[str, float]

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
    return {"message": "API is working"} # Simple root endpoint to verify API is up and running

@app.post("/predict")

def predict(request: PredictionRequest):
    if model is None:
        print("Error: Model is not loaded, cannot perform prediction.")
        raise HTTPException(status_code=503, detail="Model not loaded - MLflow server may not be running")

    # Check if model feature count matches our current features
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(numeric_features):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Model feature mismatch",
                "message": f"Model was trained with {model.n_features_in_} features but current configuration uses {len(numeric_features)} features. The model needs to be retrained with the corrected feature set (excluding tip_amount to prevent data leakage).",
                "model_features": model.n_features_in_,
                "current_features": len(numeric_features),
                "required_features": numeric_features
            }
        )

    print (f"Received prediction request with features: {request.features}")

    missing_features = [f for f in numeric_features if f not in request.features]
    if missing_features:
        print (f"Error: Missing required features: {missing_features}. Required features are: {numeric_features}")
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing_features, "required_features": numeric_features},
        )

    input_data = []
    for feature in numeric_features:
        value = request.features[feature]
        # Validate individual feature values by using the Pydantic validator logic defined above
        PredictionRequest(features={feature: value})
        input_data.append(float(value))

    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    global prediction_id
    prediction_id += 1
    return {
        "prediction": float(prediction),
        "model_version": loaded_model_version,
        "model_stage": loaded_model_stage,
        "model_uri": loaded_model_uri,
        "prediction_id": prediction_id,
    }
    
@app.post("/predict/batch") # Accept up to 100 records for batch prediction
def predict_batch(requests: List[PredictionRequest]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded - MLflow server may not be running")

    # Check if model feature count matches our current features
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(numeric_features):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Model feature mismatch",
                "message": f"Model was trained with {model.n_features_in_} features but current configuration uses {len(numeric_features)} features. The model needs to be retrained with the corrected feature set (excluding tip_amount to prevent data leakage).",
                "model_features": model.n_features_in_,
                "current_features": len(numeric_features),
                "required_features": numeric_features
            }
        )

    if len(requests) > 100:
        raise HTTPException(status_code=422, detail="Batch size cannot exceed 100 records")

    input_data = []
    for req in requests:
        missing_features = [f for f in numeric_features if f not in req.features]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={"missing_features": missing_features, "required_features": numeric_features},
            )
        row = []
        for feature in numeric_features:
            value = req.features[feature]
            PredictionRequest(features={feature: value})
            row.append(float(value))
        input_data.append(row)

    input_scaled = scaler.transform(input_data)
    predictions = model.predict(input_scaled)
    global prediction_id
    results = []
    for pred in predictions:
        prediction_id += 1
        results.append({
            "prediction": float(pred),
            "model_version": loaded_model_version,
            "model_stage": loaded_model_stage,
            "model_uri": loaded_model_uri,
            "prediction_id": prediction_id,
        })
    return results


def _get_model_versions(name: str) -> List[Dict[str, Any]]:
    try:
        versions = mlflow_client.search_model_versions(f"name='{name}'")
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "status": v.status,
                "run_id": v.run_id,
                "source": v.source,
                "creation_timestamp": v.creation_timestamp,
            }
            for v in versions
        ]
    except Exception:
        return []


@app.get("/health")
def health():
    model_status = "loaded" if model is not None else "not_loaded"
    feature_mismatch = False
    if model is not None and hasattr(model, 'n_features_in_'):
        feature_mismatch = model.n_features_in_ != len(numeric_features)

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_fitted": bool(numeric_features),
        "model_uri": loaded_model_uri,
        "model_version": loaded_model_version,
        "model_stage": loaded_model_stage,
        "feature_count_match": not feature_mismatch,
        "model_features": model.n_features_in_ if model and hasattr(model, 'n_features_in_') else None,
        "current_features": len(numeric_features),
        "data_leakage_fixed": "tip_amount" not in numeric_features
    }


@app.get("/model/info")
def model_info():
    model_status = "loaded" if model is not None else "not_loaded"
    feature_mismatch = False
    if model is not None and hasattr(model, 'n_features_in_'):
        feature_mismatch = model.n_features_in_ != len(numeric_features)

    return {
        "model_name": model_name,
        "tracking_uri": mlflow.get_tracking_uri(),
        "model_loaded": model is not None,
        "model_uri": loaded_model_uri,
        "model_version": loaded_model_version,
        "model_stage": loaded_model_stage,
        "required_features": numeric_features,
        "feature_count": len(numeric_features),
        "model_feature_count": model.n_features_in_ if model and hasattr(model, 'n_features_in_') else None,
        "feature_mismatch": feature_mismatch,
        "data_leakage_status": "fixed" if "tip_amount" not in numeric_features else "present",
        "registered_versions": _get_model_versions(model_name),
    }


@app.get("/tests/run", response_model=TestSummary)
def run_tests():
    """Run all tests and return their results"""
    try:
        result = subprocess.run(
            ["py", "-3.12", "-m", "pytest", "test_app.py", "-v", "--tb=short", "--json-report", "--json-report-file=test_report.json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse pytest output to extract test results
        output_lines = result.stdout.split('\n')
        tests = []
        passed_count = 0
        failed_count = 0
        
        for line in output_lines:
            if "PASSED" in line:
                # Extract test name
                test_name = line.split("::")[1].split(" ")[0] if "::" in line else "Unknown"
                tests.append(TestResult(test_name=test_name, passed=True))
                passed_count += 1
            elif "FAILED" in line:
                test_name = line.split("::")[1].split(" ")[0] if "::" in line else "Unknown"
                tests.append(TestResult(test_name=test_name, passed=False))
                failed_count += 1
        
        total = passed_count + failed_count
        all_passed = failed_count == 0 and total > 0
        
        # If no tests were parsed, try to get from pytest summary line
        if total == 0:
            for line in output_lines:
                if "passed" in line or "failed" in line:
                    # Try to extract counts from summary line like "5 passed in 0.45s"
                    passed_match = re.search(r'(\d+) passed', line)
                    failed_match = re.search(r'(\d+) failed', line)
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
            tests=tests
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Test execution timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running tests: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    print("Running in production mode")

