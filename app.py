import os
import subprocess
import json
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, validator
from sklearn.preprocessing import StandardScaler
import uvicorn

app = FastAPI()




# Model and scaler loading
model: Optional[Any] = None
numeric_features: List[str] = []
scaler = StandardScaler()
global prediction_id
prediction_id = 0


# Load numeric features, scaler, and model from disk
numeric_features_path = "data/processed/numeric_features.joblib"
scaler_path = "data/processed/scaler.joblib"
classification_model_path = "data/processed/classification_model.joblib"
regression_model_path = "data/processed/regression_model.joblib"

if os.path.exists(numeric_features_path):
    numeric_features = joblib.load(numeric_features_path)
    print(f"Loaded numeric features: {numeric_features}")
else:
    print("Warning: numeric_features.joblib not found.")

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Scaler loaded from scaler.joblib.")
else:
    print("Warning: scaler.joblib not found.")

if os.path.exists(classification_model_path):
    classification_model = joblib.load(classification_model_path)
    print("Classification model loaded from classification_model.joblib.")
    if hasattr(classification_model, 'n_features_in_'):
        print(f"Classification model expects {classification_model.n_features_in_} features.")
else:
    print("Warning: classification_model.joblib not found.")
    raise HTTPException(status_code=503, detail="Classification model not loaded - classification_model.joblib not found or failed to load")

if os.path.exists(regression_model_path):
    regression_model = joblib.load(regression_model_path)
    print("Regression model loaded from regression_model.joblib.")
    if hasattr(regression_model, 'n_features_in_'):
        print(f"Regression model expects {regression_model.n_features_in_} features.")
else:
    print("Warning: regression_model.joblib not found.")
    raise HTTPException(status_code=503, detail="Regression model not loaded - regression_model.joblib not found or failed to load")





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
        raise HTTPException(status_code=503, detail="Model not loaded - model.joblib not found or failed to load")

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
        "prediction_id": prediction_id,
    }
    
@app.post("/predict/batch") # Accept up to 100 records for batch prediction
def predict_batch(requests: List[PredictionRequest]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded - model.joblib not found or failed to load")

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
            "prediction_id": prediction_id,
        })
    return results





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
        "model_loaded": model is not None,
        "required_features": numeric_features,
        "feature_count": len(numeric_features),
        "model_feature_count": model.n_features_in_ if model and hasattr(model, 'n_features_in_') else None,
        "feature_mismatch": feature_mismatch,
        "data_leakage_status": "fixed" if "tip_amount" not in numeric_features else "present",
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

