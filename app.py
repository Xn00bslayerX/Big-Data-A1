import os
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
        if df[col].dtype.is_numeric() and col not in {"tip_amount", "high_tip"} # Read data from all columns except target variables
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


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    missing_features = [f for f in numeric_features if f not in request.features]
    if missing_features:
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
        raise HTTPException(status_code=500, detail="Model not loaded")

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
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_fitted": bool(numeric_features),
        "model_uri": loaded_model_uri,
        "model_version": loaded_model_version,
        "model_stage": loaded_model_stage,
    }


@app.get("/model/info")
def model_info():
    return {
        "model_name": model_name,
        "tracking_uri": mlflow.get_tracking_uri(),
        "model_loaded": model is not None,
        "model_uri": loaded_model_uri,
        "model_version": loaded_model_version,
        "model_stage": loaded_model_stage,
        "required_features": numeric_features,
        "registered_versions": _get_model_versions(model_name),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    print("Running in production mode")

