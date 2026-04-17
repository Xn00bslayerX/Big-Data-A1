# Docker Compose Workflow Documentation

## Overview

This document outlines the complete workflow for deploying the Big Data Assignment 4 application using Docker Compose. The setup consists of two services:

- **api**: FastAPI-based prediction service with ML models
- **mlflow**: MLflow tracking server for model versioning and experiment tracking

## Prerequisites

Before starting, ensure:

1. Docker and Docker Compose are installed and running
2. All trained model files exist in `data/processed/`:
   - `classification_model.joblib`
   - `regression_model.joblib`
   - `scaler.joblib`
   - `numeric_features.joblib`
   - `regression_model.pt`
3. `requirements.prod.txt` is present with production dependencies
4. `Dockerfile` is configured to copy model files into the container
5. `app.py` is configured to load models from environment variable paths

## Starting the Services

To start the API and MLflow tracking server with a fresh build, run:

```powershell
docker compose up --build
```

This command will:

1. Build the API Docker image from the Dockerfile
2. Create and start the `bigdata_api` container on port 8000
3. Create and start the `bigdata_mlflow` container on port 5000
4. Establish the `bigdata_net` bridge network for inter-service communication
5. Mount volumes for data persistence and model access

### Expected Output

You should see output similar to:

```txt
[+] Running 2/2
 ✓ Container bigdata_mlflow  Created
 ✓ Container bigdata_api     Created
Attaching to bigdata_mlflow, bigdata_api
bigdata_mlflow  | [2026-04-17 12:00:00 +0000] [1] [INFO] Starting gunicorn 20.1.0
bigdata_api     | INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Service Accessibility

Once running, access the services at:

- **API Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)

## Making Prediction Requests

The API accepts POST requests at `/predict` endpoint. Make at least 3 prediction requests with different feature values:

### Request 1: Morning Commute

```powershell
curl -X POST "http://localhost:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"features": {"fare_amount": 10, "trip_distance": 2, "pickup_hour": 14}}'
```

**Expected Response:**

```json
{
  "prediction": <numeric_value>,
  "model_used": "regression_model",
  "timestamp": "2026-04-17T12:00:00"
}
```

### Request 2: Longer Trip

```powershell
curl -X POST "http://localhost:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"features": {"fare_amount": 20, "trip_distance": 5, "pickup_hour": 9}}'
```

### Request 3: Evening Trip

```powershell
curl -X POST "http://localhost:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"features": {"fare_amount": 15, "trip_distance": 3, "pickup_hour": 20}}'
```

### Alternative: Using PowerShell Invoke-WebRequest

```powershell
$body = @{
    features = @{
        fare_amount = 10
        trip_distance = 2
        pickup_hour = 14
    }
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/predict" `
                  -Method POST `
                  -ContentType "application/json" `
                  -Body $body
```

## Verifying MLflow Integration

The API should automatically log predictions and metrics to the MLflow tracking server. Verify this by:

1. Opening the MLflow UI at <http://localhost:5000>
2. Check that new experiments and runs appear as you make predictions
3. Review metrics, parameters, and artifacts logged by the API

## Monitoring Containers

### Check Running Containers

```powershell
docker ps
```

### View Container Logs

```powershell
# API logs
docker logs bigdata_api

# MLflow logs
docker logs bigdata_mlflow

# Follow logs in real-time
docker logs -f bigdata_api
```

### Inspect Container Network

```powershell
docker network inspect bigdata_net
```

## Shutting Down

To stop and remove all containers cleanly, run:

```powershell
docker compose down
```

This command will:

1. Stop both `bigdata_api` and `bigdata_mlflow` containers
2. Remove the containers
3. Remove the `bigdata_net` network
4. Preserve volumes (mlruns data persists)

### Shutting Down and Removing Volumes

To completely clean up including volumes:

```powershell
docker compose down -v
```

**Warning**: This will delete all MLflow run history stored in `./mlruns/`

## Container Information

### Image Specifications

#### API Container (bigdata_api)

- **Base Image**: `python:3.12-slim`
- **Dependencies**: Production dependencies from `requirements.prod.txt`
- **Size**: ~9 GB (includes model files and dependencies)
- **Restart Policy**: `unless-stopped`

#### MLflow Container (bigdata_mlflow)

- **Base Image**: `python:3.12-slim`
- **MLflow Version**: 2.11.3
- **Size**: ~500 MB (Python base + MLflow)
- **Restart Policy**: `unless-stopped`

### Size Optimization

The API container uses `python:3.12-slim` (not full `python:3.12`) to minimize base image size. The large final size (9 GB) is primarily due to:

- Pre-trained ML model files (classification, regression, PyTorch)
- Production dependencies installed via pip
- Scaler and feature definitions

## Configuration Details

### Environment Variables

The API container is configured with these environment variables:

- `CLASSIFICATION_MODEL_PATH=/app/data/processed/classification_model.joblib`
- `REGRESSION_MODEL_PATH=/app/data/processed/regression_model.joblib`
- `SCALER_PATH=/app/data/processed/scaler.joblib`
- `NUMERIC_FEATURES_PATH=/app/data/processed/numeric_features.joblib`
- `MLFLOW_TRACKING_URI=http://mlflow:5000`

### Volume Mounts

- **API**: `./data:/app/data` - Mounts local data directory for model file access
- **MLflow**: `./mlruns:/mlflow/mlruns` - Persists experiment tracking data locally

### Networking

- **Network Driver**: Bridge (`bigdata_net`)
- **API Service Hostname**: `api` (accessible as `http://api:8000` within network)
- **MLflow Service Hostname**: `mlflow` (accessible as `http://mlflow:5000` within network)
- The API resolves MLflow at `http://mlflow:5000` through Docker's internal DNS

## Troubleshooting

### Models Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: '/app/data/processed/model.joblib'`

**Solution**: Verify all model files exist in `data/processed/` before running `docker compose up --build`

### Port Already in Use

**Error**: `Error response from daemon: Ports are not available`

**Solution**:

```powershell
# Kill process on port 8000 or 5000
Get-Process | Where-Object {$_.Handles -gt 0} | Stop-Process -Force
# Or change port in docker-compose.yml
```

### API Cannot Connect to MLflow

**Error**: `Connection refused` when logging to MLflow

**Solution**:

1. Verify MLflow container is running: `docker ps`
2. Verify network connectivity: `docker network inspect bigdata_net`
3. Check MLflow logs: `docker logs bigdata_mlflow`

### Container Exits Immediately

**Error**: Container starts then stops

**Solution**:

1. Check logs: `docker logs bigdata_api`
2. Verify `app.py` starts correctly locally
3. Ensure all dependencies in `requirements.prod.txt` are correct

## Configuration Notes

- The API expects model files in `/app/data/processed/` (mounted from `./data` at runtime)
- The API communicates with MLflow using the service name `mlflow` via the Docker internal DNS
- All services are isolated on the `bigdata_net` Docker network for internal communication
- The `restart: unless-stopped` policy ensures containers restart automatically unless manually stopped

## Summary

**No additional configuration is required to run the project with Docker Compose.** Simply ensure model files exist locally, then:

1. Run: `docker compose up --build`
2. Make prediction requests to `http://localhost:8000/predict`
3. Monitor MLflow at `http://localhost:5000`
4. Shut down with: `docker compose down`

All inter-service communication, networking, and data persistence are automatically handled by the Docker Compose configuration.
