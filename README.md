
# Big-Data-A4

This project predicts NYC taxi trip tips and passenger tipping behavior using machine learning models trained on real taxi trip data (January 2024). It provides:

- Data analysis and feature engineering in Jupyter Notebooks
- A production-ready FastAPI prediction service (with Docker support)
- MLflow tracking for experiment management
- Automated tests for API and model validation

## Project Structure & Workflow

- **Jupyter Notebooks**: Data download, preprocessing, feature engineering, model training, and evaluation (`assignment1.ipynb`, `assignment1_executed.ipynb`).
- **FastAPI Service**: REST API for predictions (`app.py`).
- **MLflow**: Experiment tracking and model registry (runs as a service via Docker Compose).
- **Docker Compose**: Orchestrates API and MLflow services for local development/production.
- **Automated Tests**: API and model validation (`test_app.py`).

## Requirements

- Python 3.10 or higher
- Docker & Docker Compose (for containerized workflow)
- Jupyter Notebook (for interactive analysis)
- See `requirements.txt` and `requirements.prod.txt` for dependencies

## Getting Started

### 1. Clone the repository

### 2. Run Data Analysis & Model Training (Jupyter Notebook)

```bash
jupyter notebook assignment1.ipynb
```

Run all cells to download data, preprocess, engineer features, and train models. This will generate processed data and model artifacts in `data/processed/`. An internet connection is needed to connect to the server to download the data files.

### 3. Run the API & MLflow with Docker Compose

Build and start all services:

```bash
docker compose up --build
```

This launches:

- **API** (<http://localhost:8000>): FastAPI prediction service
- **MLflow** (<http://localhost:5000>): Experiment tracking UI

The API loads the trained model and exposes endpoints for prediction and health checks.

### 4. Make Prediction Requests

Example (single prediction):

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "features": {
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
    "pickup_hour": 14.0
  }
}'
```

See `DOCKER_COMPOSE_QUICKSTART.md` for more examples.

### 5. Run Tests

Tests are provided for API endpoints and input validation:

```bash
pytest test_app.py
```

Or trigger tests via the API:

```bash
curl http://localhost:8000/tests/run
```

### 6. Shut Down

```bash
docker compose down
```

## Key Files & Directories

- `assignment1.ipynb` / `assignment1_executed.ipynb`: Data analysis, feature engineering, model training
- `app.py`: FastAPI prediction service (loads model, exposes `/predict`, `/health`, `/tests/run` endpoints)
- `test_app.py`: Automated tests for API/model
- `docker-compose.yml`: Orchestrates API and MLflow services
- `Dockerfile`: Containerizes the API for production
- `requirements.txt` / `requirements.prod.txt`: Development and production dependencies
- `data/`: This is generated from running. Not found in the repository.
  - `raw/`: Downloaded source data
  - `processed/`: Cleaned data, trained model, scaler, and feature list (`model.joblib`, `scaler.joblib`, `numeric_features.joblib`)
- `mlruns/`: MLflow experiment tracking and model registry

## Machine Learning Models

**Target Variables:**

- Regression: `tip_amount` (continuous)
- Classification: `high_tip` (binary: tip > 20% of fare)

**Models Trained:**

- Linear Regression
- Random Forest (Regressor & Classifier)
- Logistic Regression
- Neural Network (PyTorch)

**Evaluation Metrics:**

- Regression: MAE, RMSE, R²
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC

**Key Features:**

- Temporal: pickup_hour, pickup_day_of_week, is_weekend
- Trip: trip_duration_minutes, trip_speed_mph, trip_distance
- Fare: fare_per_mile, fare_per_minute, log_trip_distance
- Location: PULocationID, DOLocationID, pickup/dropoff zones

---
This project is for educational purposes as part of a Big Data assignment.
