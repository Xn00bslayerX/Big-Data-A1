# Docker Compose Workflow Documentation

## Starting the Services

To start the API and MLflow tracking server, run:

```ps
docker compose up --build
```

- [API](http://localhost:8000)
- [MLflow tracking server](http://localhost:5000)

## Making Prediction Requests

You can make prediction requests to the API (replace feature values as appropriate):

```
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"fare_amount": 10, "trip_distance": 2, "pickup_hour": 14}}'

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"fare_amount": 20, "trip_distance": 5, "pickup_hour": 9}}'

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"fare_amount": 15, "trip_distance": 3, "pickup_hour": 20}}'
```

## Shutting Down

To stop and remove all containers, run:

```ps
docker compose down
```

## Container Size

- The API container is based on `python:3.12-slim` and only installs production dependencies from `requirements.prod.txt` for a minimal image size.
- The MLflow container uses the official MLflow image.

## Configuration Notes

- The API expects model files in `/app/data/processed/` (mounted/copied at build time).
- The API communicates with MLflow using the service name `mlflow` (see `MLFLOW_TRACKING_URI`).
- All services are on the `bigdata_net` Docker network for internal communication.

---

**No additional configuration is required to run the project with Docker Compose.**
