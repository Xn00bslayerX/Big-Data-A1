# Docker Compose Quick Reference

**For detailed documentation, see [DOCKER_COMPOSE_WORKFLOW.md](DOCKER_COMPOSE_WORKFLOW.md)**

## Prerequisites

✓ All model files exist in `data/processed/`  
✓ Docker and Docker Compose installed  
✓ Ports 8000 (API) and 5000 (MLflow) are available

## Start Services

```powershell
docker compose up --build
```

- **API**: <http://localhost:8000>
- **MLflow UI**: <http://localhost:5000>

## Make 3 Prediction Requests

### Request 1: Standard Trip

```powershell
curl -X POST "http://localhost:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{
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

### Request 2: Short Trip

```powershell
curl -X POST "http://localhost:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{
  "features": {
    "VendorID": 2.0,
    "passenger_count": 1.0,
    "trip_distance": 3.0,
    "RatecodeID": 1.0,
    "PULocationID": 120.0,
    "DOLocationID": 210.0,
    "payment_type": 2.0,
    "fare_amount": 15.0,
    "extra": 0.0,
    "mta_tax": 0.5,
    "tolls_amount": 0.0,
    "improvement_surcharge": 0.3,
    "total_amount": 15.8,
    "congestion_surcharge": 2.5,
    "Airport_fee": 0.0,
    "trip_duration_minutes": 10.0,
    "trip_speed_mph": 18.0,
    "log_trip_distance": 1.098,
    "fare_per_mile": 5.0,
    "fare_per_minute": 1.5,
    "pickup_hour": 9.0
  }
}'
```

### Request 3: Long Trip with Tolls

```powershell
curl -X POST "http://localhost:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{
  "features": {
    "VendorID": 1.0,
    "passenger_count": 3.0,
    "trip_distance": 7.0,
    "RatecodeID": 2.0,
    "PULocationID": 130.0,
    "DOLocationID": 220.0,
    "payment_type": 1.0,
    "fare_amount": 25.0,
    "extra": 1.0,
    "mta_tax": 0.5,
    "tolls_amount": 5.0,
    "improvement_surcharge": 0.3,
    "total_amount": 31.8,
    "congestion_surcharge": 2.5,
    "Airport_fee": 0.0,
    "trip_duration_minutes": 20.0,
    "trip_speed_mph": 21.0,
    "log_trip_distance": 1.946,
    "fare_per_mile": 3.6,
    "fare_per_minute": 1.25,
    "pickup_hour": 20.0
  }
}'
```

## Monitor Services

```powershell
# View running containers
docker ps

# Check API logs
docker logs bigdata_api

# Follow logs in real-time
docker logs -f bigdata_api

# Check MLflow logs
docker logs bigdata_mlflow
```

## Shut Down

```powershell
docker compose down
```

To also remove volumes (MLflow history):

```powershell
docker compose down -v
```

## Container Specifications

| Container | Base Image | Size | Port | Status |
|-----------|-----------|------|------|--------|
| bigdata_api | python:3.12-slim | ~9 GB | 8000 | ✓ Required |
| bigdata_mlflow | python:3.12-slim + MLflow 2.11.3 | ~500 MB | 5000 | ✓ Bonus (+5) |

## Configuration

| Setting | Value |
|---------|-------|
| Network | bigdata_net (bridge) |
| MLflow URI | <http://mlflow:5000> |
| Model Path | /app/data/processed/ |
| Data Volume | ./data:/app/data |
| MLflow Volume | ./mlruns:/mlflow/mlruns |
| Restart Policy | unless-stopped |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Change ports in docker-compose.yml |
| Models not found | Verify data/processed/ contains all .joblib files |
| Connection refused | Check `docker ps` to verify both containers are running |
| API won't start | Review `docker logs bigdata_api` for errors |

**For complete documentation and detailed troubleshooting, see [DOCKER_COMPOSE_WORKFLOW.md](DOCKER_COMPOSE_WORKFLOW.md)**

## Configuration

- Model files expected in `/app/data/processed/`
- API communicates with MLflow at `http://mlflow:5000`
- Both services share the `bigdata_net` Docker network

---
**No extra configuration is needed.**
