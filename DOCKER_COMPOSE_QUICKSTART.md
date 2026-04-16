# Docker Compose Quick Reference

## Start Services

```ps
docker compose up --build
```

## Make Prediction Requests
```
curl -X POST "<http://localhost:8000/predict>" -H "Content-Type: application/json" -d '{
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
curl -X POST "<http://localhost:8000/predict>" -H "Content-Type: application/json" -d '{
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
curl -X POST "<http://localhost:8000/predict>" -H "Content-Type: application/json" -d '{
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

## Shut Down

```ps
docker compose down
```

## Container Size

- API: Based on `python:3.12-slim`, production dependencies only (small footprint)
- MLflow: Official MLflow image

## Configuration

- Model files expected in `/app/data/processed/`
- API communicates with MLflow at `http://mlflow:5000`
- Both services share the `bigdata_net` Docker network

---
**No extra configuration is needed.**
