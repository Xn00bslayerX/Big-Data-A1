import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("taxi-tip-prediction")

# Create dummy data with 21 features
X, y = make_regression(n_samples=1000, n_features=21, noise=0.1, random_state=42)

# Train a model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# Log the model
run_id = None
with mlflow.start_run(run_name="Dummy Regression Model") as run:
    run_id = run.info.run_id
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 10)
    mlflow.sklearn.log_model(model, "random_forest_regressor_model")

# Register the model
client = mlflow.tracking.MlflowClient()
model_uri = f"runs:/{run_id}/random_forest_regressor_model"
model_details = mlflow.register_model(model_uri, "TaxiTipRandomForestRegressor")

# Set to production
client.transition_model_version_stage(name="TaxiTipRandomForestRegressor", version=model_details.version, stage="Production")

print("Model logged and registered")