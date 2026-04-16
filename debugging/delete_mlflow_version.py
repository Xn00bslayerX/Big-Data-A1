import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('http://localhost:5000')
client = MlflowClient()
name = 'TaxiTipRandomForestRegressor'
version = '1'

print(f'Deleting model version {version} for {name}...')
client.delete_model_version(name, version)
print('Deleted. Current versions:')
for v in client.search_model_versions(f"name='{name}'"):
    print(v.version, v.current_stage, v.status, v.run_id, v.source)
