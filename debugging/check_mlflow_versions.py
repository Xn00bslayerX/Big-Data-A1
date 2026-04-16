import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('http://localhost:5000')
client = MlflowClient()
name = 'TaxiTipRandomForestRegressor'

print('=== get_latest_versions ===')
for v in client.get_latest_versions(name):
    print(v.version, v.current_stage, v.status, v.run_id, v.source)

print('\n=== search_model_versions ===')
for v in client.search_model_versions("name='{}'".format(name)):
    print(v.version, v.current_stage, v.status, v.run_id, v.source)
