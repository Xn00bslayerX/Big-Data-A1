import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('http://localhost:5000')
client = MlflowClient()
name = 'TaxiTipRandomForestRegressor'

versions = client.search_model_versions(f"name='{name}'")
print('Registered versions:')
for v in versions:
    print('version', v.version, 'stage', v.current_stage, 'status', v.status, 'run_id', v.run_id, 'source', v.source)

# load latest registered version
if versions:
    v = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    print('\nLoading version', v.version, 'uri', f'models:/{name}/{v.version}')
    model = mlflow.sklearn.load_model(f'models:/{name}/{v.version}')
    print('model type:', type(model))
    print('n_features_in_:', getattr(model, 'n_features_in_', None))
    print('feature_names_in_:', getattr(model, 'feature_names_in_', None))
    print('steps:', getattr(model, 'steps', None))
    if hasattr(model, 'get_params'):
        print('\nParameter keys:', list(model.get_params().keys())[:50])
    if hasattr(model, 'get_params') and 'step' in str(type(model)):
        try:
            for name, step in model.steps:
                print('pipeline step', name, type(step), getattr(step, 'n_features_in_', None), getattr(step, 'feature_names_in_', None))
        except Exception as e:
            print('pipeline introspect error', e)
    if hasattr(model, 'coef_'):
        print('coef_ shape:', getattr(model, 'coef_', None).shape)
    if hasattr(model, 'estimators_'):
        print('estimators_ size:', len(model.estimators_))
    print('\nModel repr:')
    print(model)
