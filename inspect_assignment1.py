import json
from pathlib import Path

nb_path = Path('assignment1.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
print('Total cells:', len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if any(token in src for token in [
        'numeric_features', 'tip_amount', 'train_df', 'X_train', 'feature_names_in_',
        'mlflow', 'log_model', 'register_model', 'start_run', 'model_name'
    ]):
        print('='*40)
        print('CELL', i)
        print(src)
