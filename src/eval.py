from sklearn.preprocessing import StandardScaler
from utils.utils import (import_data, extract_model, evaluate_model)
from models import (models_list, linear_models)
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='./data/train.csv', help='path to file with train data')
args = parser.parse_args()

train_data_path = Path(args.data)
assert(train_data_path.exists())

_, _, X_eval, y_eval = import_data(train_data_path.absolute())

for model_name, _ in models_list.items():
    model = extract_model(f"./saved_models/{model_name}.pkl")
    if model_name in linear_models:
        # Масштабирование для линейных моделей и SVM
        scaler = StandardScaler()
        X_eval = scaler.fit_transform(X_eval)

    _, _, report, _  = evaluate_model(model, X_eval, y_eval)
    print(report)