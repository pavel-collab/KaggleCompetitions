from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils.utils import (import_data, train_model, save_model)
from models import (models_list, linear_models)
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='./data/train.csv', help='path to file with train data')
args = parser.parse_args()

train_data_path = Path(args.data)
assert(train_data_path.exists())

X_train, y_train, X_eval, y_eval = import_data(train_data_path.absolute())

for model_name, model in models_list.items():
    print(f"[DEBUG] start to train model {model_name}")
    
    if model_name in linear_models:
        # Масштабирование для линейных моделей и SVM
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.fit_transform(X_eval)

    trained_model = train_model(model, X_train, y_train)
    
    y_pred = trained_model.predict(X_eval)
    print(f"[DEBUG] {model_name} Accuracy: {accuracy_score(y_eval, y_pred):.4f}\n")
    
    save_model(trained_model, f"./saved_models/{model_name}.pkl")
