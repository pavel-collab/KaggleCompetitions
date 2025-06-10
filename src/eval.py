from sklearn.preprocessing import StandardScaler
from utils.utils import (import_data, extract_model, evaluate_model)
from models import (models_list, linear_models)

_, _, x_eval, y_eval = import_data('../data/')

for model_name, _ in models_list.items():
    model = extract_model(f"../saved_models/{model_name}.pkl")
    if model_name in linear_models:
        # Масштабирование для линейных моделей и SVM
        scaler = StandardScaler()
        X_eval = scaler.fit_transform(X_eval)

    _, _, report, _  = evaluate_model(model, X_eval, y_eval)
    print(report)