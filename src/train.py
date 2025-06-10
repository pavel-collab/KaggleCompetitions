from sklearn.preprocessing import StandardScaler
from utils.utils import (import_data, train_model, save_model)
from models import (models_list, linear_models)

X_train, y_train, _, _ = import_data('../data/')

for model_name, model in models_list.items():
    if model_name in linear_models:
        # Масштабирование для линейных моделей и SVM
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    trained_model = train_model(model, X_train, y_train)
    save_model(train_model, f"../saved_models/{model_name}.pkl")
