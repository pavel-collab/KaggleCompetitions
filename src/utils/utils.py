import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

def define_random_seed(seed=20):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
def get_device():
    # детектируем девайс
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def evaluate_model(model, X_eval, y_eval, device='cpu'):
    model.to(device)
    model.eval()

    y_pred = model.predict(X_eval)

    cm = confusion_matrix(y_eval, y_pred)
    report = classification_report(y_eval, y_pred)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    # Вычисление взвешенной F1-меры для текущей модели
    weighted_f1 = f1_score(y_eval, y_pred, average='weighted')
    return weighted_f1, cm, report, accuracy 

def train_model(model, X_train, y_train, device='cpu'):
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    model_trained = model.fit(X_train, y_train)
    return model_trained

def import_data(data_path):
    data = pd.read_csv(data_path)

    #TODO: make a data preprocessing
    X = data.drop('target_column', axis=1) # make a data matrix
    y = data['target_column'] # extract target label

    # Разделение данных
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_eval, y_eval

def save_model(trained_model, save_model_filepath: str):
    with open(save_model_filepath, 'wb') as file:
        pickle.dump(trained_model, file)

def extract_model(save_model_filepath):
    with open(save_model_filepath, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model