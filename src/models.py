from model.linear_models import lr, ridge
from model.random_forest import rf
from model.boosting import xgb
from model.svm import svm
from model.decision_tree import dt
from model.stacking import voting, stacking

models_list = {
    "logestic_regression": lr,
    "ridge_classifier": ridge,  
    "xgboost": xgb,
    "svm": svm,
    "decision_tree": dt,
    "random_forest": rf,
    # "voting": voting,
    "stacking": stacking
}

linear_models = ["logestic_regression", 
                 "ridge_classifier"]