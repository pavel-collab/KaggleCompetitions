from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from .linear_models import lr
from .random_forest import rf
from .boosting import xgb
from .svm import svm
from .decision_tree import dt

# Простое голосование
voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='soft'
)

# Стекинг моделей
stacking = StackingClassifier(
    estimators=[
        ('dt', dt),
        ('svm', svm),
        ('xgb', xgb)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)