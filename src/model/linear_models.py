from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Логистическая регрессия
lr = LogisticRegression(max_iter=1000, random_state=42)

# Гребневая регрессия (вариация линейной)
ridge = RidgeClassifier(random_state=42)