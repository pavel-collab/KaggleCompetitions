from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=7,
    random_state=42
)