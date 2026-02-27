from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def get_random_forest_model():
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    return model

def train_random_forest(X_train, y_train):
    model = get_random_forest_model()
    model.fit(X_train, y_train)
    print(f"OOB Score: {model.oob_score_:.4f}")
    importances = sorted(enumerate(model.feature_importances_), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 features:", importances)
    return model

def evaluate_random_forest(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report, proba
