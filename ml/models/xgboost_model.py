import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

def get_xgboost_model():
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    return model

def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    model = get_xgboost_model()
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
    else:
        model.fit(X_train, y_train)
    return model

def evaluate_xgboost(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report, proba
