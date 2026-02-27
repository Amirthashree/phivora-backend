import sys, os, numpy as np
sys.path.append(os.path.abspath("."))
from backend.services.model_loader import model_loader
from backend.services.feature_service import extract_features

model_loader.load()
url = "http://paypal-secure-login.tk/verify/account"

print("feature_cols type:", type(model_loader.feature_cols))
print("feature_cols length:", len(model_loader.feature_cols))
print("first 5 cols:", model_loader.feature_cols[:5])

try:
    features = extract_features(url, "", model_loader.feature_cols)
    print("features length:", len(features))
    print("first 5 values:", features[:5])
    X = np.array(features, dtype=float).reshape(1, -1)
    print("X shape:", X.shape)
    xgb_p = float(model_loader.xgboost.predict_proba(X)[0][1])
    print("XGBoost score:", xgb_p)
    sgd_p = float(model_loader.sgd.predict_proba(X)[0][1])
    print("SGD score:", sgd_p)
    rf_p = float(model_loader.random_forest.predict_proba(X)[0][1])
    print("RF score:", rf_p)
except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
