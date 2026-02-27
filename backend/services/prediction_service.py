import sys, os, numpy as np
sys.path.append(os.path.abspath("."))
from urllib.parse import urlparse
from backend.services.model_loader import model_loader
from backend.services.feature_service import get_data_type, extract_features
from backend.config import settings

TRUSTED_EXACT = [
    "google.com","facebook.com","amazon.com","microsoft.com","apple.com",
    "netflix.com","instagram.com","twitter.com","linkedin.com","youtube.com",
    "github.com","wikipedia.org","stackoverflow.com","gmail.com","outlook.com",
    "yahoo.com","bing.com","reddit.com"
]

def is_trusted(url: str) -> bool:
    try:
        domain = urlparse(url if url.startswith("http") else "http://"+url).netloc.lower().replace("www.","")
        return any(domain == t or domain.endswith("."+t) for t in TRUSTED_EXACT)
    except:
        return False

def predict(url: str = "", text: str = "") -> dict:
    url  = str(url  or "").strip()
    text = str(text or "").strip()
    data_type = get_data_type(url, text)
    scores = {}

    if url and url != "nan" and len(url) > 3:
        # XGBoost + RF — feature vectors
        try:
            features = extract_features(url, text, model_loader.feature_cols)
            X = np.array(features, dtype=float).reshape(1, -1)
            scores["xgboost"]       = round(float(model_loader.xgboost.predict_proba(X)[0][1]),       4)
            scores["random_forest"] = round(float(model_loader.random_forest.predict_proba(X)[0][1]), 4)
        except Exception as e:
            print(f"XGB/RF error: {e}")
            scores["xgboost"]       = 0.5
            scores["random_forest"] = 0.5

        # SGD — raw URL string (TF-IDF char ngrams)
        try:
            scores["sgd"] = round(float(model_loader.sgd.predict_proba([url])[0][1]), 4)
        except Exception as e:
            print(f"SGD error: {e}")
            scores["sgd"] = 0.5

        # Naive Bayes — text if available else URL
        try:
            nb_input = text if text and len(text) > 10 else url
            scores["naive_bayes"] = round(float(model_loader.naive_bayes.predict_proba([nb_input])[0][1]), 4)
        except Exception as e:
            print(f"NB error: {e}")
            scores["naive_bayes"] = 0.5

    elif text and text != "nan" and len(text) > 10:
        try:
            scores["naive_bayes"] = round(float(model_loader.naive_bayes.predict_proba([text])[0][1]), 4)
        except Exception as e:
            print(f"NB error: {e}")
            scores["naive_bayes"] = 0.5

    xgb = scores.get("xgboost",       0.5)
    rf  = scores.get("random_forest",  0.5)
    sgd = scores.get("sgd",            0.5)
    nb  = scores.get("naive_bayes",    0.5)

    if data_type == "text":
        final = nb

    elif data_type == "url":
        if is_trusted(url):
            # trusted: feature models dominate
            final = xgb*0.40 + rf*0.30 + sgd*0.20 + nb*0.10
        else:
            # unknown: best weights from grid search
            final = sgd*0.50 + xgb*0.25 + rf*0.20 + nb*0.05

    else:  # both url + text
        final = sgd*0.30 + xgb*0.20 + rf*0.15 + nb*0.35

    final   = round(final, 4)
    label   = 1 if final >= 0.55 else 0
    verdict = "PHISHING" if label == 1 else "LEGITIMATE"

    if final >= 0.85:   severity = "HIGH"
    elif final >= 0.65: severity = "MEDIUM"
    elif final >= 0.55: severity = "LOW"
    else:               severity = "SAFE"

    return {
        "label":        label,
        "verdict":      verdict,
        "confidence":   final,
        "severity":     severity,
        "data_type":    data_type,
        "model_scores": scores
    }
