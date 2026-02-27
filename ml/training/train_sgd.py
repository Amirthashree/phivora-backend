import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath("."))
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier

def run_train_sgd():
    print("=== Training Gradient Boosting Model ===")
    df = pd.read_csv("dataset/train_features.csv", low_memory=False)
    df = df.fillna(0)
    df_url = df[df["is_url_row"] == 1]
    print(f"URL rows: {len(df_url)}")
    phishing = df_url[df_url["label"] == 1]
    legit    = df_url[df_url["label"] == 0]
    min_size = min(len(phishing), len(legit), 200000)
    df_balanced = pd.concat([phishing.sample(min_size, random_state=42), legit.sample(min_size, random_state=42)]).sample(frac=1, random_state=42)
    print(f"Balanced: {len(df_balanced)} rows")
    feature_cols = [c for c in df_balanced.columns if c != "label"]
    X = df_balanced[feature_cols].values.astype(float)
    y = df_balanced["label"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, min_samples_split=10, min_samples_leaf=5, random_state=42, verbose=1)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc   = accuracy_score(y_val,   model.predict(X_val))
    val_proba = model.predict_proba(X_val)[:, 1]
    roc       = roc_auc_score(y_val, val_proba)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy  : {val_acc:.4f}")
    print(f"ROC-AUC       : {roc:.4f}")
    if train_acc - val_acc > 0.05:
        print("WARNING: Overfitting")
    else:
        print("Overfitting check PASSED")
    print(classification_report(y_val, model.predict(X_val)))
    os.makedirs("ml/saved_models", exist_ok=True)
    with open("ml/saved_models/sgd.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("ml/saved_models/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print("Saved to ml/saved_models/sgd.pkl")
    return model

if __name__ == "__main__":
    run_train_sgd()
