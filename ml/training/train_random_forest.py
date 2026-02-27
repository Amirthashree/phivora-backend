import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath("."))
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from ml.models.random_forest_model import train_random_forest, evaluate_random_forest

def run_train_random_forest():
    print("=== Training Random Forest Model ===")
    print("Loading features...")
    df = pd.read_csv("dataset/train_features.csv", low_memory=False)
    df = df.fillna(0)

    print(f"Total rows: {len(df)}")
    df_url = df[df["is_url_row"] == 1]
    print(f"URL-only rows: {len(df_url)}")

    phishing = df_url[df_url["label"] == 1]
    legit    = df_url[df_url["label"] == 0]
    min_size = min(len(phishing), len(legit))
    print(f"Balancing: {min_size} phishing + {min_size} legit")
    df_balanced = pd.concat([
        phishing.sample(min_size, random_state=42),
        legit.sample(min_size, random_state=42)
    ]).sample(frac=1, random_state=42)

    feature_cols = [c for c in df_balanced.columns if c != "label"]
    X = df_balanced[feature_cols].values.astype(float)
    y = df_balanced["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")
    print("Training Random Forest on URL features...")

    model = train_random_forest(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc   = accuracy_score(y_val,   model.predict(X_val))
    val_proba = model.predict_proba(X_val)[:, 1]
    roc       = roc_auc_score(y_val, val_proba)

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Val Accuracy   : {val_acc:.4f}")
    print(f"ROC-AUC        : {roc:.4f}")

    if train_acc - val_acc > 0.05:
        print("WARNING: Possible overfitting detected")
    else:
        print("Overfitting check PASSED")

    acc, report, _ = evaluate_random_forest(model, X_val, y_val)
    print(f"\nFinal Val Accuracy: {acc:.4f}")
    print(report)

    os.makedirs("ml/saved_models", exist_ok=True)
    with open("ml/saved_models/random_forest.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Random Forest saved to ml/saved_models/random_forest.pkl")
    return model

if __name__ == "__main__":
    run_train_random_forest()
