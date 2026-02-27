import pickle
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath("."))
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from backend.services.feature_service import extract_features

print("Loading models...")
with open("ml/saved_models/xgboost.pkl",       "rb") as f: xgb = pickle.load(f)
with open("ml/saved_models/random_forest.pkl",  "rb") as f: rf  = pickle.load(f)
with open("ml/saved_models/sgd.pkl",            "rb") as f: sgd = pickle.load(f)
with open("ml/saved_models/naive_bayes.pkl",    "rb") as f: nb  = pickle.load(f)
with open("ml/saved_models/feature_cols.pkl",   "rb") as f: cols = pickle.load(f)

print("Loading test data...")
df = pd.read_csv("dataset/combined_test.csv", low_memory=False)
df = df.fillna("").reset_index(drop=True)
df["url"]   = df["url"].astype(str)
df["text"]  = df["text"].astype(str)
df["label"] = df["label"].astype(int)

# URL evaluation
url_df = df[(df["url"].str.len()>5) & (df["url"]!="nan")].reset_index(drop=True)
sample = url_df.sample(min(2000, len(url_df)), random_state=42).reset_index(drop=True)
print(f"\nEvaluating on {len(sample)} URL samples...")

X = np.array([extract_features(u, "", cols) for u in sample["url"]], dtype=float)
y = sample["label"].values

xgb_preds  = xgb.predict(X)
xgb_proba  = xgb.predict_proba(X)[:,1]
rf_preds   = rf.predict(X)
rf_proba   = rf.predict_proba(X)[:,1]
sgd_proba  = sgd.predict_proba(sample["url"].tolist())[:,1]
sgd_preds  = (sgd_proba >= 0.5).astype(int)

# Ensemble
final = sgd_proba*0.50 + xgb_proba*0.25 + rf_proba*0.20 + 0.05*0.5
ens_preds = (final >= 0.55).astype(int)

print("\n" + "="*50)
print(f"{'Model':<20} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'ROC-AUC':>8}")
print("="*50)
for name, preds, proba in [
    ("XGBoost",       xgb_preds,  xgb_proba),
    ("Random Forest", rf_preds,   rf_proba),
    ("SGD",           sgd_preds,  sgd_proba),
    ("Ensemble",      ens_preds,  final),
]:
    acc  = accuracy_score(y, preds)
    f1   = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    roc  = roc_auc_score(y, proba)
    print(f"{name:<20} {acc:>10.4f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {roc:>8.4f}")

print("="*50)

# Text evaluation
text_df = df[(df["text"].str.len()>10) & (df["text"]!="nan")].reset_index(drop=True)
text_sample = text_df.sample(min(2000, len(text_df)), random_state=42).reset_index(drop=True)
print(f"\nEvaluating Naive Bayes on {len(text_sample)} text samples...")
nb_preds = nb.predict(text_sample["text"].tolist())
nb_proba = nb.predict_proba(text_sample["text"].tolist())[:,1]
acc  = accuracy_score(text_sample["label"], nb_preds)
f1   = f1_score(text_sample["label"], nb_preds, zero_division=0)
roc  = roc_auc_score(text_sample["label"], nb_proba)
print(f"Naive Bayes  ->  Accuracy: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {roc:.4f}")
print("="*50)
print("\nDone!")
