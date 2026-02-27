import sys
import pandas as pd
import numpy as np
import pickle
import os
sys.path.append(".")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

print("Loading clean data...")
df = pd.read_csv("dataset/combined_train_clean.csv", low_memory=False)
df = df.fillna("").reset_index(drop=True)
df["url"]   = df["url"].astype(str)
df["label"] = df["label"].astype(int)

df_url = df[(df["url"].str.len()>5) & (df["url"]!="nan")].reset_index(drop=True)
print(f"URL rows: {len(df_url):,}  (phish={(df_url['label']==1).sum():,} | legit={(df_url['label']==0).sum():,})")

phish = df_url[df_url["label"]==1]
legit = df_url[df_url["label"]==0]
n     = min(len(phish), len(legit), 500000)
df_bal = pd.concat([
    phish.sample(n, random_state=42),
    legit.sample(n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced: {len(df_bal):,} ({n:,} each)")

X = df_bal["url"].astype(str).tolist()
y = df_bal["label"].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp)
print(f"Train={len(X_train):,} Val={len(X_val):,} Test={len(X_test):,}")

print("\nTraining SGD TF-IDF (char 2-6 ngrams)...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        max_features=500000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode"
    )),
    ("clf", CalibratedClassifierCV(
        SGDClassifier(
            loss="modified_huber",
            penalty="elasticnet",
            alpha=0.00001,
            l1_ratio=0.10,
            max_iter=300,
            tol=1e-5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        cv=3,
        method="isotonic"
    ))
])

pipeline.fit(X_train, y_train)

train_acc  = accuracy_score(y_train, pipeline.predict(X_train))
val_acc    = accuracy_score(y_val,   pipeline.predict(X_val))
test_acc   = accuracy_score(y_test,  pipeline.predict(X_test))
val_roc    = roc_auc_score(y_val,   pipeline.predict_proba(X_val)[:,1])
test_roc   = roc_auc_score(y_test,  pipeline.predict_proba(X_test)[:,1])
gap        = train_acc - val_acc

print(f"\n{'='*45}")
print(f"Train Acc  : {train_acc:.4f}")
print(f"Val Acc    : {val_acc:.4f}")
print(f"Test Acc   : {test_acc:.4f}")
print(f"Val ROC    : {val_roc:.4f}")
print(f"Test ROC   : {test_roc:.4f}")
print(f"Overfit    : {gap:.4f} ({'WARNING' if gap>0.03 else 'OK'})")
print(f"90% check  : {'PASSED' if test_acc>=0.90 else 'FAILED'}")
print(f"{'='*45}")
print(classification_report(y_test, pipeline.predict(X_test)))

print("\nSanity check:")
tests = [
    ("http://paypal-secure-login.tk/verify/account",   "PHISHING"),
    ("https://www.google.com",                          "LEGITIMATE"),
    ("https://github.com/openai/gpt-4",                "LEGITIMATE"),
    ("http://192.168.1.1/banking/login.php",           "PHISHING"),
    ("http://apple-id-verify.cf/confirm/password",     "PHISHING"),
    ("https://www.amazon.com/dp/B09G9HD6PD",           "LEGITIMATE"),
    ("http://microsoft-update-secure.tk/login",        "PHISHING"),
    ("https://stackoverflow.com/questions/12345",      "LEGITIMATE"),
]
for url, expected in tests:
    score  = pipeline.predict_proba([url])[0][1]
    pred   = "PHISHING" if score >= 0.5 else "LEGITIMATE"
    status = "OK" if pred == expected else "WRONG"
    print(f"  [{status}] {pred} ({score:.4f}) {url[:65]}")

with open("ml/saved_models/sgd.pkl","wb") as f:
    pickle.dump(pipeline, f)
print("\nSaved ml/saved_models/sgd.pkl")
print("Now update prediction_service.py to feed raw URL string to SGD")
print("and feature vectors to XGBoost and RF only.")
