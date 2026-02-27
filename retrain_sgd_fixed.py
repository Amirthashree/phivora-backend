import pandas as pd
import numpy as np
import pickle
import sys, os
sys.path.append(os.path.abspath("."))
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

print("Loading data...")
df = pd.read_csv("dataset/combined_train.csv", low_memory=False)
df = df.fillna("")
df_url = df[(df["url"].astype(str).str.len() > 5) & (df["url"].astype(str) != "nan")]
df_url = df_url[df_url["label"].isin([0, 1])]
print(f"URL rows: {len(df_url)}")

phishing = df_url[df_url["label"] == 1]
legit    = df_url[df_url["label"] == 0]
min_size = min(len(phishing), len(legit), 300000)
df_balanced = pd.concat([
    phishing.sample(min_size, random_state=42),
    legit.sample(min_size, random_state=42)
]).sample(frac=1, random_state=42)
print(f"Balanced: {len(df_balanced)} rows ({min_size} each)")

X = df_balanced["url"].astype(str).tolist()
y = df_balanced["label"].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

print("\nTraining SGD TF-IDF pipeline...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=300000,
        min_df=3,
        max_df=0.97,
        sublinear_tf=True,
        strip_accents="unicode"
    )),
    ("sgd", CalibratedClassifierCV(
        SGDClassifier(
            loss="modified_huber",
            penalty="elasticnet",
            alpha=0.00005,
            l1_ratio=0.15,
            max_iter=200,
            tol=1e-4,
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
val_proba  = pipeline.predict_proba(X_val)[:, 1]
test_proba = pipeline.predict_proba(X_test)[:, 1]
val_roc    = roc_auc_score(y_val,  val_proba)
test_roc   = roc_auc_score(y_test, test_proba)
gap        = train_acc - val_acc

print(f"\n{'='*45}")
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Val Accuracy   : {val_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Val  ROC-AUC   : {val_roc:.4f}")
print(f"Test ROC-AUC   : {test_roc:.4f}")
print(f"Overfit Gap    : {gap:.4f} ({'OVERFIT WARNING' if gap > 0.03 else 'OK'})")
print(f"Min 90% check  : {'PASSED' if test_acc >= 0.90 else 'FAILED'}")
print(f"{'='*45}")

print("\nClassification Report (Test):")
print(classification_report(y_test, pipeline.predict(X_test)))

print("\nRunning 5-fold CV (on sample for speed)...")
sample_idx = np.random.choice(len(X), min(50000, len(X)), replace=False)
X_cv = [X[i] for i in sample_idx]
y_cv = y[sample_idx]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_cv, y_cv, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"CV Scores : {[round(s,4) for s in cv_scores]}")
print(f"CV Mean   : {cv_scores.mean():.4f}")
print(f"CV Std    : {cv_scores.std():.4f} ({'STABLE' if cv_scores.std() < 0.02 else 'HIGH VARIANCE'})")

with open("ml/saved_models/sgd.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("\nSaved to ml/saved_models/sgd.pkl")

# Quick sanity check
print("\nSanity check:")
tests = [
    ("http://paypal-secure-login.tk/verify/account", "PHISHING"),
    ("https://www.google.com", "LEGITIMATE"),
    ("https://www.facebook.com", "LEGITIMATE"),
    ("http://192.168.1.1/banking/login.php", "PHISHING"),
    ("https://github.com/openai/gpt-4", "LEGITIMATE"),
]
for url, expected in tests:
    score = pipeline.predict_proba([url])[0][1]
    verdict = "PHISHING" if score >= 0.5 else "LEGITIMATE"
    status = "OK" if verdict == expected else "WRONG"
    print(f"  [{status}] {verdict} ({score:.4f}) - {url[:60]}")
