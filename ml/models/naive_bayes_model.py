from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import numpy as np

def train_naive_bayes(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True,
            min_df=2,
            strip_accents='unicode',
            decode_error='replace'
        )),
        ('clf', ComplementNB(alpha=0.1))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_naive_bayes(model, X_val, y_val):
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    report = classification_report(y_val, preds)
    try:
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        print(f'ROC-AUC: {auc:.4f}')
    except Exception:
        pass
    return acc, report, preds