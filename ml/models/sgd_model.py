from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def get_sgd_model():
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=200000,
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            strip_accents='unicode'
        )),
        ('sgd', SGDClassifier(
            loss='modified_huber',
            penalty='elasticnet',
            alpha=0.00001,
            l1_ratio=0.15,
            max_iter=200,
            tol=1e-4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    return model

def train_sgd(urls, labels):
    urls = [str(u) if u else '' for u in urls]
    model = get_sgd_model()
    model.fit(urls, labels)
    return model

def evaluate_sgd(model, urls, y_test):
    urls  = [str(u) if u else '' for u in urls]
    preds = model.predict(urls)
    proba = model.predict_proba(urls)[:, 1]
    acc   = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report, proba

def predict_sgd(model, urls):
    urls = [str(u) if u else '' for u in urls]
    return model.predict_proba(urls)[:, 1]
