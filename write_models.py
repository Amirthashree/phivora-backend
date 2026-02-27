import os

os.makedirs("ml/models", exist_ok=True)
os.makedirs("ml/training", exist_ok=True)
os.makedirs("ml/evaluation", exist_ok=True)

files = {}

files["ml/models/xgboost_model.py"] = """
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

URL_FEATURE_COLS = [
    'url_length','domain_length','path_length','num_dots','num_hyphens',
    'num_underscores','num_slashes','num_at','num_question','num_ampersand',
    'num_equal','num_percent','num_digits','digit_ratio','letter_ratio',
    'url_entropy','domain_entropy','has_ip','is_https','has_port',
    'num_subdomains','is_shortener','has_suspicious_keyword',
    'suspicious_keyword_count','has_double_slash','has_prefix_suffix',
    'query_length','num_query_params','has_homoglyph','brand_impersonation_count',
    'tld_risk','is_url_row','is_text_row','suspicious_and_short',
    'ip_and_http','many_dots_short_domain'
]

def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=200000,
            sublinear_tf=True,
            min_df=2,
            strip_accents='unicode',
            decode_error='replace'
        )),
        ('scaler', MaxAbsScaler()),
        ('clf', SGDClassifier(
            loss='modified_huber',
            alpha=1e-3,
            max_iter=200,
            tol=1e-3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_xgboost(model, X_val, y_val):
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
"""

files["ml/models/random_forest_model.py"] = """
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import re

def url_tokenizer(u):
    return [t for t in re.split(r'[/.\\-_?=&#@%+]', u.lower()) if len(t) > 1]

def train_random_forest(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='word',
            tokenizer=url_tokenizer,
            ngram_range=(1, 2),
            max_features=200000,
            sublinear_tf=True,
            min_df=2,
            decode_error='replace',
            token_pattern=None
        )),
        ('scaler', MaxAbsScaler()),
        ('clf', SGDClassifier(
            loss='log_loss',
            alpha=1e-3,
            max_iter=200,
            tol=1e-3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_random_forest(model, X_val, y_val):
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
"""

files["ml/models/sgd_model.py"] = """
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train_sgd(X_train, y_train):
    pipeline = Pipeline([
        ('cv', CountVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=200000,
            min_df=2,
            strip_accents='unicode',
            decode_error='replace'
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_sgd(model, X_val, y_val):
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
"""

files["ml/models/naive_bayes_model.py"] = """
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
"""

files["ml/models/ensemble.py"] = """
import pickle
import os
import sys
sys.path.append(os.path.abspath('.'))

class PhishingEnsemble:
    def __init__(self, model_dir='ml/saved_models'):
        self.model_dir   = model_dir
        self.model1      = None
        self.model2      = None
        self.model3      = None
        self.naive_bayes = None

    def load_models(self):
        with open(f'{self.model_dir}/xgboost.pkl',       'rb') as f: self.model1      = pickle.load(f)
        with open(f'{self.model_dir}/random_forest.pkl',  'rb') as f: self.model2      = pickle.load(f)
        with open(f'{self.model_dir}/sgd.pkl',            'rb') as f: self.model3      = pickle.load(f)
        with open(f'{self.model_dir}/naive_bayes.pkl',    'rb') as f: self.naive_bayes = pickle.load(f)
        print('All models loaded')

    def predict(self, url='', text=''):
        url  = str(url  or '').strip()
        text = str(text or '').strip()
        has_url  = len(url)  > 3 and url  != 'nan'
        has_text = len(text) > 10 and text != 'nan'

        if has_text and not has_url:   data_type = 'text'
        elif has_url and not has_text: data_type = 'url'
        elif has_url and has_text:     data_type = 'both'
        else:                          data_type = 'unknown'

        scores = {}
        if has_url:
            scores['model1'] = float(self.model1.predict_proba([url])[0][1])
            scores['model2'] = float(self.model2.predict_proba([url])[0][1])
            scores['model3'] = float(self.model3.predict_proba([url])[0][1])
        if has_text:
            scores['model4'] = float(self.naive_bayes.predict_proba([text])[0][1])

        if data_type == 'text':
            final = scores.get('model4', 0.5)
        elif data_type == 'url':
            final = (scores.get('model1',0.5)*0.45 +
                     scores.get('model2',0.5)*0.20 +
                     scores.get('model3',0.5)*0.35)
        elif data_type == 'both':
            final = (scores.get('model1',0.5)*0.35 +
                     scores.get('model2',0.5)*0.15 +
                     scores.get('model3',0.5)*0.25 +
                     scores.get('model4',0.5)*0.25)
        else:
            final = 0.5

        threshold = float(os.getenv('PHISHING_THRESHOLD', '0.5'))
        return {
            'label':        1 if final >= threshold else 0,
            'verdict':      'PHISHING' if final >= threshold else 'LEGITIMATE',
            'confidence':   round(final, 4),
            'data_type':    data_type,
            'model_scores': scores
        }
"""

files["ml/training/train_xgboost.py"] = """
import pandas as pd
import numpy as np
import pickle
import sys, os
sys.path.append(os.path.abspath('.'))
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml.models.xgboost_model import train_xgboost, evaluate_xgboost, URL_FEATURE_COLS

def run_train_xgboost():
    print('=== Model 1: SGD modified_huber + char ngrams(3,5) | train2+train5 ===')
    df = pd.read_csv('dataset/combined_train.csv', low_memory=False)
    df['url'] = df['url'].fillna('').astype(str)
    df = df[(df['url'].str.strip()!='')&(df['url']!='nan')&(df['url'].str.len()>5)]
    df = df[df['source'].isin(['train2','train5'])]
    p = df[df['label']==1]; l = df[df['label']==0]
    n = min(len(p), len(l), 500000)
    bal = pd.concat([p.sample(n,random_state=42), l.sample(n,random_state=42)]).sample(frac=1,random_state=42)
    X_tr,X_val,y_tr,y_val = train_test_split(bal['url'].tolist(),bal['label'].values,test_size=0.1,random_state=42,stratify=bal['label'].values)
    print(f'Train: {len(X_tr)} | Val: {len(X_val)}')
    model = train_xgboost(X_tr, y_tr)
    tr_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_acc= accuracy_score(y_val,model.predict(X_val))
    print(f'Train: {tr_acc:.4f} | Val: {val_acc:.4f}')
    print('PASSED' if tr_acc-val_acc<=0.05 else f'WARNING gap={tr_acc-val_acc:.4f}')
    acc,report,_ = evaluate_xgboost(model,X_val,y_val)
    print(f'Final Val Accuracy: {acc:.4f}'); print(report)
    os.makedirs('ml/saved_models', exist_ok=True)
    with open('ml/saved_models/xgboost.pkl','wb') as f: pickle.dump(model,f)
    with open('ml/saved_models/feature_cols.pkl','wb') as f: pickle.dump(URL_FEATURE_COLS,f)
    print('Saved xgboost.pkl')
    return model, URL_FEATURE_COLS

if __name__ == '__main__':
    run_train_xgboost()
"""

files["ml/training/train_random_forest.py"] = """
import pandas as pd
import numpy as np
import pickle
import sys, os
sys.path.append(os.path.abspath('.'))
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml.models.random_forest_model import train_random_forest, evaluate_random_forest

def run_train_random_forest():
    print('=== Model 2: SGD log_loss + word tokens | train12 ===')
    df = pd.read_csv('dataset/combined_train.csv', low_memory=False)
    df['url'] = df['url'].fillna('').astype(str)
    df = df[(df['url'].str.strip()!='')&(df['url']!='nan')&(df['url'].str.len()>5)]
    df = df[df['source']=='train12']
    p = df[df['label']==1]; l = df[df['label']==0]
    n = min(len(p), len(l), 500000)
    bal = pd.concat([p.sample(n,random_state=42), l.sample(n,random_state=42)]).sample(frac=1,random_state=42)
    X_tr,X_val,y_tr,y_val = train_test_split(bal['url'].tolist(),bal['label'].values,test_size=0.1,random_state=42,stratify=bal['label'].values)
    print(f'Train: {len(X_tr)} | Val: {len(X_val)}')
    model = train_random_forest(X_tr, y_tr)
    tr_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_acc= accuracy_score(y_val,model.predict(X_val))
    print(f'Train: {tr_acc:.4f} | Val: {val_acc:.4f}')
    print('PASSED' if tr_acc-val_acc<=0.05 else f'WARNING gap={tr_acc-val_acc:.4f}')
    acc,report,_ = evaluate_random_forest(model,X_val,y_val)
    print(f'Final Val Accuracy: {acc:.4f}'); print(report)
    os.makedirs('ml/saved_models', exist_ok=True)
    with open('ml/saved_models/random_forest.pkl','wb') as f: pickle.dump(model,f)
    print('Saved random_forest.pkl')
    return model

if __name__ == '__main__':
    run_train_random_forest()
"""

files["ml/training/train_sgd.py"] = """
import pandas as pd
import numpy as np
import pickle
import sys, os
sys.path.append(os.path.abspath('.'))
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml.models.sgd_model import train_sgd, evaluate_sgd

def run_train_sgd():
    print('=== Model 3: MultinomialNB + char ngrams(2,4) | train1+train5 ===')
    df = pd.read_csv('dataset/combined_train.csv', low_memory=False)
    df['url'] = df['url'].fillna('').astype(str)
    df = df[(df['url'].str.strip()!='')&(df['url']!='nan')&(df['url'].str.len()>5)]
    df = df[df['source'].isin(['train1','train5'])]
    p = df[df['label']==1]; l = df[df['label']==0]
    n = min(len(p), len(l), 500000)
    bal = pd.concat([p.sample(n,random_state=42), l.sample(n,random_state=42)]).sample(frac=1,random_state=42)
    X_tr,X_val,y_tr,y_val = train_test_split(bal['url'].tolist(),bal['label'].values,test_size=0.1,random_state=42,stratify=bal['label'].values)
    print(f'Train: {len(X_tr)} | Val: {len(X_val)}')
    model = train_sgd(X_tr, y_tr)
    tr_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_acc= accuracy_score(y_val,model.predict(X_val))
    print(f'Train: {tr_acc:.4f} | Val: {val_acc:.4f}')
    print('PASSED' if tr_acc-val_acc<=0.05 else f'WARNING gap={tr_acc-val_acc:.4f}')
    acc,report,_ = evaluate_sgd(model,X_val,y_val)
    print(f'Final Val Accuracy: {acc:.4f}'); print(report)
    os.makedirs('ml/saved_models', exist_ok=True)
    with open('ml/saved_models/sgd.pkl','wb') as f: pickle.dump(model,f)
    print('Saved sgd.pkl')
    return model

if __name__ == '__main__':
    run_train_sgd()
"""

files["ml/training/train_naive_bayes.py"] = """
import pandas as pd
import numpy as np
import pickle
import sys, os
sys.path.append(os.path.abspath('.'))
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml.models.naive_bayes_model import train_naive_bayes, evaluate_naive_bayes

def run_train_naive_bayes():
    print('=== Training Naive Bayes Model ===')
    df = pd.read_csv('dataset/combined_train.csv', low_memory=False)
    df = df.fillna('')
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != '']
    print(f'Text samples: {len(df)}')
    X_tr,X_val,y_tr,y_val = train_test_split(df['text'].tolist(),df['label'].values,test_size=0.1,random_state=42,stratify=df['label'].values)
    print(f'Train: {len(X_tr)} | Val: {len(X_val)}')
    model = train_naive_bayes(X_tr, y_tr)
    tr_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_acc= accuracy_score(y_val,model.predict(X_val))
    print(f'Train: {tr_acc:.4f} | Val: {val_acc:.4f}')
    print('PASSED' if tr_acc-val_acc<=0.05 else f'WARNING gap={tr_acc-val_acc:.4f}')
    acc,report,_ = evaluate_naive_bayes(model,X_val,y_val)
    print(f'Final Val Accuracy: {acc:.4f}'); print(report)
    os.makedirs('ml/saved_models', exist_ok=True)
    with open('ml/saved_models/naive_bayes.pkl','wb') as f: pickle.dump(model,f)
    print('Saved naive_bayes.pkl')
    return model

if __name__ == '__main__':
    run_train_naive_bayes()
"""

files["ml/training/train_all.py"] = """
import sys, os
sys.path.append(os.path.abspath('.'))
from ml.training.train_xgboost      import run_train_xgboost
from ml.training.train_random_forest import run_train_random_forest
from ml.training.train_naive_bayes  import run_train_naive_bayes
from ml.training.train_sgd          import run_train_sgd

if __name__ == '__main__':
    print('='*60)
    print('TRAINING ALL MODELS')
    print('='*60)
    print('[1/4] Model 1 - SGD char ngrams'); run_train_xgboost()
    print('[2/4] Model 2 - SGD word tokens'); run_train_random_forest()
    print('[3/4] Model 3 - MultinomialNB');   run_train_sgd()
    print('[4/4] Naive Bayes text');           run_train_naive_bayes()
    print('='*60)
    print('ALL MODELS TRAINED AND SAVED')
    print('='*60)
"""

files["ml/evaluation/evaluate.py"] = """
import pickle
import pandas as pd
import sys, os
sys.path.append(os.path.abspath('.'))
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ml.models.ensemble import PhishingEnsemble

def evaluate_all():
    print('=== EVALUATING ALL MODELS ===')
    ensemble = PhishingEnsemble()
    ensemble.load_models()

    # URL test data
    df = pd.read_csv('dataset/combined_test.csv', low_memory=False)
    df['url']  = df['url'].fillna('').astype(str)
    df['text'] = df['text'].fillna('').astype(str)
    url_df  = df[(df['url']!='')&(df['url']!='nan')&(df['url'].str.len()>5)]
    text_df = df[(df['text']!='')&(df['text']!='nan')&(df['text'].str.len()>10)]

    print(f'URL test rows:  {len(url_df)}')
    print(f'Text test rows: {len(text_df)}')

    # Model 1
    if len(url_df) > 0:
        sample = url_df.sample(min(10000,len(url_df)), random_state=42)
        preds = ensemble.model1.predict(sample['url'].tolist())
        acc = accuracy_score(sample['label'].values, preds)
        print(f'Model1 (char ngram) accuracy: {acc:.4f}')

    # Model 2
    if len(url_df) > 0:
        sample = url_df.sample(min(10000,len(url_df)), random_state=42)
        preds = ensemble.model2.predict(sample['url'].tolist())
        acc = accuracy_score(sample['label'].values, preds)
        print(f'Model2 (word token) accuracy: {acc:.4f}')

    # Model 3
    if len(url_df) > 0:
        sample = url_df.sample(min(10000,len(url_df)), random_state=42)
        preds = ensemble.model3.predict(sample['url'].tolist())
        acc = accuracy_score(sample['label'].values, preds)
        print(f'Model3 (MultinomialNB URL) accuracy: {acc:.4f}')

    # Naive Bayes
    if len(text_df) > 0:
        sample = text_df.sample(min(5000,len(text_df)), random_state=42)
        preds = ensemble.naive_bayes.predict(sample['text'].tolist())
        acc = accuracy_score(sample['label'].values, preds)
        print(f'Model4 (Naive Bayes text) accuracy: {acc:.4f}')

if __name__ == '__main__':
    evaluate_all()
"""

files["ml/evaluation/metrics.py"] = """
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = round(roc_auc_score(y_true, y_proba), 4)
        except Exception:
            pass
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    return metrics
"""

for path, content in files.items():
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f'Written: {path}')

print('\nALL MODEL CODE FILES WRITTEN SUCCESSFULLY')
