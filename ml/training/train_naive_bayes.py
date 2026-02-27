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