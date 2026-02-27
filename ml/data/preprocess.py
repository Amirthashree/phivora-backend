import pandas as pd
import os
import re

DATASET_BASE = "dataset"

LABEL_MAP = {
    'phishing': 1, 'bad': 1, 'malicious': 1, 'spam': 1,
    '1': 1, 1: 1, '-1': 1, -1: 1,
    'legitimate': 0, 'good': 0, 'benign': 0, 'safe': 0, 'ham': 0,
    '0': 0, 0: 0,
    'defacement': 1, 'malware': 1,
    'phishing email': 1, 'safe email': 0
}

def normalize_label(val):
    if val is None:
        return None
    v = str(val).strip().lower()
    return LABEL_MAP.get(v, None)

def load_train1():
    df = pd.read_csv(f"{DATASET_BASE}/data_train1/phishing_site_urls.csv")
    df = df.rename(columns={'URL': 'url', 'Label': 'label'})
    df['label'] = df['label'].apply(normalize_label)
    df['text'] = ''
    df['source'] = 'train1'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train2():
    df = pd.read_csv(f"{DATASET_BASE}/data_train2/malicious_phish.csv")
    df = df.rename(columns={'url': 'url', 'type': 'label'})
    df['label'] = df['label'].apply(normalize_label)
    df['text'] = ''
    df['source'] = 'train2'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train3():
    files = ['CEAS_08.csv', 'Enron.csv', 'Ling.csv',
             'Nazario.csv', 'Nigerian_Fraud.csv', 'phishing_email.csv', 'SpamAssasin.csv']
    dfs = []
    for f in files:
        try:
            path = f"{DATASET_BASE}/data_train3/{f}"
            df = pd.read_csv(path, on_bad_lines='skip')
            label_col = next((c for c in df.columns if c.lower() == 'label'), None)
            if label_col is None:
                continue
            text_col = next((c for c in ['body', 'text_combined', 'subject', 'message'] if c in df.columns), None)
            if text_col is None:
                continue
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            df['url'] = ''
            df['label'] = df['label'].apply(normalize_label)
            df['source'] = f'train3_{f}'
            dfs.append(df[['url', 'text', 'label', 'source']].dropna(subset=['label']))
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_train4():
    df = pd.read_csv(f"{DATASET_BASE}/data_train4/spam.csv", encoding='latin-1')
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    df['url'] = ''
    df['label'] = df['label'].apply(normalize_label)
    df['source'] = 'train4'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train5():
    df = pd.read_csv(f"{DATASET_BASE}/data_train5/dataset_phishing.csv")
    df = df.rename(columns={'url': 'url', 'status': 'label'})
    df['label'] = df['label'].apply(normalize_label)
    df['text'] = ''
    df['source'] = 'train5'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train6():
    df = pd.read_csv(f"{DATASET_BASE}/data_train6/phishing.csv")
    df = df.rename(columns={'Index': 'index', 'class': 'label'})
    df['url'] = ''
    df['text'] = ''
    df['label'] = df['label'].apply(normalize_label)
    df['source'] = 'train6'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train7():
    df = pd.read_csv(f"{DATASET_BASE}/data_train7/messages.csv")
    df = df.rename(columns={'message': 'text', 'label': 'label'})
    if 'text' not in df.columns and 'subject' in df.columns:
        df = df.rename(columns={'subject': 'text'})
    df['url'] = ''
    df['label'] = df['label'].apply(normalize_label)
    df['source'] = 'train7'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train8():
    df = pd.read_csv(f"{DATASET_BASE}/data_train8/dataset.csv")
    df = df.rename(columns={'URL': 'url', 'Type': 'label'})
    df['label'] = df['label'].apply(normalize_label)
    df['text'] = ''
    df['source'] = 'train8'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train9():
    df = pd.read_csv(f"{DATASET_BASE}/data_train9/data.csv")
    df = df.rename(columns={'url': 'url', 'label': 'label'})
    df['label'] = df['label'].apply(normalize_label)
    df['text'] = ''
    df['source'] = 'train9'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train10():
    df = pd.read_csv(f"{DATASET_BASE}/data_train10/dataset.csv")
    df = df.rename(columns={'Index': 'index', 'Result': 'label'})
    df['url'] = ''
    df['text'] = ''
    df['label'] = df['label'].apply(normalize_label)
    df['source'] = 'train10'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train11():
    df = pd.read_csv(f"{DATASET_BASE}/data_train11/emails.csv")
    df = df.rename(columns={'text': 'text', 'spam': 'label'})
    df['url'] = ''
    df['label'] = df['label'].apply(normalize_label)
    df['source'] = 'train11'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train12():
    df = pd.read_csv(f"{DATASET_BASE}/data_train12/new_data_urls.csv")
    df = df.rename(columns={'url': 'url', 'status': 'label'})
    df['label'] = df['label'].apply(normalize_label)
    df['text'] = ''
    df['source'] = 'train12'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train13():
    df = pd.read_csv(f"{DATASET_BASE}/data_train13/website_classification.csv")
    df = df.rename(columns={'website_url': 'url', 'cleaned_website_text': 'text'})
    df['label'] = 0
    df['text'] = df['text'].fillna('')
    df['url'] = df['url'].fillna('')
    df['source'] = 'train13'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train14():
    df = pd.read_csv(f"{DATASET_BASE}/data_train14/spam.csv", encoding='latin-1')
    if 'text' in df.columns and 'spam' in df.columns:
        df = df.rename(columns={'text': 'text', 'spam': 'label'})
    else:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    df['url'] = ''
    df['label'] = df['label'].apply(normalize_label)
    df['source'] = 'train14'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_train15():
    df = pd.read_csv(f"{DATASET_BASE}/data_train15/cybersecurity_attacks.csv")
    df['label'] = df['Attack Type'].apply(lambda x: 1 if str(x).strip().lower() != 'normal' else 0)
    df['url'] = df.get('Source IP Address', pd.Series([''] * len(df)))
    df['text'] = df.get('Payload Data', pd.Series([''] * len(df)))
    df['source'] = 'train15'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_test1():
    df = pd.read_csv(f"{DATASET_BASE}/data_test1/phishing.csv")
    df = df.rename(columns={'Index': 'index', 'class': 'label'})
    df['url'] = ''
    df['text'] = ''
    df['label'] = df['label'].apply(normalize_label)
    df['source'] = 'test1'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_test2():
    df = pd.read_csv(f"{DATASET_BASE}/data_test2/Phishing_Email.csv")
    df = df.rename(columns={'Email Text': 'text', 'Email Type': 'label'})
    df['url'] = ''
    df['label'] = df['label'].apply(normalize_label)
    df['source'] = 'test2'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_test3():
    df = pd.read_csv(f"{DATASET_BASE}/data_test3/malicious_phish.csv")
    df = df.rename(columns={'url': 'url', 'type': 'label'})
    df['label'] = df['label'].apply(normalize_label)
    df['text'] = ''
    df['source'] = 'test3'
    return df[['url', 'text', 'label', 'source']].dropna(subset=['label'])

def load_all_train():
    loaders = [
        load_train1, load_train2, load_train3, load_train4, load_train5,
        load_train6, load_train7, load_train8, load_train9, load_train10,
        load_train11, load_train12, load_train13, load_train14, load_train15
    ]
    dfs = []
    for loader in loaders:
        try:
            df = loader()
            if df is not None and len(df) > 0:
                print(f"Loaded {loader.__name__}: {len(df)} rows")
                dfs.append(df)
        except Exception as e:
            print(f"Error in {loader.__name__}: {e}")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=['label'])
    combined['label'] = combined['label'].astype(int)
    print(f"\nTotal training rows: {len(combined)}")
    print(f"Phishing: {combined['label'].sum()} | Legitimate: {(combined['label']==0).sum()}")
    return combined

def load_all_test():
    loaders = [load_test1, load_test2, load_test3]
    dfs = []
    for loader in loaders:
        try:
            df = loader()
            if df is not None and len(df) > 0:
                print(f"Loaded {loader.__name__}: {len(df)} rows")
                dfs.append(df)
        except Exception as e:
            print(f"Error in {loader.__name__}: {e}")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=['label'])
    combined['label'] = combined['label'].astype(int)
    print(f"\nTotal testing rows: {len(combined)}")
    return combined

if __name__ == "__main__":
    print("=== Loading Training Data ===")
    train = load_all_train()
    train.to_csv("dataset/combined_train.csv", index=False)
    print("Saved to dataset/combined_train.csv")

    print("\n=== Loading Testing Data ===")
    test = load_all_test()
    test.to_csv("dataset/combined_test.csv", index=False)
    print("Saved to dataset/combined_test.csv")

