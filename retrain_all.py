import sys
import pandas as pd
import numpy as np
import pickle
import os
import re
import math
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
import xgboost as xgb
sys.path.append(".")

os.makedirs("ml/saved_models", exist_ok=True)

# ── Feature extraction (same as before) ──────────────────────
SUSPICIOUS_TLDS = [".tk",".ml",".ga",".cf",".gq",".xyz",".top",".work",".click",".link",".online",".site",".info",".biz"]
TRUSTED_DOMAINS = ["google","facebook","amazon","microsoft","apple","paypal","netflix","instagram","twitter","linkedin","youtube","github","stackoverflow","wikipedia"]

def calculate_entropy(text):
    if not text: return 0
    freq = {}
    for c in str(text): freq[c] = freq.get(c,0)+1
    return round(-sum((c/len(text))*math.log2(c/len(text)) for c in freq.values()),4)

def extract_features(url, text=""):
    url = str(url or ""); text = str(text or "")
    f = {}
    try:
        parsed = urlparse(url if url.startswith("http") else "http://"+url)
        domain = parsed.netloc.lower(); path = parsed.path; query = parsed.query; full = url.lower()
    except: domain = path = query = full = ""
    f["url_length"]               = len(url)
    f["domain_length"]            = len(domain)
    f["path_length"]              = len(path)
    f["num_dots"]                 = url.count(".")
    f["num_hyphens"]              = url.count("-")
    f["num_underscores"]          = url.count("_")
    f["num_slashes"]              = url.count("/")
    f["num_at"]                   = url.count("@")
    f["num_question"]             = url.count("?")
    f["num_ampersand"]            = url.count("&")
    f["num_equal"]                = url.count("=")
    f["num_percent"]              = url.count("%")
    f["num_digits"]               = sum(c.isdigit() for c in url)
    f["digit_ratio"]              = round(sum(c.isdigit() for c in url)/max(len(url),1),4)
    f["letter_ratio"]             = round(sum(c.isalpha() for c in url)/max(len(url),1),4)
    f["url_entropy"]              = calculate_entropy(url)
    f["domain_entropy"]           = calculate_entropy(domain)
    f["has_ip"]                   = 1 if re.match(r"https?://\d+\.\d+\.\d+\.\d+",url) else 0
    f["is_https"]                 = 1 if url.startswith("https") else 0
    f["has_port"]                 = 1 if re.search(r":\d{2,5}",domain) else 0
    f["num_subdomains"]           = max(len(domain.split("."))-2,0)
    f["is_shortener"]             = 1 if any(s in url for s in ["bit.ly","tinyurl","goo.gl","t.co","ow.ly"]) else 0
    f["has_suspicious_keyword"]   = 1 if any(k in full for k in ["login","verify","secure","update","account","banking","confirm","password","suspend","unlock"]) else 0
    f["suspicious_keyword_count"] = sum(1 for k in ["login","verify","secure","update","account","banking","confirm","password","suspend","unlock"] if k in full)
    f["has_double_slash"]         = 1 if "//" in path else 0
    f["has_prefix_suffix"]        = 1 if "-" in domain else 0
    f["query_length"]             = len(query)
    f["num_query_params"]         = len(query.split("&")) if query else 0
    hmap = {"0":"o","1":"l","3":"e","4":"a","5":"s","@":"a","$":"s","vv":"w","rn":"m"}
    norm = url.lower()
    for fake,real in hmap.items(): norm = norm.replace(fake,real)
    f["has_homoglyph"]             = 1 if any(b in norm and b not in url.lower() for b in TRUSTED_DOMAINS) else 0
    f["brand_impersonation_count"] = sum(1 for b in TRUSTED_DOMAINS if b in url.lower())
    f["has_suspicious_tld"]        = 1 if any(url.lower().endswith(t) or t+"/" in url.lower() for t in SUSPICIOUS_TLDS) else 0
    f["domain_has_numbers"]        = 1 if re.search(r"\d",domain) else 0
    f["path_has_exe"]              = 1 if any(url.lower().endswith(e) for e in [".exe",".zip",".rar",".php",".js"]) else 0
    f["has_redirect_sign"]         = 1 if "redirect" in full or "forward" in full or "url=" in full else 0
    f["dots_in_domain"]            = domain.count(".")
    f["domain_token_count"]        = len(re.findall(r"[a-zA-Z0-9]+",domain))
    f["path_token_count"]          = len(re.findall(r"[a-zA-Z0-9]+",path))
    f["is_trusted_domain"]         = 1 if any(t in domain for t in TRUSTED_DOMAINS) else 0
    f["url_has_encoded_chars"]     = 1 if "%" in url else 0
    f["url_depth"]                 = path.count("/")
    f["domain_word_count"]         = len(re.findall(r"[a-z]{3,}",domain))
    f["ratio_special_chars"]       = round(sum(1 for c in url if not c.isalnum())/max(len(url),1),4)
    f["is_url_row"]                = 1 if len(url) > 5 else 0
    f["suspicious_and_short"]      = 1 if f["has_suspicious_keyword"] and len(url)<50 else 0
    f["ip_and_http"]               = 1 if f["has_ip"] and not f["is_https"] else 0
    f["many_dots_short_domain"]    = 1 if f["num_dots"]>3 and f["domain_length"]<20 else 0
    f["suspicious_tld_keyword"]    = 1 if f["has_suspicious_tld"] and f["has_suspicious_keyword"] else 0
    f["impersonation_no_https"]    = 1 if f["brand_impersonation_count"]>0 and not f["is_https"] else 0
    f["long_subdomain"]            = 1 if f["num_subdomains"]>3 else 0
    f["encoded_and_suspicious"]    = 1 if f["url_has_encoded_chars"] and f["has_suspicious_keyword"] else 0
    return f

# ── Load clean data ───────────────────────────────────────────
print("Loading clean data...")
df = pd.read_csv("dataset/combined_train_clean.csv", low_memory=False)
df = df.fillna("").reset_index(drop=True)
df["url"]   = df["url"].astype(str)
df["text"]  = df["text"].astype(str)
df["label"] = df["label"].astype(int)

df_url  = df[(df["url"].str.len()>5) & (df["url"]!="nan")].reset_index(drop=True)
df_text = df[(df["text"].str.len()>10) & (df["text"]!="nan")].reset_index(drop=True)
print(f"URL rows : {len(df_url):,}  (phish={(df_url['label']==1).sum():,} | legit={(df_url['label']==0).sum():,})")
print(f"Text rows: {len(df_text):,} (phish={(df_text['label']==1).sum():,} | legit={(df_text['label']==0).sum():,})")

# ── Balance URL data ──────────────────────────────────────────
phish = df_url[df_url["label"]==1]
legit = df_url[df_url["label"]==0]
n     = min(len(phish), len(legit), 400000)
df_bal = pd.concat([phish.sample(n,random_state=42), legit.sample(n,random_state=42)]).sample(frac=1,random_state=42).reset_index(drop=True)
print(f"\nBalanced URL data: {len(df_bal):,} ({n:,} each)")

# ── Extract features ──────────────────────────────────────────
print("Extracting features (this takes a few minutes)...")
feature_cols = None
rows = []
for i, row in df_bal.iterrows():
    f = extract_features(row["url"])
    f["label"] = row["label"]
    rows.append(f)
    if i % 50000 == 0: print(f"  {i:,}/{len(df_bal):,}")
feat_df = pd.DataFrame(rows).fillna(0)
feature_cols = [c for c in feat_df.columns if c != "label"]
X = feat_df[feature_cols].values.astype(float)
y = feat_df["label"].values
print(f"Feature matrix: {X.shape}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.111, random_state=42, stratify=y_tr)
print(f"Train={len(X_tr):,} Val={len(X_va):,} Test={len(X_te):,}")

def evaluate(name, model, X_t, y_t, X_v, y_v):
    tr_acc = accuracy_score(y_t, model.predict(X_t))
    va_acc = accuracy_score(y_v, model.predict(X_v))
    va_roc = roc_auc_score(y_v, model.predict_proba(X_v)[:,1])
    gap    = tr_acc - va_acc
    print(f"\n{'='*45}")
    print(f"[{name}]")
    print(f"Train Acc : {tr_acc:.4f}")
    print(f"Val Acc   : {va_acc:.4f}")
    print(f"ROC-AUC   : {va_roc:.4f}")
    print(f"Overfit   : {gap:.4f} ({'WARNING' if gap>0.03 else 'OK'})")
    print(f"90% check : {'PASSED' if va_acc>=0.90 else 'FAILED'}")
    print(classification_report(y_v, model.predict(X_v)))

# ── 1. XGBoost ────────────────────────────────────────────────
print("\n>>> Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, n_jobs=-1
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)
evaluate("XGBoost", xgb_model, X_tr, y_tr, X_va, y_va)
with open("ml/saved_models/xgboost.pkl","wb") as f: pickle.dump(xgb_model, f)
print("Saved xgboost.pkl")

# ── 2. Random Forest ──────────────────────────────────────────
print("\n>>> Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_split=10,
    min_samples_leaf=5, max_features="sqrt", bootstrap=True,
    oob_score=True, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_model.fit(X_tr, y_tr)
print(f"OOB Score: {rf_model.oob_score_:.4f}")
evaluate("Random Forest", rf_model, X_tr, y_tr, X_va, y_va)
with open("ml/saved_models/random_forest.pkl","wb") as f: pickle.dump(rf_model, f)
print("Saved random_forest.pkl")

# ── 3. SGD (Gradient Boosting on features) ────────────────────
print("\n>>> Training Gradient Boosting...")
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.7, max_features=0.7, min_samples_split=20,
    min_samples_leaf=10, max_leaf_nodes=31,
    validation_fraction=0.1, n_iter_no_change=20,
    tol=1e-4, ccp_alpha=0.0001, random_state=42, verbose=1
)
gb_model.fit(X_tr, y_tr)
evaluate("Gradient Boosting", gb_model, X_tr, y_tr, X_va, y_va)
with open("ml/saved_models/sgd.pkl","wb") as f: pickle.dump(gb_model, f)
print("Saved sgd.pkl")

# ── 4. Naive Bayes (text) ─────────────────────────────────────
print("\n>>> Training Naive Bayes...")
phish_t = df_text[df_text["label"]==1]
legit_t = df_text[df_text["label"]==0]
n_t     = min(len(phish_t), len(legit_t))
df_text_bal = pd.concat([phish_t.sample(n_t,random_state=42), legit_t.sample(n_t,random_state=42)]).sample(frac=1,random_state=42)
X_txt = df_text_bal["text"].astype(str).tolist()
y_txt = df_text_bal["label"].values
X_txt_tr, X_txt_te, y_txt_tr, y_txt_te = train_test_split(X_txt, y_txt, test_size=0.1, random_state=42, stratify=y_txt)
nb_model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=100000, sublinear_tf=True, min_df=2, strip_accents="unicode", decode_error="replace")),
    ("clf",   ComplementNB(alpha=0.1))
])
nb_model.fit(X_txt_tr, y_txt_tr)
nb_acc = accuracy_score(y_txt_te, nb_model.predict(X_txt_te))
nb_roc = roc_auc_score(y_txt_te, nb_model.predict_proba(X_txt_te)[:,1])
print(f"\n[Naive Bayes] Acc={nb_acc:.4f} ROC={nb_roc:.4f}")
print(f"90% check: {'PASSED' if nb_acc>=0.90 else 'FAILED'}")
print(classification_report(y_txt_te, nb_model.predict(X_txt_te)))
with open("ml/saved_models/naive_bayes.pkl","wb") as f: pickle.dump(nb_model, f)
print("Saved naive_bayes.pkl")

# ── Save feature cols ─────────────────────────────────────────
with open("ml/saved_models/feature_cols.pkl","wb") as f: pickle.dump(feature_cols, f)
print(f"\nSaved feature_cols.pkl ({len(feature_cols)} features)")

# ── Sanity check ──────────────────────────────────────────────
print("\n=== SANITY CHECK ===")
test_urls = [
    ("http://paypal-secure-login.tk/verify/account",      1, "PHISHING"),
    ("https://www.google.com",                            0, "LEGITIMATE"),
    ("https://www.github.com/openai/gpt-4",               0, "LEGITIMATE"),
    ("http://192.168.1.1/banking/login.php",              1, "PHISHING"),
    ("http://apple-id-verify.cf/confirm/password",        1, "PHISHING"),
    ("https://www.amazon.com/dp/B09G9HD6PD",              0, "LEGITIMATE"),
]
for url, expected_label, expected_verdict in test_urls:
    f    = extract_features(url)
    xv   = np.array([f.get(c,0) for c in feature_cols], dtype=float).reshape(1,-1)
    xp   = xgb_model.predict_proba(xv)[0][1]
    rfp  = rf_model.predict_proba(xv)[0][1]
    gbp  = gb_model.predict_proba(xv)[0][1]
    avg  = round((xp*0.35 + rfp*0.25 + gbp*0.25 + 0.15*0.5), 4)
    pred = "PHISHING" if avg >= 0.5 else "LEGITIMATE"
    status = "OK" if pred == expected_verdict else "WRONG"
    print(f"  [{status}] {pred} (xgb={xp:.3f} rf={rfp:.3f} gb={gbp:.3f} avg={avg:.3f}) {url[:60]}")

print("\nAll models saved. Restart your server.")
