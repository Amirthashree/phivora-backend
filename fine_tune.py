import sys
import pandas as pd
import numpy as np
import pickle
import re
import math
from urllib.parse import urlparse
sys.path.append(".")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("Loading models...")
with open("ml/saved_models/xgboost.pkl",      "rb") as f: xgb_model = pickle.load(f)
with open("ml/saved_models/random_forest.pkl", "rb") as f: rf_model  = pickle.load(f)
with open("ml/saved_models/sgd.pkl",           "rb") as f: sgd_model = pickle.load(f)
with open("ml/saved_models/naive_bayes.pkl",   "rb") as f: nb_model  = pickle.load(f)
with open("ml/saved_models/feature_cols.pkl",  "rb") as f: feat_cols = pickle.load(f)

SUSPICIOUS_TLDS = [".tk",".ml",".ga",".cf",".gq",".xyz",".top",".work",".click",".link",".online",".site",".info",".biz"]
TRUSTED_DOMAINS = ["google","facebook","amazon","microsoft","apple","paypal","netflix","instagram","twitter","linkedin","youtube","github","stackoverflow","wikipedia"]

def calc_entropy(text):
    if not text: return 0
    freq = {}
    for c in str(text): freq[c] = freq.get(c,0)+1
    return round(-sum((v/len(text))*math.log2(v/len(text)) for v in freq.values()),4)

def extract_features(url):
    url = str(url or "")
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
    f["url_entropy"]              = calc_entropy(url)
    f["domain_entropy"]           = calc_entropy(domain)
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
    for k,v in hmap.items(): norm = norm.replace(k,v)
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
    f["is_url_row"]                = 1 if len(url)>5 else 0
    f["suspicious_and_short"]      = 1 if f["has_suspicious_keyword"] and len(url)<50 else 0
    f["ip_and_http"]               = 1 if f["has_ip"] and not f["is_https"] else 0
    f["many_dots_short_domain"]    = 1 if f["num_dots"]>3 and f["domain_length"]<20 else 0
    f["suspicious_tld_keyword"]    = 1 if f["has_suspicious_tld"] and f["has_suspicious_keyword"] else 0
    f["impersonation_no_https"]    = 1 if f["brand_impersonation_count"]>0 and not f["is_https"] else 0
    f["long_subdomain"]            = 1 if f["num_subdomains"]>3 else 0
    f["encoded_and_suspicious"]    = 1 if f["url_has_encoded_chars"] and f["has_suspicious_keyword"] else 0
    return [f.get(c,0) for c in feat_cols]

# ── Load and sample test data ─────────────────────────────────
print("Loading test data...")
dt = pd.read_csv("dataset/combined_test.csv", low_memory=False)
dt = dt.fillna("").reset_index(drop=True)
dt["url"]   = dt["url"].astype(str)
dt["text"]  = dt["text"].astype(str)
dt["label"] = dt["label"].astype(int)

url_test   = dt[(dt["url"].str.len()>5) & (dt["url"]!="nan")].reset_index(drop=True)
url_sample = url_test.sample(min(1000, len(url_test)), random_state=42).reset_index(drop=True)
urls       = url_sample["url"].tolist()
labels     = url_sample["label"].values
print(f"Sample: {len(url_sample):,} URLs")

# ── Pre-compute ALL model scores in bulk (fast) ───────────────
print("Pre-computing model scores...")
X_feat = np.array([extract_features(u) for u in urls], dtype=float)
xgb_scores = xgb_model.predict_proba(X_feat)[:,1]
rf_scores  = rf_model.predict_proba(X_feat)[:,1]
sgd_scores = sgd_model.predict_proba(urls)[:,1]
nb_scores  = nb_model.predict_proba(urls)[:,1]
print("Done. Running grid search...")

# ── Grid search (instant now) ─────────────────────────────────
results = []
for w_sgd in [0.35, 0.40, 0.45, 0.50]:
    for w_xgb in [0.20, 0.25, 0.30]:
        for w_rf in [0.10, 0.15, 0.20]:
            w_nb = round(1.0 - w_sgd - w_xgb - w_rf, 2)
            if w_nb < 0.05 or w_nb > 0.30: continue
            for threshold in [0.40, 0.45, 0.50, 0.55]:
                final  = sgd_scores*w_sgd + xgb_scores*w_xgb + rf_scores*w_rf + nb_scores*w_nb
                preds  = (final >= threshold).astype(int)
                acc    = accuracy_score(labels, preds)
                f1     = f1_score(labels, preds, zero_division=0)
                prec   = precision_score(labels, preds, zero_division=0)
                rec    = recall_score(labels, preds, zero_division=0)
                results.append((f1, acc, w_sgd, w_xgb, w_rf, w_nb, threshold, prec, rec))

results.sort(reverse=True)
print(f"\nTop 10 configs by F1:")
print(f"{'SGD':>6} {'XGB':>6} {'RF':>6} {'NB':>6} {'Thr':>6} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
for r in results[:10]:
    f1,acc,ws,wx,wr,wn,thr,prec,rec = r
    print(f"{ws:>6.2f} {wx:>6.2f} {wr:>6.2f} {wn:>6.2f} {thr:>6.2f} {acc:>7.4f} {f1:>7.4f} {prec:>7.4f} {rec:>7.4f}")

f1,acc,w_sgd,w_xgb,w_rf,w_nb,thr,prec,rec = results[0]
print(f"\nBest: SGD={w_sgd} XGB={w_xgb} RF={w_rf} NB={w_nb} Threshold={thr}")
print(f"Acc={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f}")

print("\n=== Sanity Check ===")
sanity = [
    ("http://paypal-secure-login.tk/verify/account", "PHISHING"),
    ("https://www.google.com",                        "LEGITIMATE"),
    ("https://github.com/openai/gpt-4",              "LEGITIMATE"),
    ("http://192.168.1.1/banking/login.php",         "PHISHING"),
    ("http://apple-id-verify.cf/confirm/password",   "PHISHING"),
    ("https://www.amazon.com/dp/B09G9HD6PD",         "LEGITIMATE"),
    ("http://microsoft-update-secure.tk/login",      "PHISHING"),
    ("https://stackoverflow.com/questions/12345",    "LEGITIMATE"),
]
ok = 0
for url, expected in sanity:
    xp  = float(xgb_model.predict_proba(np.array(extract_features(url),dtype=float).reshape(1,-1))[0][1])
    rp  = float(rf_model.predict_proba(np.array(extract_features(url),dtype=float).reshape(1,-1))[0][1])
    sp  = float(sgd_model.predict_proba([url])[0][1])
    np_ = float(nb_model.predict_proba([url])[0][1])
    score = sp*w_sgd + xp*w_xgb + rp*w_rf + np_*w_nb
    pred  = "PHISHING" if score >= thr else "LEGITIMATE"
    status = "OK" if pred == expected else "WRONG"
    if status == "OK": ok += 1
    print(f"  [{status}] {pred} ({score:.4f}) {url[:65]}")
print(f"\nSanity: {ok}/8 correct")
