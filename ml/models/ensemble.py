import pickle
import os
import sys
import numpy as np
import re
import math
from urllib.parse import urlparse
sys.path.append(os.path.abspath("."))

SUSPICIOUS_TLDS = [".tk",".ml",".ga",".cf",".gq",".xyz",".top",".work",".click",".link",".online",".site",".info",".biz"]
TRUSTED_DOMAINS = ["google","facebook","amazon","microsoft","apple","paypal","netflix","instagram","twitter","linkedin","youtube","github","stackoverflow","wikipedia"]
TRUSTED_EXACT   = ["google.com","facebook.com","amazon.com","microsoft.com","apple.com","netflix.com",
                   "instagram.com","twitter.com","linkedin.com","youtube.com","github.com","wikipedia.org",
                   "stackoverflow.com","gmail.com","outlook.com","yahoo.com","reddit.com"]

def calc_entropy(text):
    if not text: return 0
    freq = {}
    for c in str(text): freq[c] = freq.get(c,0)+1
    return round(-sum((v/len(text))*math.log2(v/len(text)) for v in freq.values()),4)

def extract_features(url, feat_cols):
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

def is_trusted(url):
    try:
        domain = urlparse(url if url.startswith("http") else "http://"+url).netloc.lower().replace("www.","")
        return any(domain==t or domain.endswith("."+t) for t in TRUSTED_EXACT)
    except: return False

class PhishingEnsemble:
    def __init__(self, model_dir="ml/saved_models"):
        self.model_dir    = model_dir
        self.xgboost      = None
        self.random_forest= None
        self.sgd          = None
        self.naive_bayes  = None
        self.feature_cols = None

    def load_models(self):
        with open(f"{self.model_dir}/xgboost.pkl",      "rb") as f: self.xgboost       = pickle.load(f)
        with open(f"{self.model_dir}/random_forest.pkl", "rb") as f: self.random_forest = pickle.load(f)
        with open(f"{self.model_dir}/sgd.pkl",           "rb") as f: self.sgd           = pickle.load(f)
        with open(f"{self.model_dir}/naive_bayes.pkl",   "rb") as f: self.naive_bayes   = pickle.load(f)
        with open(f"{self.model_dir}/feature_cols.pkl",  "rb") as f: self.feature_cols  = pickle.load(f)
        print("All models loaded")

    def predict(self, url="", text=""):
        url  = str(url  or "").strip()
        text = str(text or "").strip()
        has_url  = len(url)  > 3 and url  != "nan"
        has_text = len(text) > 10 and text != "nan"

        if has_text and not has_url:   data_type = "text"
        elif has_url and not has_text: data_type = "url"
        elif has_url and has_text:     data_type = "both"
        else:                          data_type = "unknown"

        scores = {}
        if has_url:
            # XGBoost + RF use feature vectors
            X = np.array(extract_features(url, self.feature_cols), dtype=float).reshape(1,-1)
            scores["xgboost"]       = round(float(self.xgboost.predict_proba(X)[0][1]),       4)
            scores["random_forest"] = round(float(self.random_forest.predict_proba(X)[0][1]), 4)
            # SGD uses raw URL string
            scores["sgd"]           = round(float(self.sgd.predict_proba([url])[0][1]),        4)
            # NB uses text if available else URL
            nb_in = text if has_text else url
            scores["naive_bayes"]   = round(float(self.naive_bayes.predict_proba([nb_in])[0][1]), 4)
        elif has_text:
            scores["naive_bayes"]   = round(float(self.naive_bayes.predict_proba([text])[0][1]), 4)

        xgb = scores.get("xgboost",       0.5)
        rf  = scores.get("random_forest",  0.5)
        sgd = scores.get("sgd",            0.5)
        nb  = scores.get("naive_bayes",    0.5)

        if data_type == "text":
            final = nb
        elif data_type == "url":
            if is_trusted(url):
                final = xgb*0.40 + rf*0.30 + sgd*0.20 + nb*0.10
            else:
                final = sgd*0.40 + xgb*0.25 + rf*0.20 + nb*0.15
        elif data_type == "both":
            final = sgd*0.30 + xgb*0.20 + rf*0.15 + nb*0.35
        else:
            final = 0.5

        threshold = float(os.getenv("PHISHING_THRESHOLD", "0.5"))
        label     = 1 if final >= threshold else 0

        if final >= 0.85:   severity = "HIGH"
        elif final >= 0.65: severity = "MEDIUM"
        elif final >= 0.50: severity = "LOW"
        else:               severity = "SAFE"

        return {
            "label":        label,
            "verdict":      "PHISHING" if label == 1 else "LEGITIMATE",
            "confidence":   round(final, 4),
            "severity":     severity,
            "data_type":    data_type,
            "model_scores": scores
        }
