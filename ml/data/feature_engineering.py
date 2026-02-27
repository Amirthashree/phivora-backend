import pandas as pd
import numpy as np
import sys
import os
import re
import math
from urllib.parse import urlparse
sys.path.append(os.path.abspath("."))

SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work',
                   '.click', '.link', '.online', '.site', '.info', '.biz']
TRUSTED_DOMAINS = ['google', 'facebook', 'amazon', 'microsoft', 'apple',
                   'paypal', 'netflix', 'instagram', 'twitter', 'linkedin',
                   'youtube', 'github', 'stackoverflow', 'wikipedia']

def calculate_entropy(text):
    if not text:
        return 0
    freq = {}
    for c in str(text):
        freq[c] = freq.get(c, 0) + 1
    entropy = 0
    for count in freq.values():
        p = count / len(text)
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)

def extract_all_features(url, text):
    url  = str(url  or "")
    text = str(text or "")
    f    = {}

    # ── URL features ──────────────────────────────────────────
    try:
        parsed    = urlparse(url if url.startswith("http") else "http://" + url)
        domain    = parsed.netloc.lower()
        path      = parsed.path
        query     = parsed.query
        full      = url.lower()
    except:
        domain = path = query = full = ""

    f["url_length"]              = len(url)
    f["domain_length"]           = len(domain)
    f["path_length"]             = len(path)
    f["num_dots"]                = url.count(".")
    f["num_hyphens"]             = url.count("-")
    f["num_underscores"]         = url.count("_")
    f["num_slashes"]             = url.count("/")
    f["num_at"]                  = url.count("@")
    f["num_question"]            = url.count("?")
    f["num_ampersand"]           = url.count("&")
    f["num_equal"]               = url.count("=")
    f["num_percent"]             = url.count("%")
    f["num_digits"]              = sum(c.isdigit() for c in url)
    f["digit_ratio"]             = round(sum(c.isdigit() for c in url) / max(len(url), 1), 4)
    f["letter_ratio"]            = round(sum(c.isalpha() for c in url) / max(len(url), 1), 4)
    f["url_entropy"]             = calculate_entropy(url)
    f["domain_entropy"]          = calculate_entropy(domain)
    f["has_ip"]                  = 1 if re.match(r"https?://\d+\.\d+\.\d+\.\d+", url) else 0
    f["is_https"]                = 1 if url.startswith("https") else 0
    f["has_port"]                = 1 if re.search(r":\d{2,5}", domain) else 0
    f["num_subdomains"]          = max(len(domain.split(".")) - 2, 0)
    f["is_shortener"]            = 1 if any(s in url for s in ["bit.ly","tinyurl","goo.gl","t.co","ow.ly"]) else 0
    f["has_suspicious_keyword"]  = 1 if any(k in full for k in ["login","verify","secure","update","account","banking","confirm","password","suspend","unlock"]) else 0
    f["suspicious_keyword_count"]= sum(1 for k in ["login","verify","secure","update","account","banking","confirm","password","suspend","unlock"] if k in full)
    f["has_double_slash"]        = 1 if "//" in path else 0
    f["has_prefix_suffix"]       = 1 if "-" in domain else 0
    f["query_length"]            = len(query)
    f["num_query_params"]        = len(query.split("&")) if query else 0

    # Homoglyph
    homoglyph_map = {"0":"o","1":"l","3":"e","4":"a","5":"s","@":"a","$":"s","vv":"w","rn":"m"}
    normalized = url.lower()
    for fake, real in homoglyph_map.items():
        normalized = normalized.replace(fake, real)
    f["has_homoglyph"] = 1 if any(brand in normalized and brand not in url.lower() for brand in TRUSTED_DOMAINS) else 0
    f["brand_impersonation_count"] = sum(1 for brand in TRUSTED_DOMAINS if brand in url.lower())

    # Extra URL signals
    f["has_suspicious_tld"]      = 1 if any(url.lower().endswith(t) or t+"/" in url.lower() for t in SUSPICIOUS_TLDS) else 0
    f["domain_has_numbers"]      = 1 if re.search(r"\d", domain) else 0
    f["path_has_exe"]            = 1 if any(url.lower().endswith(e) for e in [".exe",".zip",".rar",".php",".js"]) else 0
    f["has_redirect_sign"]       = 1 if "redirect" in full or "forward" in full or "url=" in full else 0
    f["dots_in_domain"]          = domain.count(".")
    f["domain_token_count"]      = len(re.findall(r"[a-zA-Z0-9]+", domain))
    f["path_token_count"]        = len(re.findall(r"[a-zA-Z0-9]+", path))
    f["is_trusted_domain"]       = 1 if any(t in domain for t in TRUSTED_DOMAINS) else 0
    f["url_has_encoded_chars"]   = 1 if "%" in url else 0
    f["url_depth"]               = path.count("/")
    f["domain_word_count"]       = len(re.findall(r"[a-z]{3,}", domain))
    f["ratio_special_chars"]     = round(sum(1 for c in url if not c.isalnum()) / max(len(url), 1), 4)
    f["is_url_row"]              = 1 if len(url) > 5 else 0

    # ── Text features ─────────────────────────────────────────
    URGENCY = ["urgent","immediately","suspended","verify now","click here","act now",
               "limited time","expires","warning","congratulations","winner","prize",
               "free","risk","threat","unusual activity","unauthorized","confirm your",
               "update your","dear customer","dear user","account locked","action required"]
    PHISHING_RE = [
        r"dear\s+(customer|user|member)",
        r"click\s+(here|below|this link)",
        r"verify\s+your\s+(account|identity|email)",
        r"account\s+(has been|will be|is)\s+(suspended|blocked|locked)",
        r"(win|won|winner).{0,20}(prize|reward|gift|cash)",
        r"(update|confirm|validate).{0,20}(password|credential)",
        r"(bank|paypal|amazon|apple).{0,30}(verify|confirm|update)",
    ]

    tl = text.lower()
    words = text.split()

    f["text_length"]             = len(text)
    f["word_count"]              = len(words)
    f["sentence_count"]          = len(re.split(r"[.!?]", text))
    f["avg_word_length"]         = round(sum(len(w) for w in words) / max(len(words), 1), 4)
    f["urgency_word_count"]      = sum(1 for w in URGENCY if w in tl)
    f["phishing_pattern_count"]  = sum(1 for p in PHISHING_RE if re.search(p, tl))
    f["url_count_in_text"]       = len(re.findall(r"http[s]?://\S+", text))
    f["html_tag_count"]          = len(re.findall(r"<[^>]+>", text))
    f["exclamation_count"]       = text.count("!")
    f["question_count"]          = text.count("?")
    f["capital_ratio"]           = round(sum(1 for c in text if c.isupper()) / max(len(text), 1), 4)
    f["text_digit_ratio"]        = round(sum(1 for c in text if c.isdigit()) / max(len(text), 1), 4)
    f["text_entropy"]            = calculate_entropy(text[:500])
    f["has_html"]                = 1 if f["html_tag_count"] > 0 else 0
    f["has_url_in_text"]         = 1 if f["url_count_in_text"] > 0 else 0
    f["is_text_row"]             = 1 if len(text) > 10 else 0

    # ── Combined signals ──────────────────────────────────────
    f["suspicious_and_short"]    = 1 if f["has_suspicious_keyword"] and len(url) < 50 else 0
    f["ip_and_http"]             = 1 if f["has_ip"] and not f["is_https"] else 0
    f["many_dots_short_domain"]  = 1 if f["num_dots"] > 3 and f["domain_length"] < 20 else 0
    f["high_urgency_score"]      = 1 if f["urgency_word_count"] >= 3 else 0
    f["suspicious_tld_keyword"]  = 1 if f["has_suspicious_tld"] and f["has_suspicious_keyword"] else 0
    f["impersonation_no_https"]  = 1 if f["brand_impersonation_count"] > 0 and not f["is_https"] else 0
    f["long_subdomain"]          = 1 if f["num_subdomains"] > 3 else 0
    f["encoded_and_suspicious"]  = 1 if f["url_has_encoded_chars"] and f["has_suspicious_keyword"] else 0

    return f

def build_features(df):
    print("Extracting features...")
    all_features = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100000 == 0:
            print(f"  Progress: {i}/{total}")
        feats = extract_all_features(
            row.get("url", ""),
            row.get("text", "")
        )
        feats["label"] = row["label"]
        all_features.append(feats)
    feature_df = pd.DataFrame(all_features)
    feature_df = feature_df.fillna(0)
    print(f"Feature matrix shape: {feature_df.shape}")
    return feature_df

if __name__ == "__main__":
    print("=== Building Training Features ===")
    train = pd.read_csv("dataset/combined_train.csv", low_memory=False)
    train_features = build_features(train)
    train_features.to_csv("dataset/train_features.csv", index=False)
    print("Saved to dataset/train_features.csv")

    print("\n=== Building Testing Features ===")
    test = pd.read_csv("dataset/combined_test.csv", low_memory=False)
    test_features = build_features(test)
    test_features.to_csv("dataset/test_features.csv", index=False)
    print("Saved to dataset/test_features.csv")
