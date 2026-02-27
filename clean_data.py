import sys
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
sys.path.append(".")

print("Loading data...")
df = pd.read_csv("dataset/combined_train.csv", low_memory=False)
df = df.fillna("")
df["url"]   = df["url"].astype(str)
df["label"] = df["label"].astype(int)
df = df.reset_index(drop=True)
original_len = len(df)
print(f"Original rows: {original_len:,}")

PHISHING_TLDS = [".tk", ".ml", ".ga", ".cf", ".gq"]
TRUSTED_EXACT = [
    "google.com", "youtube.com", "facebook.com", "twitter.com",
    "instagram.com", "linkedin.com", "github.com", "wikipedia.org",
    "stackoverflow.com", "microsoft.com", "apple.com", "amazon.com",
    "reddit.com", "netflix.com", "yahoo.com", "bing.com",
    "cnn.com", "bbc.com", "nytimes.com", "theguardian.com"
]

def get_domain(url):
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        return parsed.netloc.lower().replace("www.", "")
    except:
        return ""

def fix_and_filter(df):
    urls   = df["url"].str.lower()
    labels = df["label"].copy()
    domains = df["url"].apply(get_domain)

    # Force PHISHING
    has_phishing_tld = urls.apply(lambda u: any(u.endswith(t) or (t+"/") in u for t in PHISHING_TLDS))
    has_raw_ip       = urls.str.match(r"https?://\d+\.\d+\.\d+\.\d+")
    labels[has_phishing_tld] = 1
    labels[has_raw_ip]       = 1

    # Force LEGITIMATE
    is_trusted = domains.apply(lambda d: any(d == t or d.endswith("."+t) for t in TRUSTED_EXACT))
    labels[is_trusted] = 0

    df = df.copy()
    df["label"] = labels

    # Remove contradictory rows
    still_bad = (
        ((df["label"] == 0) & has_phishing_tld) |
        ((df["label"] == 0) & has_raw_ip) |
        ((df["label"] == 1) & is_trusted)
    )
    df = df[~still_bad].reset_index(drop=True)

    # Remove train15 (network attack data, not phishing)
    df = df[df["source"] != "train15"].reset_index(drop=True)

    return df

print("Fixing labels and removing contradictory rows...")
df_clean = fix_and_filter(df)

print(f"Removed  : {original_len - len(df_clean):,} rows")
print(f"Remaining: {len(df_clean):,} rows")

url_rows  = df_clean[df_clean["url"].str.len() > 5]
text_rows = df_clean[df_clean["text"].str.len() > 10]
print(f"\nURL rows : {len(url_rows):,}  (phishing={(url_rows['label']==1).sum():,} | legit={(url_rows['label']==0).sum():,})")
print(f"Text rows: {len(text_rows):,} (phishing={(text_rows['label']==1).sum():,} | legit={(text_rows['label']==0).sum():,})")

print("\n--- Verify fixes ---")
tk = url_rows[url_rows["url"].str.contains(r"\.tk", na=False, regex=True)]
print(f".tk    -> phishing={( tk['label']==1).sum():,} | legit={(tk['label']==0).sum():,}")
gh = url_rows[url_rows["url"].str.contains("github", na=False)]
print(f"github -> phishing={(gh['label']==1).sum():,} | legit={(gh['label']==0).sum():,}")
ip = url_rows[url_rows["url"].str.match(r"https?://\d+\.\d+\.\d+\.\d+")]
print(f"raw IP -> phishing={(ip['label']==1).sum():,} | legit={(ip['label']==0).sum():,}")

df_clean.to_csv("dataset/combined_train_clean.csv", index=False)
print("\nSaved to dataset/combined_train_clean.csv")
