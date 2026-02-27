import sys
import pandas as pd
import numpy as np
sys.path.append(".")

print("Loading combined_train.csv...")
df = pd.read_csv("dataset/combined_train.csv", low_memory=False)
df = df.fillna("")
df["url"] = df["url"].astype(str)
df["label"] = df["label"].astype(int)

df_url = df[(df["url"].str.len() > 5) & (df["url"] != "nan")]

print(f"\nTotal URL rows     : {len(df_url)}")
print(f"Phishing (1)       : {(df_url['label']==1).sum()}")
print(f"Legitimate (0)     : {(df_url['label']==0).sum()}")

print("\n--- Sample PHISHING URLs ---")
phish_samples = df_url[df_url["label"]==1]["url"].sample(20, random_state=42).tolist()
for u in phish_samples:
    print(f"  {u[:100]}")

print("\n--- Sample LEGITIMATE URLs ---")
legit_samples = df_url[df_url["label"]==0]["url"].sample(20, random_state=42).tolist()
for u in legit_samples:
    print(f"  {u[:100]}")

print("\n--- URLs by source ---")
print(df_url.groupby(["source","label"]).size().unstack(fill_value=0))

print("\n--- Checking known phishing URL in data ---")
tk_urls = df_url[df_url["url"].str.contains(".tk", na=False)]
print(f".tk URLs in dataset: {len(tk_urls)}")
print(f"  Phishing: {(tk_urls['label']==1).sum()}")
print(f"  Legit   : {(tk_urls['label']==0).sum()}")

github_urls = df_url[df_url["url"].str.contains("github", na=False)]
print(f"\ngithub URLs in dataset: {len(github_urls)}")
print(f"  Phishing: {(github_urls['label']==1).sum()}")
print(f"  Legit   : {(github_urls['label']==0).sum()}")
