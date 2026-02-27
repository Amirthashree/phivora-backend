import sys
import pandas as pd
sys.path.append(".")

df = pd.read_csv("dataset/combined_train.csv", low_memory=False)
df = df.fillna("")

print("=== SMS/Text Data Coverage ===")
sms_sources = ["train4", "train7", "train14"]
for src in sms_sources:
    sub = df[df["source"] == src]
    print(f"\n{src}: {len(sub):,} rows")
    print(f"  Phishing: {(sub['label']==1).sum():,}")
    print(f"  Legit   : {(sub['label']==0).sum():,}")
    print(f"  Samples:")
    for t in sub["text"].dropna().sample(min(3,len(sub)), random_state=42):
        print(f"    >> {str(t)[:120]}")

print("\n=== NB Sanity Check on SMS ===")
import pickle
with open("ml/saved_models/naive_bayes.pkl","rb") as f: nb = pickle.load(f)
sms_tests = [
    ("FREE entry in 2 a weekly competition to win FA Cup final tkts! Text FA to 87121", "PHISHING"),
    ("URGENT: Your bank account suspended. Verify now at http://secure-bank.tk",        "PHISHING"),
    ("Hey, are we still meeting at 3pm today?",                                          "LEGITIMATE"),
    ("Your OTP is 482910. Valid for 5 minutes. Do not share.",                           "LEGITIMATE"),
    ("Congratulations! You won $1000. Call 555-0100 to claim your prize now!",          "PHISHING"),
    ("Hi team, the meeting notes are attached. Please review before Friday.",            "LEGITIMATE"),
]
for text, expected in sms_tests:
    score  = nb.predict_proba([text])[0][1]
    pred   = "PHISHING" if score >= 0.5 else "LEGITIMATE"
    status = "OK" if pred == expected else "WRONG"
    print(f"  [{status}] {pred} ({score:.4f}) {text[:80]}")
