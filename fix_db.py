import sys
sys.path.append(".")
from db.connection import get_db

db = get_db()

# Clear old scans where models were broken (sgd=0.5 or sgd=1.0 with xgb=0.5)
result = db["scan_history"].delete_many({
    "$or": [
        {"model_scores.sgd": 1.0},
        {"$and": [
            {"model_scores.sgd": 0.5},
            {"model_scores.xgboost": 0.5},
            {"data_type": "url"}
        ]}
    ]
})
print(f"Deleted {result.deleted_count} bad scans")

# Check what remains
remaining = db["scan_history"].count_documents({})
phishing  = db["scan_history"].count_documents({"label": 1})
legit     = db["scan_history"].count_documents({"label": 0})
print(f"Remaining: {remaining} scans ({phishing} phishing, {legit} legit)")

# Save current model metrics
from datetime import datetime, timezone
db["model_metrics"].insert_one({
    "timestamp":      datetime.now(timezone.utc),
    "training_date":  "2026-02-26",
    "ensemble": {
        "accuracy":  0.8800,
        "f1":        0.8276,
        "precision": 0.8496,
        "recall":    0.8067,
        "roc_auc":   0.7593,
        "threshold": 0.55
    },
    "weights": {
        "sgd":          0.50,
        "xgboost":      0.25,
        "random_forest":0.20,
        "naive_bayes":  0.05
    },
    "individual": {
        "naive_bayes_acc": 0.9697,
        "naive_bayes_roc": 0.9964,
        "sgd_sanity":      "8/8",
    },
    "notes": "SGD=TF-IDF char ngrams, XGB+RF=feature vectors, NB=text/SMS"
})
print("Saved model metrics")

# Verify metrics
metrics = list(db["model_metrics"].find({}, {"_id": 0}))
print(f"Metrics documents: {len(metrics)}")
for m in metrics:
    print(f"  {m.get('training_date')} - acc={m.get('ensemble',{}).get('accuracy')} f1={m.get('ensemble',{}).get('f1')}")
