from backend.db import db
from bson import ObjectId

async def save_scan(doc):
    db.scans.insert_one(doc)

async def get_history(limit=50):
    results = list(db.scans.find().sort("timestamp", -1).limit(limit))
    for r in results:
        r["_id"] = str(r["_id"])
    return results

async def get_stats():
    total = db.scans.count_documents({})
    phishing = db.scans.count_documents({"verdict": "phishing"})
    legitimate = db.scans.count_documents({"verdict": "legitimate"})
    return {"total": total, "phishing": phishing, "legitimate": legitimate}
