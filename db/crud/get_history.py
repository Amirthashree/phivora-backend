import sys
import os
sys.path.append(os.path.abspath("."))
from db.connection import get_db

async def get_history(limit: int = 50) -> list:
    database = get_db()
    if database is None:
        return []
    docs = list(
        database["scan_history"]
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    for d in docs:
        if hasattr(d.get("timestamp"), "isoformat"):
            d["timestamp"] = d["timestamp"].isoformat()
    return docs

async def get_stats() -> dict:
    database = get_db()
    if database is None:
        return {"total": 0, "phishing": 0, "legitimate": 0}
    total    = database["scan_history"].count_documents({})
    phishing = database["scan_history"].count_documents({"label": 1})
    return {"total": total, "phishing": phishing, "legitimate": total - phishing}
