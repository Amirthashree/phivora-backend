import sys
import os
sys.path.append(os.path.abspath("."))
from db.connection import get_db

async def get_threats(limit: int = 50) -> list:
    database = get_db()
    if database is None:
        return []
    docs = list(
        database["threat_log"]
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    for d in docs:
        if hasattr(d.get("timestamp"), "isoformat"):
            d["timestamp"] = d["timestamp"].isoformat()
    return docs

async def get_severity_stats() -> dict:
    database = get_db()
    if database is None:
        return {}
    return {
        "high":   database["threat_log"].count_documents({"severity": "HIGH"}),
        "medium": database["threat_log"].count_documents({"severity": "MEDIUM"}),
        "low":    database["threat_log"].count_documents({"severity": "LOW"})
    }
