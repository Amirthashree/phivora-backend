import sys
import os
sys.path.append(os.path.abspath("."))
from db.connection import get_db
from datetime import datetime, timezone

async def save_scan(doc: dict):
    database = get_db()
    if database is None:
        return
    database["scan_history"].insert_one(doc.copy())
    if doc.get("label") == 1:
        database["threat_log"].insert_one({
            "scan_id":    doc["scan_id"],
            "verdict":    doc["verdict"],
            "confidence": doc["confidence"],
            "severity":   doc.get("severity", "UNKNOWN"),
            "data_type":  doc["data_type"],
            "input":      doc["input"],
            "timestamp":  doc["timestamp"]
        })

async def get_scan_history(limit: int = 50) -> list:
    return []
