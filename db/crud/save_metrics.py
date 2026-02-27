import sys
import os
sys.path.append(os.path.abspath("."))
from db.connection import get_db
from datetime import datetime, timezone

async def save_metrics(metrics: dict):
    database = get_db()
    if database is None:
        return
    metrics["timestamp"] = datetime.now(timezone.utc)
    database["model_metrics"].insert_one(metrics.copy())

async def get_metrics() -> dict:
    database = get_db()
    if database is None:
        return {}
    docs = list(
        database["model_metrics"]
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(10)
    )
    return {"metrics": docs}
