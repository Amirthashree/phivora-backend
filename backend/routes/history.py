from fastapi import APIRouter, Query
import sys
import os
sys.path.append(os.path.abspath("."))
from db.crud.get_history import get_history, get_stats

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/")
async def history(limit: int = Query(50, ge=1, le=500)):
    data  = await get_history(limit)
    stats = await get_stats()
    return {
        "total":      stats.get("total", 0),
        "phishing":   stats.get("phishing", 0),
        "legitimate": stats.get("legitimate", 0),
        "scans":      data
    }
