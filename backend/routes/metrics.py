from fastapi import APIRouter
import sys
import os
sys.path.append(os.path.abspath("."))
from db.crud.save_metrics import get_metrics

router = APIRouter(prefix="/metrics", tags=["Metrics"])

@router.get("/")
async def metrics():
    return await get_metrics()
