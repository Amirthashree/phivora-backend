from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
import uuid
import sys
import os
sys.path.append(os.path.abspath("."))
from backend.schemas.scan_schema import ScanRequest
from backend.schemas.response_schema import ScanResponse
from backend.services.prediction_service import predict
from backend.services.model_loader import model_loader
from db.crud.save_scan import save_scan

router = APIRouter(prefix="/scan", tags=["Scan"])

@router.post("/", response_model=ScanResponse)
async def scan(req: ScanRequest):
    if not req.url and not req.text:
        raise HTTPException(status_code=400, detail="Provide url or text")
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    result  = predict(url=req.url or "", text=req.text or "")
    scan_id = str(uuid.uuid4())
    ts      = datetime.now(timezone.utc)

    doc = {
        "scan_id":      scan_id,
        "label":        result["label"],
        "verdict":      result["verdict"],
        "confidence":   result["confidence"],
        "severity":     result["severity"],
        "data_type":    result["data_type"],
        "model_scores": result["model_scores"],
        "timestamp":    ts,
        "input":        {"url": req.url, "text": req.text}
    }

    try:
        await save_scan(doc)
    except Exception as e:
        print(f"DB save error: {e}")

    return ScanResponse(**doc)
