from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class ScanResponse(BaseModel):
    scan_id:      str
    label:        int
    verdict:      str
    confidence:   float
    data_type:    str
    model_scores: Dict[str, float]
    timestamp:    datetime
    input:        Dict[str, Any]
