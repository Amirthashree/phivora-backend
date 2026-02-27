from pydantic import BaseModel, Field
from typing import Optional

class ScanRequest(BaseModel):
    url:  Optional[str] = Field(None, description="URL to scan")
    text: Optional[str] = Field(None, description="Email or SMS text to scan")
