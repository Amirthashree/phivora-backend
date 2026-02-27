import sys
import os
sys.path.append(os.path.abspath("."))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.config import settings
from backend.routes.scan    import router as scan_router
from backend.routes.history import router as history_router
from backend.routes.metrics import router as metrics_router
from backend.services.model_loader import model_loader
from db.database import db
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    model_loader.load()
    print("Models ready")
    yield
    print("Shutting down")

app = FastAPI(
    title="Phishing Detection API",
    description="Detects phishing in URLs, emails and SMS using ML ensemble",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(scan_router)
app.include_router(history_router)
app.include_router(metrics_router)

@app.get("/")
async def root():
    return {"status": "running", "models": model_loader.is_loaded(), "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": model_loader.is_loaded()}

@app.get("/test-db")
def test_db():
    db.test.insert_one({"message": "MongoDB Connected"})
    return {"status": "Inserted Successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
