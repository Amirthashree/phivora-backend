import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    MONGO_URI: str   = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME: str     = os.getenv("DB_NAME", "phishing_detection")
    MODEL_DIR: str   = os.getenv("MODEL_DIR", "ml/saved_models")
    THRESHOLD: float = float(os.getenv("PHISHING_THRESHOLD", "0.5"))
    HOST: str        = os.getenv("HOST", "0.0.0.0")
    PORT: int        = int(os.getenv("PORT", "8000"))
    DEBUG: bool      = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings()
