from pymongo import MongoClient
import os

MONGO_URL = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = MongoClient(MONGO_URL)
db = client["phishing_detection_db"]

print("✅ MongoDB Connected Successfully")