from pymongo import MongoClient

MONGO_URL = "mongodb://localhost:27017"

client = MongoClient(MONGO_URL)

db = client["phishing_detection_db"]

print("✅ MongoDB Connected Successfully")