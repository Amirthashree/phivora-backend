from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "phishing_detection")

client = None
db = None

def get_db():
    global client, db
    if client is None:
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            db = client[DB_NAME]
            print(f"Connected to MongoDB: {DB_NAME}")
        except ConnectionFailure as e:
            print(f"MongoDB connection failed: {e}")
            raise
    return db

def close_db():
    global client
    if client:
        client.close()
        client = None
        print("MongoDB connection closed")
