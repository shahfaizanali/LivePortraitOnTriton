from pymongo import MongoClient
import os

# Database connection setup
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db["users"]

async def get_user_by_id(user_id):
    """
    Fetch a user by their ID from the database.
    """
    return await users_collection.find_one({"_id": user_id})
