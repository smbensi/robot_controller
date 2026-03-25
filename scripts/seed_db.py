#!/usr/bin/env python3
"""
Seed MongoDB with sample users and locations for testing.
Run: python3 scripts/seed_db.py
"""

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["robot"]

# Seed users
users = [
    {"name": "John", "role": "patient", "room": "101"},
    {"name": "Jane", "role": "patient", "room": "102"},
    {"name": "Dr. Smith", "role": "doctor", "department": "cardiology"},
    {"name": "Nurse Alice", "role": "nurse", "shift": "morning"},
]

db.users.drop()
result = db.users.insert_many(users)
print(f"Inserted {len(result.inserted_ids)} users")

# Seed locations
locations = [
    {"name": "Room 101", "floor": 1, "type": "patient_room"},
    {"name": "Room 102", "floor": 1, "type": "patient_room"},
    {"name": "Nurse Station", "floor": 1, "type": "station"},
    {"name": "Pharmacy", "floor": 0, "type": "service"},
    {"name": "Lobby", "floor": 0, "type": "common"},
]

db.locations.drop()
result = db.locations.insert_many(locations)
print(f"Inserted {len(result.inserted_ids)} locations")

# Create indexes
db.users.create_index("name")
db.locations.create_index("name")
print("Indexes created")

client.close()
print("Done.")
