"""Diagnostic script to check data pipeline status."""

import os

from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("DATA PIPELINE STATUS CHECK")
print("=" * 60)

# Check Qdrant
print("\n1. Checking Qdrant...")
try:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=os.getenv("QDRANT_CLOUD_URL"), api_key=os.getenv("QDRANT_APIKEY"))
    try:
        collections = client.get_collections()
        print("✓ Connected to Qdrant")
        print(f"  Collections found: {len(collections.collections)}")
        for col in collections.collections:
            info = client.get_collection(col.name)
            print(f"    - {col.name}: {info.points_count} vectors")
    except Exception as e:
        print(f"✗ Qdrant collections error: {e}")
        print("  → Run: poetry poe run-feature-engineering-pipeline")
except Exception as e:
    print(f"✗ Qdrant connection failed: {e}")

# Check MongoDB
print("\n2. Checking MongoDB...")
try:
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi

    client = MongoClient(os.getenv("DATABASE_HOST"), server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    print("✓ Connected to MongoDB")

    db = client["twin"]
    collections = db.list_collection_names()
    print(f"  Collections: {len(collections)}")
    for col in collections:
        count = db[col].count_documents({})
        print(f"    - {col}: {count} documents")
except Exception as e:
    print(f"✗ MongoDB error: {e}")
    print("  → Check DATABASE_HOST in .env file")

# Check environment variables
print("\n3. Checking Environment Variables...")
required_vars = [
    "OPENAI_API_KEY",
    "HUGGINGFACE_ACCESS_TOKEN",
    "COMET_API_KEY",
    "DATABASE_HOST",
    "QDRANT_CLOUD_URL",
    "QDRANT_APIKEY",
]

for var in required_vars:
    value = os.getenv(var)
    if value:
        # Mask sensitive data
        if len(value) > 10:
            masked = value[:6] + "..." + value[-4:]
        else:
            masked = "***"
        print(f"  ✓ {var}: {masked}")
    else:
        print(f"  ✗ {var}: NOT SET")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)
print("If Qdrant collections are empty or missing:")
print("  1. Run: poetry poe run-feature-engineering-pipeline")
print("\nIf MongoDB is empty:")
print("  1. Run: poetry poe run-digital-data-etl")
print("  2. Then: poetry poe run-feature-engineering-pipeline")
print("\nAfter data is populated, retry:")
print("  poetry poe call-rag-retrieval-module")
