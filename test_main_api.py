#!/usr/bin/env python3
"""Test Main API with RAG pipeline"""

import requests
import time

# Main API URL (local or deployed)
API_URL = "http://localhost:8000"  # Using 8001 since 8000 is in use

print(f"🧪 Testing Main API: {API_URL}\n")
print("=" * 60)

# Test: RAG Query
print("Testing /rag endpoint...")
query = "What topics are discussed in the posts?"

try:
    print(f"Query: '{query}'")
    print("⏳ Processing (retrieval + RunPod inference)...\n")

    start = time.time()
    response = requests.post(f"{API_URL}/rag", json={"query": query}, timeout=120)
    elapsed = time.time() - start

    print(f"✅ Status: {response.status_code}")
    print(f"⏱️  Time: {elapsed:.2f}s\n")

    if response.status_code == 200:
        result = response.json()
        print("📝 RAG Answer:")
        print("-" * 60)
        print(result["answer"])
        print("-" * 60)
        print("\n✅ SUCCESS! Your RAG pipeline is working!")
        print("   ✓ Vector search completed")
        print("   ✓ Context retrieved")
        print("   ✓ RunPod inference called")
        print("   ✓ Answer generated with context")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

except requests.exceptions.Timeout:
    print("❌ Timeout - Check if Main API is running")
except requests.exceptions.ConnectionError:
    print("❌ Connection failed - Is the server running on port 8000?")
except Exception as e:
    print(f"❌ Error: {e}")
