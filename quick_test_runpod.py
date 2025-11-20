#!/usr/bin/env python3
"""Quick RunPod endpoint test with detailed output"""

import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("RUNPOD_ENDPOINT_URL", "https://muzug3zj4lubtw.api.runpod.ai")
API_KEY = os.getenv("RUNPOD_API_KEY")

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

print(f"Testing: {ENDPOINT}")
print(f"API Key: {API_KEY[:20]}...")
print("-" * 60)

try:
    print("\n🔍 Attempting health check...")
    print("⏳ Waiting up to 120 seconds...")

    start = time.time()
    response = requests.get(f"{ENDPOINT}/health", headers=headers, timeout=120)
    elapsed = time.time() - start

    print(f"\n✅ Got response in {elapsed:.1f}s")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

except requests.exceptions.Timeout:
    print(f"\n❌ Timeout after 120 seconds")
    print("\n💡 Possible issues:")
    print("   1. Worker not starting (check RunPod logs)")
    print("   2. Container port not set to 8000")
    print("   3. Health check path not configured")

except requests.exceptions.RequestException as e:
    print(f"\n❌ Request failed: {e}")

except Exception as e:
    print(f"\n❌ Error: {e}")
