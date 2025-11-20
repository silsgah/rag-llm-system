#!/usr/bin/env python3
"""Test RunPod Pod endpoint"""

import requests
import time

# Your RunPod Pod URL (no auth needed for Pods!)
ENDPOINT = "https://ik8o9j4ddydsv4-8000.proxy.runpod.net"

print(f"🚀 Testing RunPod Pod: {ENDPOINT}\n")
print("=" * 60)

# Test 1: Health Check
print("1️⃣ Health Check...")
try:
    response = requests.get(f"{ENDPOINT}/health", timeout=10)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
except Exception as e:
    print(f"   ❌ Failed: {e}\n")
    exit(1)

# Test 2: Inference
print("2️⃣ Testing Inference...")
prompt = "What is Retrieval-Augmented Generation?"

try:
    start = time.time()
    response = requests.post(f"{ENDPOINT}/infer", json={"prompt": prompt}, timeout=60)
    elapsed = time.time() - start

    print(f"   Status: {response.status_code}")
    print(f"   Time: {elapsed:.2f}s")

    result = response.json()
    print(f"\n📝 Generated Text:\n{'-'*60}")
    print(result["generated_text"][:300] + "...")
    print(f"{'-'*60}")
    print(f"Compliance: {result['compliance_status']}")

    print("\n✅ SUCCESS! Your RunPod Pod is working perfectly! 🎉")

except Exception as e:
    print(f"   ❌ Failed: {e}")
