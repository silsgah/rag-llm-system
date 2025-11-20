#!/usr/bin/env python3
"""
RunPod Inference Endpoint Test Script
Test your deployed LLM inference server on RunPod
"""

import os
import sys
import time
import requests
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Your RunPod endpoint URL
ENDPOINT = os.getenv("RUNPOD_ENDPOINT_URL", "https://muzug3zj4lubtw.api.runpod.ai")

# RunPod API Key (loaded from .env file)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

if not RUNPOD_API_KEY:
    print("❌ Error: RUNPOD_API_KEY not found in environment variables")
    print("💡 Create a .env file with: RUNPOD_API_KEY=your_key_here")
    sys.exit(1)

# Headers with authentication
HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}


def test_health_check():
    """Test the health endpoint."""
    print("=" * 60)
    print("1️⃣  HEALTH CHECK")
    print("=" * 60)

    try:
        print(f"Calling: {ENDPOINT}/health")
        print("⏳ Waiting for worker to spin up (may take 30-60s on cold start)...")
        response = requests.get(f"{ENDPOINT}/health", headers=HEADERS, timeout=120)

        print(f"✅ Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response:")
            print(f"   - Status: {data.get('status')}")
            print(f"   - Device: {data.get('device')}")
            print(f"   - Model: {data.get('model')}")
            return True
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timed out (10s)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_inference(prompt: str):
    """Test the inference endpoint."""
    print("\n" + "=" * 60)
    print("2️⃣  INFERENCE TEST")
    print("=" * 60)

    try:
        print(f"Prompt: '{prompt}'")
        print(f"Calling: {ENDPOINT}/infer")
        print("⏳ Waiting for response (may take 30-60s on cold start)...")

        start_time = time.time()

        response = requests.post(
            f"{ENDPOINT}/infer",
            headers=HEADERS,
            json={"prompt": prompt},
            timeout=120,  # 2 minutes timeout for cold start
        )

        elapsed_time = time.time() - start_time

        print(f"✅ Status Code: {response.status_code}")
        print(f"⏱️  Response Time: {elapsed_time:.2f} seconds")

        if response.status_code == 200:
            data = response.json()
            print(f"\n📝 Generated Text:")
            print("-" * 60)
            print(data.get("generated_text", "No text generated"))
            print("-" * 60)
            print(f"\n🔒 Compliance Status: {data.get('compliance_status')}")
            return True
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timed out (120s)")
        print("💡 Tip: First request may take longer due to cold start")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("RUNPOD ENDPOINT TEST SUITE")
    print("🚀 " * 20)
    print(f"\nEndpoint: {ENDPOINT}\n")

    # Test 1: Health Check
    health_passed = test_health_check()

    if not health_passed:
        print("\n❌ Health check failed. Cannot proceed with inference test.")
        print("💡 Tips:")
        print("   - Check if RunPod endpoint is running")
        print("   - Verify the endpoint URL is correct")
        print("   - Check RunPod logs for errors")
        sys.exit(1)

    # Test 2: Inference
    test_prompts = [
        "What is Retrieval-Augmented Generation (RAG)? Explain in one paragraph.",
    ]

    for prompt in test_prompts:
        inference_passed = test_inference(prompt)

        if not inference_passed:
            print("\n❌ Inference test failed.")
            print("💡 Tips:")
            print("   - Check RunPod logs for errors")
            print("   - Verify GPU is available")
            print("   - Ensure HUGGINGFACE_HUB_TOKEN is set")
            sys.exit(1)

    # All tests passed
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🎉 Your RunPod inference endpoint is working correctly!")
    print(f"\n📌 Endpoint URL: {ENDPOINT}")
    print("\n💡 You can now integrate this endpoint into your application.")


if __name__ == "__main__":
    main()
