# 🚀 RunPod Deployment Session Note
**Date:** 2025-11-18
**Session Goal:** Deploy RAG-LLM inference server to RunPod GPU platform
**Current Status:** ✅ Docker image built successfully, ready for GPU testing

---

## 📝 What We've Accomplished So Far

### ✅ Step 1: Created RunPod Dockerfile (COMPLETED)

**Files Created (NO deletions made):**

1. **`Dockerfile.runpod`** (3.5KB)
   - Location: `/teamspace/studios/this_studio/rag-llm-system/Dockerfile.runpod`
   - Purpose: RunPod-specific Dockerfile for GPU inference
   - Base Image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
   - Includes: transformers, fastapi, uvicorn, accelerate, loguru
   - Exposes: Port 8000 (FastAPI inference API)
   - Entry Point: `python -m rag_llm_system.infrastructure.local.local_deploy`

2. **`rag_llm_system/infrastructure/local/__init__.py`** (75 bytes)
   - Required Python module file for proper imports

**Original Files:** All untouched ✅

### ✅ Docker Image Build (COMPLETED)
```bash
# Build command that was run:
docker build -f Dockerfile.runpod -t rag-inference-runpod:latest .

# Status: SUCCESS
# Image created: rag-inference-runpod:latest
```

---

## 🎯 NEXT STEPS - When You Return with GPU

### Step 2: Test Docker Image on GPU

**Important Prerequisites:**
- ✅ Switch to GPU environment
- ✅ Ensure NVIDIA Docker runtime is installed
- ✅ Have your Hugging Face token ready

**Commands to Run:**

```bash
# 1. Verify GPU is available
nvidia-smi

# 2. Verify Docker image exists
docker images | grep rag-inference-runpod

# 3. Set your Hugging Face token (REQUIRED)
export HUGGINGFACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"

# 4. Test run the container with GPU
docker run --gpus all -p 8000:8000 \
  -e HF_MODEL_ID="mlabonne/TwinLlama-3.1-8B-DPO" \
  -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}" \
  rag-inference-runpod:latest

# Expected output:
# - Model downloading from HuggingFace Hub (~16GB)
# - Loading onto GPU
# - FastAPI server starts on port 8000
# - "🔥 Running Local LLM API on port 8000"
```

### Step 3: Test the Inference API

**Open a NEW terminal while container is running:**

```bash
# Test 1: Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"running","device":"cuda","model":"mlabonne/TwinLlama-3.1-8B-DPO"}

# Test 2: Inference request
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Retrieval-Augmented Generation (RAG)? Explain in one paragraph."
  }'

# Expected response:
# {
#   "generated_text": "Retrieval-Augmented Generation (RAG) is...",
#   "compliance_status": "compliant"
# }
```

---

## 📊 What to Check During Testing

### ✅ Success Indicators:
- [ ] GPU detected (device="cuda" in health check)
- [ ] Model loads successfully (~16GB VRAM used)
- [ ] FastAPI server starts on port 8000
- [ ] Health endpoint returns status 200
- [ ] Inference generates coherent text
- [ ] Response time < 5 seconds for short prompts

### ⚠️ Potential Issues & Solutions:

**Issue 1: CUDA Out of Memory**
```bash
# Solution: Use 8-bit quantization (reduce VRAM from 16GB to 8GB)
# We'll create this in next session if needed
```

**Issue 2: Model Download Fails**
```bash
# Solution: Verify Hugging Face token
huggingface-cli login
# Or set token: export HUGGINGFACE_HUB_TOKEN="hf_..."
```

**Issue 3: Port 8000 Already in Use**
```bash
# Solution: Use different port
docker run --gpus all -p 8001:8000 ...
# Then test on: http://localhost:8001
```

---

## 🗂️ Project Structure Snapshot

```
rag-llm-system/
├── Dockerfile                    # Original (untouched)
├── Dockerfile.runpod            # ✨ NEW - RunPod GPU image
├── rag_llm_system/
│   ├── infrastructure/
│   │   └── local/
│   │       ├── __init__.py      # ✨ NEW - Module file
│   │       ├── local_deploy.py  # Inference server (existing)
│   │       └── governance.py    # Compliance monitor (existing)
│   └── settings.py              # Config (existing)
└── SESSION_NOTE_RUNPOD_DEPLOYMENT.md  # ✨ THIS FILE
```

---

## 🔐 Environment Variables Needed

**For Docker Testing:**
```bash
# Required
HUGGINGFACE_HUB_TOKEN=hf_your_token_here
HF_MODEL_ID=mlabonne/TwinLlama-3.1-8B-DPO  # (default in code)

# Optional (for full integration later)
OPENAI_API_KEY=sk-...
QDRANT_CLOUD_URL=https://...
QDRANT_APIKEY=...
DATABASE_HOST=mongodb+srv://...
```

---

## 📋 Timeline of Next Sessions

### **Session 2 (Current - GPU Testing):**
- ✅ Test Docker image on GPU
- ✅ Verify inference works
- ✅ Measure performance (latency, VRAM usage)

### **Session 3 (Push to Docker Hub):**
- Push image to Docker Hub or GitHub Container Registry
- Tag and version the image
- Document deployment instructions

### **Session 4 (Deploy to RunPod):**
- Create RunPod account (if not already)
- Deploy as RunPod Serverless endpoint OR Pod
- Configure auto-scaling
- Get inference endpoint URL

### **Session 5 (Integrate with Main API):**
- Create RunPod inference client
- Update settings.py with RunPod config
- Test end-to-end RAG pipeline
- Deploy main API to Render.com

---

## 🧪 Expected GPU Testing Results

**Model:** TwinLlama-3.1-8B-DPO (8 billion parameters)

| Metric | Expected Value |
|--------|----------------|
| **Model Size** | ~16GB (FP16) |
| **VRAM Required** | 18-20GB (with overhead) |
| **Load Time** | 30-60 seconds (first run) |
| **Inference Time** | 0.5-2 seconds (per 100 tokens) |
| **Recommended GPU** | RTX 4090 (24GB), A4000 (16GB), or better |

---

## 📞 Quick Commands Reference

```bash
# Check this file anytime
cat SESSION_NOTE_RUNPOD_DEPLOYMENT.md

# View Dockerfile
cat Dockerfile.runpod

# Check Docker images
docker images | grep rag-inference

# Stop running container
docker ps  # Get container ID
docker stop <container_id>

# View container logs
docker logs <container_id>

# Remove container after testing
docker rm <container_id>
```

---

## 🚨 Important Notes

1. **No Files Deleted:** All original project files remain intact
2. **Model Download:** First run will download ~16GB model (takes time)
3. **GPU Required:** This Docker image requires GPU (`--gpus all` flag)
4. **Port 8000:** Ensure nothing else is using this port
5. **Token Required:** Hugging Face token needed for model download

---

## 🎯 Resume Point

**When you return with GPU:**

```bash
# You are here: 👇
cd /teamspace/studios/this_studio/rag-llm-system

# Verify image exists
docker images | grep rag-inference-runpod

# Set your HF token
export HUGGINGFACE_HUB_TOKEN="hf_YOUR_TOKEN"

# Run the container
docker run --gpus all -p 8000:8000 \
  -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}" \
  rag-inference-runpod:latest

# Then test in another terminal:
curl http://localhost:8000/health
```

---

## ✅ Completion Checklist

- [x] Dockerfile.runpod created
- [x] __init__.py files created
- [x] Docker image built
- [ ] **→ GPU testing** (NEXT STEP)
- [ ] Inference verified
- [ ] Performance benchmarked
- [ ] Pushed to Docker registry
- [ ] Deployed to RunPod
- [ ] Integrated with main API

---

**Session saved:** 2025-11-18 20:16 UTC
**Progress:** 30% complete (3/10 steps)
**Status:** Ready for GPU testing 🚀

---

## 💡 Pro Tips

1. **Model Caching:** After first download, model is cached in container
2. **Logs:** Use `docker logs -f <container_id>` to watch live logs
3. **Memory:** Monitor with `nvidia-smi` during inference
4. **Cleanup:** Use `docker system prune` to free space if needed

---

**Good luck with GPU testing! 🎉**
**Come back to this file when ready to continue.**
