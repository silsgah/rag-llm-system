# Quick Cloud Deployment Guide

Deploy your RAG system to the cloud in 15 minutes for ~$7/month.

## Option 1: Render.com (Recommended - $7/month)

### Step 1: Prepare Environment
```bash
# Make sure you have:
# - MongoDB Atlas account (free)
# - Qdrant Cloud account (free)
# - OpenAI API key
```

### Step 2: Modify API to Use OpenAI Instead of SageMaker

Edit `rag_llm_system/infrastructure/inference_pipeline_api.py` line 27-31:

```python
# Replace this:
@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()
    return answer

# With this:
from openai import OpenAI

@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    if context:
        messages.append({"role": "system", "content": f"Context: {context}"})
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL_ID,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content
```

### Step 3: Deploy to Render

1. Push code to GitHub
2. Go to https://render.com
3. Click "New +" → "Blueprint"
4. Connect your GitHub repo
5. Select `render.yaml`
6. Add environment variables:
   - `OPENAI_API_KEY=sk-...`
   - `QDRANT_CLOUD_URL=https://...`
   - `QDRANT_APIKEY=...`
   - `DATABASE_HOST=mongodb+srv://...`
   - `HUGGINGFACE_ACCESS_TOKEN=hf_...`
   - `COMET_API_KEY=...`
7. Deploy!

### Step 4: Test

```bash
curl -X POST https://rag-api.onrender.com/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain RAG systems"}'
```

---

## Option 2: Fly.io (~$5/month)

### Deploy
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Launch app
flyctl launch --dockerfile Dockerfile

# Set secrets
flyctl secrets set OPENAI_API_KEY=sk-...
flyctl secrets set QDRANT_CLOUD_URL=https://...
flyctl secrets set QDRANT_APIKEY=...
flyctl secrets set DATABASE_HOST=mongodb+srv://...

# Deploy
flyctl deploy
```

---

## Option 3: DigitalOcean App Platform (~$5/month)

### Deploy via UI
1. Go to https://cloud.digitalocean.com
2. Apps → Create App
3. Connect GitHub repo
4. Select `Dockerfile`
5. Set environment variables
6. Deploy

---

## Cost Breakdown

| Service | Cost/Month | Notes |
|---------|-----------|-------|
| **Render.com** | $7 (or FREE) | 512MB RAM, auto-sleep on free tier |
| **MongoDB Atlas** | FREE | 512MB storage |
| **Qdrant Cloud** | FREE | 1GB vectors |
| **OpenAI API** | ~$3-10 | Pay per use (~1000 queries) |
| **Total** | **~$10-17/month** | Or **$3-10** on free tier |

vs AWS SageMaker: **$864/month** 💰

---

## Next Steps

1. **Deploy backend** (15 minutes)
2. **Create Next.js frontend** (see frontend/ folder)
3. **Test end-to-end**
4. **Monitor via Opik dashboard**

Done! 🚀
