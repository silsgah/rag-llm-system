"""
Local deployment entry point for running Hugging Face models locally (GPU-enabled).
This mirrors the SageMaker deployment flow, but runs entirely on local infrastructure.
"""

import os

import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from rag_llm_system.infrastructure.local.governance import GovernanceMonitor
from rag_llm_system.settings import settings

# -------------------------------------------------------------------
# 1. Configuration & Model Setup
# -------------------------------------------------------------------

HF_MODEL_ID = os.getenv("HF_MODEL_ID", settings.HF_MODEL_ID or "meta-llama/Llama-3-8b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", settings.HUGGINGFACE_ACCESS_TOKEN)

logger.info(f"🚀 Starting local model deployment for {HF_MODEL_ID} on {DEVICE}...")

# Authenticate if necessary
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    logger.info("✅ Using authenticated Hugging Face Hub access.")

# Load model and tokenizer
logger.info("🔄 Loading tokenizer and model from Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    token=HF_TOKEN,
)

# Create a text-generation pipeline
logger.info("⚙️ Initializing text generation pipeline...")
generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)

# -------------------------------------------------------------------
# 2. Governance & Compliance Wrapper
# -------------------------------------------------------------------

governance = GovernanceMonitor(log_to="logs/governance_local.log")

# -------------------------------------------------------------------
# 3. FastAPI App for Local Inference
# -------------------------------------------------------------------

app = FastAPI(title="Local LLM Inference API", version="1.0.0")


class InferenceRequest(BaseModel):
    prompt: str


class InferenceResponse(BaseModel):
    generated_text: str
    compliance_status: str


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    prompt = request.prompt

    # Governance pre-check (e.g., banned terms, PII filters)
    if not governance.check_input(prompt):
        return InferenceResponse(
            generated_text="Input rejected by governance filter.", compliance_status="non_compliant"
        )

    logger.info(f"🧠 Generating response for: {prompt[:80]}...")
    output = generator(prompt, max_new_tokens=256, do_sample=True)[0]["generated_text"]

    # Governance post-check
    compliance_status = governance.check_output(output)

    governance.log_inference(prompt, output, compliance_status)

    return InferenceResponse(generated_text=output, compliance_status=compliance_status)


@app.get("/health")
async def health_check():
    return {"status": "running", "device": DEVICE, "model": HF_MODEL_ID}


# -------------------------------------------------------------------
# 4. Main entry point for Poetry script
# -------------------------------------------------------------------
def main():
    import uvicorn

    logger.info("🔥 Running Local LLM API on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
