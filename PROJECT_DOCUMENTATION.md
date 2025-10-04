# Complete Project Documentation

## 🎯 Project Overview

This document provides a comprehensive guide to the RAG system with LLM fine-tuning capabilities, including setup, troubleshooting, and deployment.

---

## 📚 Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Issues Resolved](#issues-resolved)
3. [Complete Workflow](#complete-workflow)
4. [Architecture Deep Dive](#architecture-deep-dive)
5. [Configuration Guide](#configuration-guide)
6. [Deployment Guide](#deployment-guide)
7. [Migration to UV](#migration-to-uv)

---

## 🔧 Setup & Installation

### Prerequisites Checklist

- [x] Python 3.11.8
- [x] Poetry >= 1.8.3
- [x] Docker >= 27.1.1
- [x] AWS CLI >= 2.15.42
- [x] Git >= 2.44.0

### Cloud Services Required

| Service | Purpose | Cost |
|---------|---------|------|
| OpenAI API | Query expansion, metadata extraction | Pay-per-use (~$3) |
| Hugging Face | Model registry | Free |
| Comet ML/Opik | Experiment tracking, monitoring | Free tier available |
| MongoDB Atlas | Document storage | Free tier (512MB) |
| Qdrant Cloud | Vector database | Free tier (1GB) |
| AWS SageMaker | Model training & deployment | ~$20-25 per run |

### Initial Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd <repo-name>

# 2. Python environment
pyenv install 3.11.8
pyenv local 3.11.8

# 3. Install dependencies
poetry install

# 4. Setup environment
cp .env.example .env
# Edit .env with your credentials

# 5. Pre-commit hooks
poetry run pre-commit install

# 6. Start local infrastructure
poetry poe local-infrastructure-up
```

---

## 🐛 Issues Resolved

### 1. Pydantic Version Conflict

**Problem:**
```
RuntimeError: Unable to apply constraint 'union_mode' to schema of type 'tagged-union'
```

**Cause:** ZenML 0.74.0 incompatible with Pydantic 2.11.9

**Solution:**
```bash
poetry add "pydantic>=2.8.0,<2.9.0"
poetry install
```

**Files affected:** All imports using Pydantic models

---

### 2. PyTorch/Torchvision Compatibility

**Problem:**
```
AttributeError: module 'torch.library' has no attribute 'register_fake'
```

**Cause:** PyTorch 2.2.2 requires torchvision 0.17.2, but 0.22.1 was installed

**Solution:**
```bash
poetry add "torchvision==0.17.2"
poetry install
```

---

### 3. Hugging Face Hub API Changes

**Problem:**
```
ImportError: cannot import name 'LocalEntryNotFoundError' from 'huggingface_hub.errors'
```

**Cause:** Downgraded huggingface-hub (0.24.5) missing required classes

**Solution:**
```bash
poetry add "huggingface-hub>=0.25.0"
poetry install
```

---

### 4. MongoDB Authentication Failure

**Problem:**
```
OperationFailure: bad auth : authentication failed
```

**Cause:** Incorrect credentials or network access restrictions

**Solutions:**
1. Verify credentials in `.env`
2. Add IP to MongoDB Atlas Network Access (0.0.0.0/0 for testing)
3. Check connection string format:
   ```
   mongodb+srv://username:password@cluster.mongodb.net
   ```

---

### 5. Qdrant Connection Issues

**Problem:**
```
HTTP/1.1 404 Not Found
```

**Causes & Solutions:**

**A. Wrong URL format (Cloud)**
```bash
# Wrong - includes port
QDRANT_CLOUD_URL=https://cluster.qdrant.io:6333

# Correct - no port for cloud
QDRANT_CLOUD_URL=https://cluster.qdrant.io
```

**B. Collections don't exist**
```bash
# Run feature engineering to create collections
poetry poe run-feature-engineering-pipeline
```

**C. Wrong cluster**
- Switched Qdrant clusters
- Data exists in old cluster, querying new cluster
- Solution: Re-run feature engineering on new cluster

---

### 6. RAG Returning 0 Documents

**Problem:**
```
0 documents retrieved successfully
```

**Causes & Solutions:**

**A. Author filter mismatch**
```python
# Query asks for "Paul Iusztin"
query = "My name is Paul Iusztin. ..."

# But database only has "Ghana News Collector"
# Solution: Remove author name or collect relevant data
query = "Could you draft a LinkedIn post..."
```

**B. Empty collections**
```bash
# Check collections
poetry run python check_data_status.py

# Populate if empty
poetry poe run-feature-engineering-pipeline
```

**C. Missing collection types**
```bash
# RAG searches 3 collections:
# - embedded_posts (LinkedIn/social posts)
# - embedded_articles (blog articles)
# - embedded_repositories (GitHub repos)

# If only articles exist, posts/repos will 404 (expected)
```

---

## 🔄 Complete Workflow

### Phase 1: Data Collection (Local)

```bash
# Step 1: Collect raw data
poetry poe run-digital-data-etl
```

**What happens:**
- Crawls configured URLs (news sites, Medium, etc.)
- Stores in MongoDB `articles` collection
- Creates `users` collection with authors

**Output:**
- MongoDB: 19 articles
- MongoDB: 3 users

**Duration:** ~30 seconds

---

### Phase 2: Feature Engineering (Local)

```bash
# Step 2: Process and embed documents
poetry poe run-feature-engineering-pipeline
```

**What happens:**
1. **Query MongoDB**: Load raw articles
2. **Clean**: Remove HTML, normalize text
3. **Chunk**: Split into 256-token pieces
4. **Embed**: Generate 384-dim vectors (sentence-transformers)
5. **Store in Qdrant**: Create collections

**Output:**
- Qdrant: `cleaned_articles` (19 docs)
- Qdrant: `embedded_articles` (50 chunks)

**Duration:** ~5-10 seconds

**ZenML Pipeline Steps:**
```
query_data_warehouse → clean_documents → chunk_and_embed → load_to_vector_db
```

---

### Phase 3: Dataset Generation (Uses OpenAI)

```bash
# Step 3: Generate instruction dataset
poetry poe run-generate-instruct-datasets-pipeline

# Step 4: Generate preference dataset
poetry poe run-generate-preference-datasets-pipeline
```

**What happens:**

**Instruct Dataset (SFT):**
- Uses GPT-4o-mini to generate Q&A pairs from articles
- Creates training data for supervised fine-tuning
- Format: `{"prompt": "...", "completion": "..."}`

**Preference Dataset (DPO):**
- Generates chosen vs rejected responses
- Creates training data for preference alignment
- Format: `{"prompt": "...", "chosen": "...", "rejected": "..."}`

**Output:**
- Saved to Hugging Face Hub
- Also available in ZenML artifacts

**Duration:** 10-30 minutes (depends on data size)
**Cost:** ~$2-3 in OpenAI API calls

---

### Phase 4: Training (AWS SageMaker)

```bash
# Step 5: Fine-tune with SFT
poetry poe run-training-pipeline

# Step 6: Fine-tune with DPO (update config first)
# Edit configs/training.yaml: finetuning_type: "dpo"
poetry poe run-training-pipeline

# Step 7: Evaluate models
poetry poe run-evaluation-pipeline
```

**What happens:**

**SFT Training:**
- Trains Llama 3.1-8B on instruct dataset
- Uses Unsloth for 2x speed + memory optimization
- Uploads to Hugging Face

**DPO Training:**
- Further aligns model with preferences
- Trains on preference dataset
- Uploads to Hugging Face

**Evaluation:**
- Runs test prompts
- Computes metrics
- Saves results to Hugging Face

**Output:**
- Hugging Face: `<username>/TwinLlama-3.1-8B-SFT`
- Hugging Face: `<username>/TwinLlama-3.1-8B-DPO`
- Comet ML: Experiment tracking

**Duration:** 2-4 hours per training
**Cost:** ~$20-25 per training run

---

### Phase 5: Inference & RAG (Local + AWS)

```bash
# Step 8: Test RAG retrieval
poetry poe call-rag-retrieval-module

# Step 9: Deploy to SageMaker
poetry poe deploy-inference-endpoint

# Step 10: Test SageMaker endpoint
poetry poe test-sagemaker-endpoint

# Step 11: Start local API server
poetry poe run-inference-ml-service

# Step 12: Test API
poetry poe call-inference-ml-service
```

**What happens:**

**RAG Flow:**
```
Query → Self-Query → Query Expansion → Embedding
  ↓
Vector Search (Qdrant) → Reranking → Top-k docs
  ↓
LLM Generation (via SageMaker) → Response
```

**Components:**
1. **Self-Query**: Extract author name using GPT
2. **Query Expansion**: Generate 3 variations using GPT
3. **Embedding**: sentence-transformers (local)
4. **Search**: Qdrant vector similarity
5. **Rerank**: Cross-encoder scoring (local)
6. **Generate**: Fine-tuned Llama via SageMaker

**Output:**
- Retrieved documents with relevance scores
- Generated response

**Duration:** 2-5 seconds per query
**Cost:** SageMaker endpoint ~$100-200/month

---

## 🏗️ Architecture Deep Dive

### Domain-Driven Design Structure

```
llm_engineering/
├── domain/              # Core business entities
│   ├── base/           # Abstract base classes
│   │   ├── nosql.py   # MongoDB ODM
│   │   └── vector.py  # Qdrant ODM
│   ├── documents.py    # Raw documents (Article, Post, Repo)
│   ├── chunks.py       # Chunked documents
│   ├── embedded_chunks.py  # Vector embeddings
│   ├── queries.py      # Search queries
│   └── dataset.py      # Training datasets
│
├── application/        # Business logic
│   ├── crawlers/       # Web scrapers
│   │   ├── base.py
│   │   ├── linkedin.py
│   │   ├── medium.py
│   │   └── github.py
│   ├── preprocessing/  # Data processing
│   │   ├── dispatchers.py       # Factory pattern
│   │   ├── cleaning_data_handlers.py
│   │   ├── chunking_data_handlers.py
│   │   └── embedding_data_handlers.py
│   ├── rag/            # RAG implementation
│   │   ├── retriever.py         # Main RAG class
│   │   ├── self_query.py        # Metadata extraction
│   │   ├── query_expanison.py   # Query expansion
│   │   ├── reranking.py         # Cross-encoder reranking
│   │   └── prompt_templates.py  # LLM prompts
│   ├── dataset/        # Dataset generation
│   │   └── generation.py
│   └── networks/       # ML models
│       └── embeddings.py
│
├── model/              # LLM training & inference
│   ├── finetuning/
│   │   ├── finetune.py         # Training logic
│   │   └── sagemaker.py        # SageMaker integration
│   ├── evaluation/
│   │   └── evaluate.py
│   └── inference/
│       └── inference.py
│
└── infrastructure/     # External integrations
    ├── db/
    │   ├── mongo.py    # MongoDB connection
    │   └── qdrant.py   # Qdrant connection
    └── aws/
        └── deploy/     # SageMaker deployment
```

### Design Patterns Used

1. **Factory Pattern**: `dispatchers.py` creates handlers based on data type
2. **Singleton Pattern**: `embeddings.py` for model instances
3. **Repository Pattern**: `base/nosql.py` and `base/vector.py`
4. **Strategy Pattern**: Different cleaning/chunking handlers
5. **Observer Pattern**: ZenML for pipeline orchestration

---

## ⚙️ Configuration Guide

### Environment Variables (.env)

```bash
# === REQUIRED FOR LOCAL DEVELOPMENT ===

# OpenAI (for RAG query expansion)
OPENAI_MODEL_ID=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Hugging Face (for models)
HUGGINGFACE_ACCESS_TOKEN=hf_...

# Comet ML (for tracking)
COMET_API_KEY=...
COMET_PROJECT=twin

# === REQUIRED FOR DEPLOYMENT ===

# MongoDB Atlas
DATABASE_HOST=mongodb+srv://user:pass@cluster.mongodb.net
DATABASE_NAME=twin

# Qdrant Cloud
USE_QDRANT_CLOUD=true
QDRANT_CLOUD_URL=https://cluster.qdrant.io
QDRANT_APIKEY=...

# === REQUIRED FOR TRAINING ===

# AWS
AWS_REGION=eu-north-1
AWS_ACCESS_KEY=...
AWS_SECRET_KEY=...
AWS_ARN_ROLE=arn:aws:iam::...

# === OPTIONAL TWEAKS ===

# Model settings
HF_MODEL_ID=mlabonne/TwinLlama-3.1-8B-DPO
TEXT_EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
RERANKING_CROSS_ENCODER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-4-v2

# SageMaker instance
GPU_INSTANCE_TYPE=ml.g5.2xlarge
SM_NUM_GPUS=1

# RAG settings
RAG_MODEL_DEVICE=cpu  # or "cuda" if GPU available
```

### Pipeline Configurations (configs/)

**Training (`configs/training.yaml`):**
```yaml
finetuning_type: "sft"  # or "dpo"
model_id: "meta-llama/Llama-3.1-8B-Instruct"
num_train_epochs: 3
per_device_train_batch_size: 2
learning_rate: 3e-4
max_seq_length: 2048
dataset_huggingface_workspace: "your-username"
```

**Feature Engineering (`configs/feature_engineering.yaml`):**
```yaml
data_collection:
  user: "Ghana News Collector"
  chunk_size: 256
  chunk_overlap: 50
embedding:
  model_id: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
```

---

## 🚢 Deployment Guide

### Local Development

**Infrastructure:**
```bash
# Start all services
poetry poe local-infrastructure-up

# Services available:
# - MongoDB: mongodb://localhost:27017
# - Qdrant: http://localhost:6333
# - ZenML: http://localhost:8237
```

**API Server:**
```bash
# Start FastAPI server
poetry poe run-inference-ml-service

# Test endpoint
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain RAG"}'
```

---

### AWS SageMaker Production

**Prerequisites:**
```bash
# Install AWS dependencies
poetry install --with aws

# Configure AWS CLI
aws configure

# Create SageMaker roles
poetry poe create-sagemaker-role
poetry poe create-sagemaker-execution-role
```

**Training:**
```bash
# Train model
poetry poe run-training-pipeline

# Monitor in Comet ML
open https://www.comet.com/
```

**Deployment:**
```bash
# Deploy endpoint
poetry poe deploy-inference-endpoint

# Test
poetry poe test-sagemaker-endpoint

# Delete when done (to save costs)
poetry poe delete-inference-endpoint
```

**Cost Optimization:**
- Use `ml.g5.2xlarge` (cheapest GPU instance)
- Delete endpoints when not in use
- Use spot instances for training (50-70% cheaper)

---

### Docker Deployment

**Build:**
```bash
poetry poe build-docker-image
```

**Run:**
```bash
docker run --rm \
  --network host \
  --env-file .env \
  llmtwin \
  poetry poe run-inference-ml-service
```

---

## 🔄 Migration to UV

UV is a modern, faster alternative to Poetry. See `RENAME_PROJECT_GUIDE.md` for details.

**Benefits:**
- 10-100x faster dependency resolution
- Better caching
- pip-compatible

**Migration:**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Convert project
uv sync

# Use instead of poetry
uv run python -m tools.rag
```

---

## 📊 Performance Benchmarks

### RAG System

| Metric | Value |
|--------|-------|
| **Query Latency** | 2-5 seconds |
| **Embedding Time** | 50-100ms |
| **Vector Search** | 10-20ms |
| **Reranking** | 1-2 seconds |
| **LLM Generation** | 1-3 seconds |
| **Throughput** | ~10 req/sec |

### Training

| Phase | Duration | Cost |
|-------|----------|------|
| **Data Collection** | 30s | Free |
| **Feature Engineering** | 10s | Free |
| **Dataset Generation** | 20min | ~$3 |
| **SFT Training** | 2-4h | ~$20 |
| **DPO Training** | 2-4h | ~$20 |
| **Evaluation** | 30min | ~$5 |

---

## 🎓 Learning Resources

1. **Book**: [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/)
2. **ZenML**: https://docs.zenml.io/
3. **Unsloth**: https://github.com/unslothai/unsloth
4. **LangChain**: https://python.langchain.com/
5. **Qdrant**: https://qdrant.tech/documentation/

---

## 📝 Changelog

### Version 1.0.0 (2025-10-04)

**Added:**
- Complete RAG system with advanced retrieval
- SFT & DPO fine-tuning pipelines
- AWS SageMaker deployment
- Comprehensive monitoring with Opik

**Fixed:**
- Pydantic version conflict (2.11.9 → 2.8.2)
- PyTorch/Torchvision compatibility (0.17.2)
- Hugging Face Hub API compatibility (>=0.25.0)
- MongoDB authentication issues
- Qdrant connection issues

**Improved:**
- Documentation (README, guides, traces)
- Error handling and logging
- Configuration management

---

**Document Version:** 1.0
**Last Updated:** 2025-10-04
**Maintainer:** Project Team
