# 🚀 Production-Ready RAG System with LLM Fine-Tuning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> A complete, production-ready RAG (Retrieval-Augmented Generation) system with custom LLM fine-tuning, following industry best practices and MLOps principles.

---

## 🎯 Project Highlights

This is a **full-stack, production-grade AI system** that demonstrates:

- ✅ **Complete ML Pipeline**: From raw data ingestion to production deployment
- ✅ **Advanced RAG Implementation**: Query expansion, self-querying, and cross-encoder reranking
- ✅ **Custom LLM Fine-Tuning**: Supervised Fine-Tuning (SFT) + Direct Preference Optimization (DPO)
- ✅ **Production Infrastructure**: Microservices architecture with separate RAG and inference services
- ✅ **Multi-Cloud Deployment**: AWS SageMaker, RunPod, and Render.com support
- ✅ **Full Observability**: Experiment tracking (Comet ML) and prompt monitoring (Opik)
- ✅ **Clean Architecture**: Domain-Driven Design with ~6,000 lines of well-structured code
- ✅ **CI/CD Pipeline**: Automated testing, linting, and deployment workflows

**Built following**: [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/) best practices.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Documentation](#-documentation)

---

## ✨ Features

### Core Capabilities

**Data Pipeline**
- Multi-source web crawlers (LinkedIn, Medium, GitHub, news sites)
- Automated document processing, chunking, and cleaning
- Vector embeddings generation with sentence-transformers
- Storage in MongoDB (documents) and Qdrant (vectors)

**Advanced RAG System**
- **Self-Query**: Automatic metadata extraction (author, topic, platform)
- **Query Expansion**: Generates 3 diverse query variations for better recall
- **Hybrid Search**: Combines semantic search with metadata filtering
- **Cross-Encoder Reranking**: Precision scoring of retrieved documents
- **Context-Aware Generation**: Structured prompts with retrieved context

**LLM Fine-Tuning**
- Automated dataset generation (instruction + preference pairs)
- Supervised Fine-Tuning (SFT) on AWS SageMaker with Unsloth
- Direct Preference Optimization (DPO) for alignment
- Comprehensive evaluation metrics tracking

**Production Infrastructure**
- FastAPI-based REST API with health checks
- Separate RAG orchestration and LLM inference services
- Docker containerization with multi-stage builds
- ZenML pipeline orchestration
- Monitoring with Comet ML and Opik

### Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11 |
| **LLM** | Llama 3.1-8B (fine-tuned) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-4-v2 |
| **Vector DB** | Qdrant Cloud |
| **Document DB** | MongoDB Atlas |
| **Orchestration** | ZenML |
| **Training** | AWS SageMaker + Unsloth |
| **API Framework** | FastAPI |
| **Frontend** | Next.js + React |
| **Monitoring** | Comet ML + Opik |
| **Deployment** | Docker + AWS/RunPod |

---

## 🏛️ Architecture

### System Architecture

```
┌────────────────────────────────────────────────────────┐
│                   USER INTERFACE                       │
│              Next.js Frontend (Vercel)                 │
└────────────────────────────────────────────────────────┘
                          ↓ HTTP
┌────────────────────────────────────────────────────────┐
│                  RAG ORCHESTRATION API                 │
│        FastAPI • Query Processing • Retrieval          │
│     (Render.com - CPU Only, $7/mo or FREE)             │
│                                                        │
│  • Self-query metadata extraction                      │
│  • Query expansion (3 variations)                      │
│  • Vector search (Qdrant Cloud)                        │
│  • Cross-encoder reranking                             │
│  • Context formatting                                  │
└────────────────────────────────────────────────────────┘
                          ↓ HTTP
┌────────────────────────────────────────────────────────┐
│                  LLM INFERENCE SERVICE                 │
│         Llama 3.1-8B • Text Generation                 │
│      (RunPod/SageMaker - GPU Required)                 │
│                                                        │
│  • Model loading with vLLM                             │
│  • Response generation                                 │
│  • Temperature-controlled sampling                     │
└────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────────────┐
│  Data Collection │  Web Crawlers → MongoDB
└────────┬─────────┘
         ↓
┌──────────────────┐
│Feature Engineering│  Clean → Chunk → Embed → Qdrant
└────────┬─────────┘
         ↓
┌──────────────────┐
│Dataset Generation│  Instruction + Preference Datasets
└────────┬─────────┘
         ↓
┌──────────────────┐
│  LLM Fine-Tuning │  SFT → DPO → Evaluation
└────────┬─────────┘
         ↓
┌──────────────────┐
│  RAG Inference   │  Query → Retrieve → Generate
└──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry (dependency manager)
- Docker & Docker Compose
- API Keys: OpenAI, Hugging Face, Comet ML

### 5-Minute Setup

```bash
# 1. Clone repository
git clone https://github.com/silsgah/rag-llm-system.git
cd rag-llm-system

# 2. Install dependencies
poetry install

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start local infrastructure (MongoDB + Qdrant + ZenML)
poetry poe local-infrastructure-up

# 5. Run data pipeline
poetry poe run-digital-data-etl
poetry poe run-feature-engineering-pipeline

# 6. Test RAG system
poetry poe call-rag-retrieval-module
```

### Test the API

```bash
# Terminal 1: Start API server
poetry poe run-inference-ml-service

# Terminal 2: Send test query
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What topics are discussed in the posts?"}'
```

---

## 📁 Project Structure

```
rag-llm-system/
├── rag_llm_system/           # Main package (6,000+ LOC)
│   ├── domain/               # Core entities (DDD pattern)
│   │   ├── documents.py      # Document models
│   │   ├── embedded_chunks.py # Vector embeddings
│   │   └── queries.py        # Query models
│   ├── application/          # Business logic
│   │   ├── crawlers/         # Web scrapers (LinkedIn, Medium, GitHub)
│   │   ├── rag/              # RAG implementation
│   │   │   ├── retriever.py  # Main retrieval logic
│   │   │   ├── query_expansion.py
│   │   │   ├── self_query.py
│   │   │   └── reranking.py
│   │   ├── preprocessing/    # Document processing
│   │   ├── dataset/          # Training dataset generation
│   │   └── networks/         # Embedding models
│   ├── model/                # LLM training & inference
│   │   ├── finetuning/       # SFT & DPO training
│   │   ├── evaluation/       # Model evaluation
│   │   └── inference/        # Deployment logic
│   └── infrastructure/       # External integrations
│       ├── db/               # MongoDB & Qdrant clients
│       ├── aws/              # SageMaker deployment
│       └── inference_pipeline_api.py  # FastAPI app
│
├── pipelines/                # ZenML ML pipelines
├── steps/                    # Reusable pipeline steps
├── tools/                    # Utility scripts
│   ├── run.py               # Pipeline executor
│   ├── rag.py               # RAG demo
│   └── ml_service.py        # API server entry point
├── configs/                  # YAML configurations
├── frontend/                 # Next.js chat interface
├── tests/                    # Test suite
├── .github/workflows/        # CI/CD pipelines
│   ├── ci.yaml              # Linting, testing
│   └── cd.yaml              # Deployment
├── docker-compose.yml        # Local infrastructure
├── Dockerfile                # Production build
├── pyproject.toml           # Poetry dependencies
└── README.md                # This file
```

---

## 💻 Usage

### Data Pipelines

```bash
# Collect data from web sources
poetry poe run-digital-data-etl

# Process and embed documents
poetry poe run-feature-engineering-pipeline

# Generate training datasets
poetry poe run-generate-instruct-datasets-pipeline
poetry poe run-generate-preference-datasets-pipeline

# Or run all at once
poetry poe run-end-to-end-data-pipeline
```

### RAG System

```bash
# Interactive CLI test
poetry poe call-rag-retrieval-module

# Start API server
poetry poe run-inference-ml-service

# Test API endpoint
poetry poe call-inference-ml-service
```

### Training (Requires AWS)

```bash
# Setup AWS credentials
poetry install --with aws
poetry poe create-sagemaker-role

# Fine-tune with SFT
poetry poe run-training-pipeline

# Fine-tune with DPO (update configs/training.yaml first)
poetry poe run-training-pipeline

# Evaluate models
poetry poe run-evaluation-pipeline
```

### Available Commands

| Command | Description |
|---------|-------------|
| `poetry poe local-infrastructure-up` | Start MongoDB + Qdrant + ZenML |
| `poetry poe run-digital-data-etl` | Collect web data |
| `poetry poe run-feature-engineering-pipeline` | Process & embed documents |
| `poetry poe call-rag-retrieval-module` | Test RAG retrieval |
| `poetry poe run-inference-ml-service` | Start FastAPI server |
| `poetry poe run-training-pipeline` | Fine-tune LLM on SageMaker |
| `poetry poe test` | Run test suite |
| `poetry poe lint-check` | Check code quality |
| `poetry poe format-fix` | Auto-format code |

---

## 🚢 Deployment

### Architecture Overview

The system uses a **microservices architecture** with two separate services:

1. **RAG Orchestration API** (CPU-only, lightweight)
   - Handles query processing, retrieval, and context formatting
   - Deploy to: Render.com ($7/mo) or Free tier
   - Requirements: 1GB RAM, no GPU needed

2. **LLM Inference Service** (GPU-required, heavy)
   - Hosts the fine-tuned Llama 3.1-8B model
   - Deploy to: RunPod ($10-20/mo) or AWS SageMaker
   - Requirements: 16GB VRAM (A10G/T4 GPU)

3. **Frontend** (Static, free)
   - Next.js chat interface
   - Deploy to: Vercel (FREE)

### Option 1: Local Development

```bash
# Start local infrastructure
poetry poe local-infrastructure-up

# Access ZenML dashboard
open http://localhost:8237

# Run inference locally (requires GPU)
poetry poe deploy-inference-local
poetry poe test-inference-local
```

### Option 2: Cloud Deployment (Recommended)

**Step 1: Deploy Inference Service to RunPod**

```bash
# Build and push Docker image
docker build -f Dockerfile.runpod -t rag-inference:latest .
docker tag rag-inference:latest <your-dockerhub>/rag-inference:latest
docker push <your-dockerhub>/rag-inference:latest

# Deploy on RunPod:
# 1. Go to https://runpod.io/console/pods
# 2. Select GPU: RTX 4090 or A10G (24GB VRAM)
# 3. Use custom Docker image
# 4. Set env var: HUGGINGFACE_HUB_TOKEN
# 5. Expose port: 8000
# 6. Note the endpoint URL
```

**Step 2: Deploy RAG API to Render.com**

```bash
# Update .env with RunPod endpoint
USE_LOCAL_INFERENCE=true
LOCAL_INFERENCE_ENDPOINT_URL=https://your-runpod-endpoint.proxy.runpod.net/infer

# Deploy using render.yaml
# 1. Connect GitHub repo to Render.com
# 2. Create new Web Service
# 3. Use existing render.yaml configuration
# 4. Add environment variables from .env
# 5. Deploy
```

**Step 3: Deploy Frontend to Vercel**

```bash
cd frontend

# Install Vercel CLI
npm i -g vercel

# Set API URL
echo "API_URL=https://your-api.onrender.com" > .env.local

# Deploy
vercel --prod
```

### Option 3: AWS SageMaker (Enterprise)

```bash
# Setup AWS
poetry install --with aws
poetry poe create-sagemaker-execution-role

# Deploy endpoint
poetry poe deploy-inference-endpoint

# Test
poetry poe test-sagemaker-endpoint

# Update .env
USE_LOCAL_INFERENCE=false
SAGEMAKER_ENDPOINT_INFERENCE=your-endpoint-name
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file from template:

```bash
# OpenAI (Required for RAG)
OPENAI_API_KEY=sk-...
OPENAI_MODEL_ID=gpt-4o-mini

# Hugging Face (Required)
HUGGINGFACE_ACCESS_TOKEN=hf_...

# Comet ML (Required for monitoring)
COMET_API_KEY=...
COMET_PROJECT=rag-llm-system

# MongoDB Atlas (Required)
DATABASE_HOST=mongodb+srv://user:pass@cluster.mongodb.net

# Qdrant Cloud (Required)
USE_QDRANT_CLOUD=true
QDRANT_CLOUD_URL=https://your-cluster.qdrant.io
QDRANT_APIKEY=...

# Inference Configuration
USE_LOCAL_INFERENCE=true  # or false for SageMaker
LOCAL_INFERENCE_ENDPOINT_URL=http://localhost:8000/infer

# AWS (Optional - only for training/SageMaker)
AWS_REGION=us-east-1
AWS_ACCESS_KEY=...
AWS_SECRET_KEY=...
AWS_ARN_ROLE=arn:aws:iam::...
```

### Customize Data Sources

Edit `configs/digital_data_etl_*.yaml`:

```yaml
links:
  - https://medium.com/@yourusername
  - https://dev.to/yourusername
  - https://github.com/yourusername
```

### Customize Training

Edit `configs/training.yaml`:

```yaml
finetuning_type: "sft"  # or "dpo"
num_train_epochs: 3
per_device_train_batch_size: 2
learning_rate: 3e-4
model_id: "meta-llama/Llama-3.1-8B-Instruct"
```

### Customize RAG

Edit `rag_llm_system/settings.py`:

```python
TEXT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKING_CROSS_ENCODER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-4-v2"
OPENAI_MODEL_ID = "gpt-4o-mini"
TEMPERATURE_INFERENCE = 0.7
```

---

## 📊 Performance

- **RAG Latency**: 2-5 seconds (including reranking)
- **Training Time**: 2-4 hours (SFT on 8B model with Unsloth)
- **Throughput**: ~10 requests/second (with GPU inference)
- **Context Window**: 8K tokens (Llama 3.1)
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)

---

## 🐛 Troubleshooting

### Common Issues

**MongoDB Authentication Failed**
```bash
# Verify credentials in .env
# Add IP whitelist in MongoDB Atlas: 0.0.0.0/0 (for testing)
```

**Qdrant 404 Not Found**
```bash
# Ensure cloud URL has no :6333 port suffix
QDRANT_CLOUD_URL=https://cluster.qdrant.io  # ✓ Correct
```

**OpenAI Rate Limit**
```bash
# Reduce batch size in dataset generation configs
# Or add retry delays in code
```

**Pydantic Version Conflict**
```bash
poetry add "pydantic>=2.8.0,<2.9.0"
poetry install
```

### Debug Tools

```bash
# Check data status
poetry run python check_data_status.py

# View ZenML pipeline runs
open http://localhost:8237

# View logs
tail -f logs/app.log

# Test RAG retrieval
poetry poe call-rag-retrieval-module
```

---

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed system architecture and design decisions
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**: Comprehensive deployment instructions
- **[AI_GOVERNANCE_GUIDE.md](AI_GOVERNANCE_GUIDE.md)**: AI governance and compliance framework
- **[VECTOR_DB_INTEGRATION_GUIDE.md](VECTOR_DB_INTEGRATION_GUIDE.md)**: Vector database integration details
- **[RAG_MODULE_TRACE.md](RAG_MODULE_TRACE.md)**: RAG pipeline execution trace

---

## 🧪 Testing

```bash
# Run all tests
poetry poe test

# Lint check
poetry poe lint-check

# Format code
poetry poe format-fix

# Docker lint
poetry poe lint-check-docker
```

---

## 💰 Cost Estimate

**Development (One-time)**
- MongoDB Atlas: FREE (512MB tier)
- Qdrant Cloud: FREE (1GB tier)
- OpenAI API: ~$3 (dataset generation)
- AWS SageMaker Training: ~$20 (2-4 hours)
- **Total**: ~$25

**Production (Monthly)**
- Frontend (Vercel): FREE
- RAG API (Render.com): $7/mo or FREE tier
- Inference (RunPod): $10-20/mo (pay-per-use)
- OR Inference (SageMaker): ~$100-200/mo (ml.g5.xlarge)
- **Total**: $7-227/mo depending on deployment choice

---

## 🙏 Acknowledgments

- Based on [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/) by Paul Iusztin and Maxime Labonne
- Built with [ZenML](https://zenml.io/), [Unsloth](https://github.com/unslothai/unsloth), and [LangChain](https://langchain.com/)
- Fine-tuned models hosted on [Hugging Face Hub](https://huggingface.co/)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Contact

- **Author**: Silas Kwabla Gah
- **GitHub**: [@silsgah](https://github.com/silsgah)
- **LinkedIn**: [Silas Gah](https://www.linkedin.com/in/silas-gah-46b126294)
- **Email**: gahsilas@gmail.com

---

**⭐ If you found this project helpful, please star the repository!**

---

## 🎓 Learning Resources

**Key Concepts Demonstrated:**
- RAG system architecture and implementation
- LLM fine-tuning (SFT + DPO)
- Vector database integration
- MLOps best practices
- Microservices architecture
- CI/CD for ML systems
- Production deployment strategies
- Monitoring and observability

**Skills Showcased:**
- Python (FastAPI, Poetry, type hints)
- Machine Learning (transformers, sentence-transformers)
- MLOps (ZenML, Docker, CI/CD)
- Cloud Infrastructure (AWS, Docker)
- Database Design (MongoDB, Qdrant)
- API Design (REST, async)
- Frontend Development (Next.js)
- System Architecture
