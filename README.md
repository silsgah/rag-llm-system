# 🚀 Production-Ready RAG System with LLM Fine-Tuning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A complete, production-ready RAG (Retrieval-Augmented Generation) system with custom LLM fine-tuning, built following industry best practices. Based on the [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Pipelines](#-pipelines)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project demonstrates how to build a **complete LLM-powered system** from scratch, including:

- 🔍 **Data Collection**: Web crawlers for LinkedIn, Medium, GitHub, and news sources
- ⚙️ **Feature Engineering**: Document processing, chunking, and embedding generation
- 🤖 **LLM Fine-Tuning**: Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO)
- 📊 **RAG System**: Advanced retrieval with query expansion, self-querying, and reranking
- 🚀 **Production Deployment**: AWS SageMaker, Docker, CI/CD with GitHub Actions
- 📈 **Monitoring**: Comprehensive tracking with Comet ML and Opik

**Use Cases:**
- Build AI assistants that learn from your content
- Create personalized chatbots
- Implement enterprise knowledge bases
- Research LLM fine-tuning and RAG systems

---

## ✨ Features

### 🔧 Core Capabilities

- **Multi-Source Data Collection**: Automated crawlers for web content
- **Vector Database**: Qdrant for efficient similarity search
- **Advanced RAG**:
  - Query expansion for better recall
  - Self-querying for metadata extraction
  - Cross-encoder reranking for precision
- **Custom Fine-Tuning**: Train on your own data using SFT and DPO
- **Production-Ready**: Scalable deployment on AWS SageMaker
- **Full Observability**: Experiment tracking and prompt monitoring

### 🏗️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11 |
| **LLM** | Llama 3.1-8B (fine-tuned) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector DB** | Qdrant Cloud |
| **Document DB** | MongoDB Atlas |
| **Orchestration** | ZenML |
| **Training** | AWS SageMaker + Unsloth |
| **API Framework** | FastAPI |
| **Monitoring** | Comet ML + Opik |
| **Deployment** | Docker + AWS |

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                           │
│  Web Crawlers → MongoDB (Raw Documents)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                         │
│  Clean → Chunk → Embed → Qdrant (Vector Store)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  DATASET GENERATION                          │
│  Generate Instruct & Preference Datasets                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    LLM FINE-TUNING                           │
│  SFT Training → DPO Training → Model Evaluation              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE (RAG)                           │
│  Query → Retrieve → Rerank → Generate Response               │
└─────────────────────────────────────────────────────────────┘
```

**RAG Pipeline Detail:**
```
User Query
    ↓
[Self-Query] → Extract metadata (author, topic)
    ↓
[Query Expansion] → Generate 3 query variations
    ↓
[Embedding] → Convert to vector (384-dim)
    ↓
[Vector Search] → Query Qdrant (articles, posts, repos)
    ↓
[Reranking] → Cross-encoder scoring
    ↓
[LLM Generation] → Final response
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Poetry (dependency manager)
- Docker (for local services)
- OpenAI API key
- Hugging Face account
- Comet ML account

### 5-Minute Setup

```bash
# 1. Clone repository
git clone <your-repo-url>
cd <repo-name>

# 2. Install dependencies
poetry install

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start local infrastructure
poetry poe local-infrastructure-up

# 5. Run data pipeline
poetry poe run-digital-data-etl
poetry poe run-feature-engineering-pipeline

# 6. Test RAG
poetry poe call-rag-retrieval-module
```

**You're ready!** 🎉

---

## 📦 Installation

### Step 1: Environment Setup

```bash
# Using pyenv (recommended)
pyenv install 3.11.8
pyenv local 3.11.8

# Verify
python --version  # Should show 3.11.x
```

### Step 2: Install Dependencies

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate environment
poetry shell
```

### Step 3: Configure Environment Variables

Create `.env` file:

```bash
# OpenAI (Required)
OPENAI_API_KEY=sk-...
OPENAI_MODEL_ID=gpt-4o-mini

# Hugging Face (Required)
HUGGINGFACE_ACCESS_TOKEN=hf_...

# Comet ML (Required for training/monitoring)
COMET_API_KEY=...
COMET_PROJECT=twin

# MongoDB Atlas (Required)
DATABASE_HOST=mongodb+srv://user:pass@cluster.mongodb.net

# Qdrant Cloud (Required)
USE_QDRANT_CLOUD=true
QDRANT_CLOUD_URL=https://your-cluster.qdrant.io
QDRANT_APIKEY=...

# AWS (Required for training/deployment)
AWS_REGION=eu-north-1
AWS_ACCESS_KEY=...
AWS_SECRET_KEY=...
AWS_ARN_ROLE=arn:aws:iam::...
```

### Step 4: Start Local Services

```bash
# Start MongoDB + Qdrant + ZenML
poetry poe local-infrastructure-up

# Access ZenML dashboard
open http://localhost:8237
```

---

## 💻 Usage

### Data Pipeline

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
# Test retrieval
poetry poe call-rag-retrieval-module

# Start API server
poetry poe run-inference-ml-service

# Test API
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain RAG systems"}'
```

### Training (Requires AWS)

```bash
# Setup AWS SageMaker
poetry install --with aws
poetry poe create-sagemaker-role
poetry poe create-sagemaker-execution-role

# Fine-tune with SFT
poetry poe run-training-pipeline

# Fine-tune with DPO (update configs/training.yaml first)
poetry poe run-training-pipeline

# Evaluate models
poetry poe run-evaluation-pipeline

# Deploy to SageMaker
poetry poe deploy-inference-endpoint
poetry poe test-sagemaker-endpoint
```

---

## 📁 Project Structure

```
.
├── rag_llm_system/          # Main package (Domain-Driven Design)
│   ├── domain/              # Core entities (documents, chunks, queries)
│   ├── application/         # Business logic
│   │   ├── crawlers/        # Web scrapers
│   │   ├── rag/             # RAG implementation
│   │   ├── preprocessing/   # Data processing
│   │   ├── dataset/         # Dataset generation
│   │   └── networks/        # ML models
│   ├── model/               # LLM training & inference
│   │   ├── finetuning/      # SFT & DPO training
│   │   ├── evaluation/      # Model evaluation
│   │   └── inference/       # Deployment
│   └── infrastructure/      # External integrations
│       ├── db/              # MongoDB & Qdrant
│       └── aws/             # SageMaker deployment
│
├── pipelines/               # ZenML ML pipelines
├── steps/                   # Reusable pipeline steps
├── tools/                   # Utility scripts
│   ├── run.py              # Pipeline executor
│   ├── rag.py              # RAG demo
│   └── ml_service.py       # API server
├── configs/                 # Configuration files
├── tests/                   # Test suite
├── .env                     # Environment variables (create from .env.example)
└── pyproject.toml          # Dependencies
```

### Key Files

| File | Description |
|------|-------------|
| `rag_llm_system/settings.py` | Configuration management |
| `rag_llm_system/application/rag/retriever.py` | RAG implementation |
| `rag_llm_system/model/finetuning/finetune.py` | Training logic |
| `tools/rag.py` | RAG demo script |
| `tools/ml_service.py` | FastAPI server |

---

## 🔄 Pipelines

### Available Commands

| Pipeline | Command | Description |
|----------|---------|-------------|
| **ETL** | `poetry poe run-digital-data-etl` | Collect web data |
| **Feature Engineering** | `poetry poe run-feature-engineering-pipeline` | Process & embed |
| **Instruct Dataset** | `poetry poe run-generate-instruct-datasets-pipeline` | Generate Q&A pairs |
| **Preference Dataset** | `poetry poe run-generate-preference-datasets-pipeline` | Generate DPO data |
| **Training** | `poetry poe run-training-pipeline` | Fine-tune LLM |
| **Evaluation** | `poetry poe run-evaluation-pipeline` | Evaluate model |
| **RAG Test** | `poetry poe call-rag-retrieval-module` | Test retrieval |
| **API Server** | `poetry poe run-inference-ml-service` | Start API |

### Pipeline Workflow

**Full End-to-End:**
```bash
# Data phase (local)
poetry poe run-end-to-end-data-pipeline

# Training phase (AWS SageMaker required)
poetry poe run-training-pipeline
poetry poe run-evaluation-pipeline

# Deployment phase
poetry poe deploy-inference-endpoint
poetry poe run-inference-ml-service
```

---

## ⚙️ Configuration

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
```

---

## 🚢 Deployment

### Local Development

```bash
# Start services
poetry poe local-infrastructure-up

# Run API
poetry poe run-inference-ml-service
```

### AWS SageMaker (Production)

```bash
# 1. Setup
poetry install --with aws
poetry poe create-sagemaker-role
poetry poe create-sagemaker-execution-role

# 2. Train model
poetry poe run-training-pipeline

# 3. Deploy endpoint
poetry poe deploy-inference-endpoint

# 4. Test
poetry poe test-sagemaker-endpoint

# 5. Delete when done
poetry poe delete-inference-endpoint
```

### Docker

```bash
# Build image
poetry poe build-docker-image

# Run pipeline
poetry poe run-docker-end-to-end-data-pipeline
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: MongoDB Authentication Failed**
```bash
# Check credentials in .env
DATABASE_HOST=mongodb+srv://user:pass@cluster.mongodb.net

# Verify network access in MongoDB Atlas
# Add IP: 0.0.0.0/0 (for testing)
```

**Issue: Qdrant 404 Not Found**
```bash
# Ensure URL has no :6333 port for cloud
QDRANT_CLOUD_URL=https://cluster.qdrant.io  # ✓ Correct
QDRANT_CLOUD_URL=https://cluster.qdrant.io:6333  # ✗ Wrong

# Re-run feature engineering
poetry poe run-feature-engineering-pipeline
```

**Issue: OpenAI Rate Limit**
```bash
# Reduce batch size in dataset generation
# Or wait 60 seconds between retries
```

**Issue: Pydantic Version Conflict**
```bash
poetry add "pydantic>=2.8.0,<2.9.0"
poetry install
```

### Debug Mode

```bash
# Check data status
poetry run python check_data_status.py

# View ZenML runs
open http://localhost:8237

# View logs
tail -f logs/app.log
```

### Get Help

- **Documentation**: Check `RAG_MODULE_TRACE.md` for detailed flow
- **Issues**: Open a GitHub issue
- **Original Book**: [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/)

---

## 💰 Cost Estimate

**One-time full run:**
- AWS SageMaker (training): ~$20
- OpenAI API (dataset generation): ~$3
- MongoDB Atlas (free tier): $0
- Qdrant Cloud (free tier): $0
- **Total**: ~$25

**Monthly (if deployed):**
- SageMaker endpoint: ~$100-200/month
- API calls: Variable

---

## 📊 Performance

- **RAG Latency**: 2-5 seconds (including reranking)
- **Training Time**: 2-4 hours (SFT on 8B model)
- **Throughput**: ~10 requests/second (SageMaker)

---

## 🧪 Testing

```bash
# Run all tests
poetry poe test

# Lint check
poetry poe lint-check

# Format check
poetry poe format-check
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- Based on [LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/) by Paul Iusztin and Maxime Labonne
- Built with [ZenML](https://zenml.io/), [Unsloth](https://github.com/unslothai/unsloth), and [LangChain](https://langchain.com/)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Contact

- **Author**: silsgah
- **GitHub**: [@silsgah](https://github.com/silsgah)

---

**⭐ Star this repo if you found it helpful!**

