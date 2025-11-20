# RAG System Deployment Guide

Complete guide for deploying your RAG system to production.

---

## Table of Contents

1. [Deployment Options Overview](#deployment-options-overview)
2. [Option 1: AWS Cloud (Current Setup)](#option-1-aws-cloud-recommended)
3. [Option 2: Docker + Kubernetes](#option-2-docker--kubernetes)
4. [Option 3: Serverless](#option-3-serverless)
5. [Option 4: Hybrid (Edge + Cloud)](#option-4-hybrid-edge--cloud)
6. [Cost Analysis](#cost-analysis)
7. [Scaling Strategies](#scaling-strategies)
8. [Monitoring & Observability](#monitoring--observability)

---

## Deployment Options Overview

| Option | Cost | Complexity | Scalability | Best For |
|--------|------|------------|-------------|----------|
| **AWS Cloud** | $$$ | Medium | High | Production (current) |
| **Docker + K8s** | $$ | High | Very High | Enterprise |
| **Serverless** | $ | Low | Auto | Variable traffic |
| **Hybrid** | $$ | High | Medium | Edge + Cloud |

---

## Option 1: AWS Cloud (Recommended)

**Your current setup** - Production-ready with AWS SageMaker.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS CLOUD DEPLOYMENT                      │
└─────────────────────────────────────────────────────────────┘

Internet
   │
   ↓
┌──────────────────┐
│   Load Balancer  │  AWS ALB (Application Load Balancer)
│   (ALB)          │  - Health checks
└──────────────────┘  - SSL termination
   │                  - Auto-scaling trigger
   ↓
┌──────────────────┐
│   FastAPI App    │  AWS ECS (Fargate) or EC2
│   (Container)    │  - 2-4 instances
│                  │  - Auto-scaling (CPU > 70%)
│  ┌────────────┐  │
│  │ Retriever  │  │
│  │ (RAG)      │  │
│  └────────────┘  │
└──────────────────┘
   │
   ├───────────────────┐
   │                   │
   ↓                   ↓
┌──────────────┐   ┌──────────────────┐
│  OpenAI API  │   │  Qdrant Cloud    │
│              │   │  (Vector DB)     │
│ Self-query   │   │  - 3 collections │
│ Expansion    │   │  - 384D vectors  │
└──────────────┘   └──────────────────┘

   ├───────────────────┐
   │                   │
   ↓                   ↓
┌──────────────┐   ┌──────────────────┐
│ MongoDB      │   │  AWS SageMaker   │
│ Atlas        │   │  (LLM Endpoint)  │
│              │   │                  │
│ User lookup  │   │  ml.g5.2xlarge   │
└──────────────┘   │  TwinLlama-8B    │
                   └──────────────────┘
```

### Components

#### **1. FastAPI Application (ECS/EC2)**

**Option A: ECS Fargate** (Recommended)
- Serverless containers
- No server management
- Auto-scaling built-in

**Option B: EC2**
- More control
- Lower cost at scale
- Manual scaling

**Container Specs:**
- **Image:** Your Dockerfile
- **CPU:** 2 vCPUs
- **Memory:** 4 GB RAM
- **Instances:** 2-4 (auto-scale)

**Dockerfile** (already in your project):
```dockerfile
FROM python:3.11-slim

# Install Chrome (for Selenium if needed)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable

# Install Poetry
RUN pip install poetry

# Copy project
COPY . /app
WORKDIR /app

# Install dependencies
RUN poetry install --no-dev

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["poetry", "run", "uvicorn", "tools.ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### **2. AWS SageMaker (LLM Inference)**

**Current Deployment:**
- Endpoint: `twin`
- Model: `mlabonne/TwinLlama-3.1-8B-DPO`
- Instance: `ml.g5.2xlarge` (1× A10G GPU, 24GB VRAM)

**Deployment Script** (already in project):
```bash
# Deploy to SageMaker
poetry poe deploy-inference-endpoint

# Internally runs:
# python -m rag_llm_system.infrastructure.aws.deploy_sagemaker
```

**Configuration:**
```python
# settings.py
SAGEMAKER_ENDPOINT_INFERENCE = "twin"
GPU_INSTANCE_TYPE = "ml.g5.2xlarge"
HF_MODEL_ID = "mlabonne/TwinLlama-3.1-8B-DPO"
```

**Cost:** ~$1.20/hour (~$864/month for 24/7)

**Optimization: Auto-Scaling**
- Use SageMaker Auto Scaling
- Scale 0→N instances based on traffic
- **Save 70%+ on low-traffic periods**

#### **3. Vector Database (Qdrant Cloud)**

**Current Setup:**
- **Host:** eu-west-2-0.aws.cloud.qdrant.io
- **Collections:** 3 (posts, articles, repos)
- **Vectors:** ~10K-1M (depending on data)

**Pricing:**
- **Free tier:** 1 GB storage
- **Paid:** $25-100/month (1-10M vectors)

**Alternative: Self-Hosted Qdrant**
```yaml
# docker-compose.yml (add to production)
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
```

**Saves:** $300+/year vs cloud

#### **4. MongoDB Atlas**

**Current Setup:**
- **Cluster:** Shared/Dedicated
- **Collections:** users, posts, articles, repos

**Pricing:**
- **Free tier:** 512 MB
- **Shared:** $9-25/month
- **Dedicated:** $57+/month

---

### Deployment Steps (AWS Cloud)

#### **Step 1: Prepare Docker Image**

```bash
# Build Docker image
docker build -t rag-llm-system:latest .

# Test locally
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e QDRANT_CLOUD_URL=$QDRANT_CLOUD_URL \
  -e QDRANT_APIKEY=$QDRANT_APIKEY \
  rag-llm-system:latest

# Push to AWS ECR
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-north-1.amazonaws.com

docker tag rag-llm-system:latest <account-id>.dkr.ecr.eu-north-1.amazonaws.com/rag-llm-system:latest

docker push <account-id>.dkr.ecr.eu-north-1.amazonaws.com/rag-llm-system:latest
```

#### **Step 2: Deploy SageMaker Endpoint**

```bash
# Already scripted in your project!
poetry install --with aws
poetry poe create-sagemaker-role
poetry poe deploy-inference-endpoint
```

**What it does:**
1. Uploads model to S3
2. Creates SageMaker model
3. Creates endpoint configuration
4. Deploys endpoint (takes ~10 minutes)

**Verify:**
```bash
poetry poe test-sagemaker-endpoint
```

#### **Step 3: Deploy FastAPI to ECS**

**Option A: Using AWS Console**

1. **Create ECS Cluster:**
   - Service: Fargate
   - Name: `rag-api-cluster`

2. **Create Task Definition:**
   - Image: `<ecr-url>/rag-llm-system:latest`
   - CPU: 2 vCPU
   - Memory: 4 GB
   - Environment variables:
     ```
     OPENAI_API_KEY=sk-...
     QDRANT_CLOUD_URL=https://...
     QDRANT_APIKEY=...
     AWS_ACCESS_KEY=...
     AWS_SECRET_KEY=...
     SAGEMAKER_ENDPOINT_INFERENCE=twin
     ```

3. **Create Service:**
   - Tasks: 2 (desired count)
   - Load balancer: ALB
   - Auto-scaling: CPU > 70% → scale up

**Option B: Using Terraform** (Infrastructure as Code)

```hcl
# terraform/main.tf
resource "aws_ecs_cluster" "rag_cluster" {
  name = "rag-api-cluster"
}

resource "aws_ecs_task_definition" "rag_task" {
  family                   = "rag-api"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "2048"
  memory                   = "4096"

  container_definitions = jsonencode([{
    name  = "rag-api"
    image = "${aws_ecr_repository.rag_repo.repository_url}:latest"
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    environment = [
      { name = "OPENAI_API_KEY", value = var.openai_api_key },
      { name = "SAGEMAKER_ENDPOINT_INFERENCE", value = "twin" }
    ]
  }])
}

resource "aws_ecs_service" "rag_service" {
  name            = "rag-api-service"
  cluster         = aws_ecs_cluster.rag_cluster.id
  task_definition = aws_ecs_task_definition.rag_task.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  load_balancer {
    target_group_arn = aws_lb_target_group.rag_tg.arn
    container_name   = "rag-api"
    container_port   = 8000
  }
}
```

Deploy:
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

#### **Step 4: Configure Load Balancer**

**Application Load Balancer (ALB):**
- **Listener:** HTTPS (443)
- **SSL Certificate:** AWS Certificate Manager
- **Health Check:** `GET /health`
- **Target Group:** ECS tasks

**DNS:**
- Route 53: `api.yourdomain.com` → ALB

#### **Step 5: Setup Monitoring**

**CloudWatch Logs:**
```python
# Update FastAPI app to log to CloudWatch
import watchtower
import logging

logger = logging.getLogger()
logger.addHandler(watchtower.CloudWatchLogHandler(log_group='/ecs/rag-api'))
```

**Opik Monitoring:**
- Already configured in your code!
- Dashboard: https://www.comet.com/opik/

---

### Cost Breakdown (AWS Cloud)

| Component | Instance | Cost/Month | Notes |
|-----------|----------|------------|-------|
| **FastAPI (ECS)** | 2× Fargate (2 vCPU, 4GB) | ~$60 | Auto-scales |
| **SageMaker** | ml.g5.2xlarge (24/7) | ~$864 | Consider auto-scaling |
| **Qdrant Cloud** | 1M vectors | ~$50 | Free tier: 1GB |
| **MongoDB Atlas** | Shared cluster | ~$25 | Free tier: 512MB |
| **OpenAI API** | ~10K queries | ~$30 | $0.003/query |
| **Data Transfer** | 100GB egress | ~$9 | First 10GB free |
| **Load Balancer** | ALB | ~$20 | |
| **Total** | | **~$1,058/month** | |

**Optimization Tips:**
1. **SageMaker Auto-Scaling:** Save 70% → $260/month
2. **Self-Host Qdrant:** Save $50 → $0
3. **Use Free Tiers:** Save $75 → $0
4. **Optimized:** **~$500-600/month**

---

## Option 2: Docker + Kubernetes

**For enterprise deployments with high scalability needs.**

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              KUBERNETES CLUSTER (AWS EKS)               │
└─────────────────────────────────────────────────────────┘

Ingress (NGINX)
   │
   ├──→ /rag → FastAPI Service (3 replicas)
   │             ├─→ Pod 1 (FastAPI + Retriever)
   │             ├─→ Pod 2 (FastAPI + Retriever)
   │             └─→ Pod 3 (FastAPI + Retriever)
   │
   └──→ /docs → FastAPI Docs

Internal Services:
   ├─→ Qdrant Service (StatefulSet)
   │   └─→ Persistent Volume (EBS)
   │
   └─→ MongoDB Service (StatefulSet)
       └─→ Persistent Volume (EBS)

External:
   ├─→ OpenAI API
   └─→ SageMaker Endpoint
```

### Kubernetes Manifests

#### **FastAPI Deployment**

```yaml
# k8s/fastapi-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: <account-id>.dkr.ecr.eu-north-1.amazonaws.com/rag-llm-system:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        - name: SAGEMAKER_ENDPOINT_INFERENCE
          value: "twin"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### **Horizontal Pod Autoscaler**

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### **Secrets Management**

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
type: Opaque
stringData:
  openai-api-key: sk-...
  qdrant-api-key: ...
  aws-access-key: ...
  aws-secret-key: ...
```

#### **Deploy to Kubernetes**

```bash
# Create EKS cluster
eksctl create cluster \
  --name rag-cluster \
  --region eu-north-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3

# Deploy application
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/fastapi-deployment.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl get pods
kubectl get svc
kubectl logs -f deployment/rag-api

# Get LoadBalancer URL
kubectl get svc rag-api-service
```

### Benefits

✅ **High Availability:** Pod auto-recovery
✅ **Auto-Scaling:** CPU/memory-based
✅ **Rolling Updates:** Zero-downtime deploys
✅ **Resource Limits:** Prevent resource exhaustion
✅ **Health Checks:** Auto-restart unhealthy pods

---

## Option 3: Serverless

**For variable traffic patterns.**

### AWS Lambda + API Gateway

**Architecture:**

```
API Gateway
   ↓
Lambda Function (FastAPI)
   ├─→ OpenAI API
   ├─→ Qdrant Cloud
   ├─→ MongoDB Atlas
   └─→ SageMaker Endpoint
```

**Limitations:**
- **Timeout:** 15 minutes max (Lambda)
- **Memory:** 10 GB max
- **Cold Start:** ~3-5 seconds

**When to Use:**
- Low traffic (<1000 requests/day)
- Bursty traffic patterns
- Cost optimization for development

**Cost:**
- **Free tier:** 1M requests/month
- **Paid:** $0.20 per 1M requests
- **Compute:** $0.0000166667 per GB-second

**Deployment:**
```bash
# Use AWS SAM or Serverless Framework
serverless deploy
```

---

## Option 4: Hybrid (Edge + Cloud)

**Edge RAG with cloud fallback.**

### Architecture

```
┌─────────────────┐
│   Edge Device   │  (Mobile, IoT, Laptop)
│                 │
│  ┌───────────┐  │
│  │   FAISS   │  │  Local vector search
│  │ (Offline) │  │  - 100K vectors in RAM
│  └───────────┘  │  - No internet needed
│                 │
│  ┌───────────┐  │
│  │  Llama.cpp│  │  Local LLM (quantized)
│  │   (4-bit) │  │  - 2GB model
│  └───────────┘  │  - CPU inference
└─────────────────┘
        │
        │ (Fallback for complex queries)
        ↓
┌─────────────────┐
│   Cloud API     │
│  (SageMaker)    │
└─────────────────┘
```

### Use Cases

✅ **Offline RAG:** Works without internet
✅ **Privacy:** Data never leaves device
✅ **Low Latency:** <100ms local search
✅ **Cost:** $0 after deployment

### Implementation

**Export FAISS Index:**
```python
# On server (one-time)
from rag_llm_system.domain.embedded_chunks import EmbeddedArticleChunk
from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory

# Load from Qdrant
articles, _ = EmbeddedArticleChunk.bulk_find(limit=100000, with_vectors=True)

# Export to FAISS
faiss_store = VectorStoreFactory.create("faiss", storage_path="./edge_deploy")
faiss_store.create_collection("articles", vector_size=384)

vectors = [a.embedding for a in articles]
metadata = [{"content": a.content, "author": str(a.author_id)} for a in articles]
faiss_store.insert("articles", vectors=vectors, metadata=metadata)
faiss_store.save_collection("articles")

# Result: edge_deploy/articles.index (~40MB for 100K vectors)
```

**Deploy to Edge:**
```bash
# Copy index to device
scp edge_deploy/articles.index device:/app/data/

# Run local inference
python edge_rag_server.py
```

---

## Cost Analysis

### Monthly Costs Comparison

| Component | AWS Cloud | Kubernetes | Serverless | Hybrid |
|-----------|-----------|------------|------------|--------|
| **Compute** | $60 (ECS) | $150 (EKS nodes) | $20 (Lambda) | $0 (local) |
| **LLM** | $864 (SageMaker) | $864 (SageMaker) | $864 (SageMaker) | $0 (local llama.cpp) |
| **Vector DB** | $50 (Qdrant) | $0 (self-hosted) | $50 (Qdrant) | $0 (FAISS) |
| **Database** | $25 (Atlas) | $0 (self-hosted) | $25 (Atlas) | $0 (local) |
| **OpenAI** | $30 | $30 | $30 | $0 (skip) |
| **Other** | $29 (LB, etc) | $50 (LB, EBS) | $10 (API GW) | $0 |
| **TOTAL** | **~$1,058** | **~$1,094** | **~$999** | **~$0** |

### Optimized Costs

| Optimization | Savings | New Total |
|-------------|---------|-----------|
| SageMaker Auto-Scaling (70% idle) | -$605 | $453 |
| Self-Host Qdrant | -$50 | $403 |
| Use Free Tiers | -$75 | $328 |
| Spot Instances (ECS) | -$40 | $288 |
| **Optimized Total** | | **~$300-500/month** |

---

## Scaling Strategies

### Vertical Scaling

**Increase instance size:**
```
t3.medium (2 vCPU, 4GB) → t3.large (2 vCPU, 8GB) → t3.xlarge (4 vCPU, 16GB)
```

**When:** Single bottleneck (CPU/memory)

### Horizontal Scaling

**Add more instances:**
```
1 instance → 2 instances → 5 instances → 10 instances
```

**When:** High request volume

**Auto-Scaling Triggers:**
- CPU > 70%
- Memory > 80%
- Request count > 1000/min

### Component-Specific Scaling

#### **1. FastAPI (Stateless)**
- Easy to scale horizontally
- Use load balancer
- No session state

#### **2. SageMaker (LLM)**
- Scale endpoint replicas
- Use auto-scaling
- Consider model quantization (reduce cost)

#### **3. Qdrant (Vector DB)**
- Vertical scaling (more RAM)
- Sharding for massive scale
- Read replicas

#### **4. Caching Layer**

Add Redis for frequently asked queries:

```python
import redis

cache = redis.Redis(host='localhost', port=6379)

def rag_with_cache(query):
    # Check cache
    cached = cache.get(f"rag:{query}")
    if cached:
        return json.loads(cached)

    # Compute
    result = rag(query)

    # Cache for 1 hour
    cache.setex(f"rag:{query}", 3600, json.dumps(result))
    return result
```

**Benefit:** Reduce OpenAI + SageMaker costs by 50-70%

---

## Monitoring & Observability

### Metrics to Track

#### **1. Latency**
```python
# Already tracked in Opik!
{
    "self_query_latency_ms": 500,
    "query_expansion_latency_ms": 800,
    "retrieval_latency_ms": 300,
    "reranking_latency_ms": 200,
    "generation_latency_ms": 1500,
    "total_latency_ms": 3300
}
```

#### **2. Throughput**
- Requests per second (RPS)
- Concurrent requests
- Queue depth

#### **3. Error Rates**
- 5xx errors (server)
- 4xx errors (client)
- Timeout errors
- OpenAI rate limits

#### **4. Cost**
- OpenAI API calls
- SageMaker invocations
- Qdrant queries
- Total cost per query

### Dashboards

**CloudWatch Dashboard:**
```
┌───────────────────┬───────────────────┐
│  Latency (avg)    │  Throughput       │
│  2.3s             │  45 req/min       │
└───────────────────┴───────────────────┘
┌───────────────────┬───────────────────┐
│  Error Rate       │  SageMaker Util   │
│  0.5%             │  65%              │
└───────────────────┴───────────────────┘
```

**Opik Dashboard:**
- Trace view (see entire request flow)
- Token usage
- Cost tracking
- Error debugging

### Alerts

**Critical Alerts:**
```yaml
- name: High Error Rate
  condition: error_rate > 5%
  action: PagerDuty alert

- name: High Latency
  condition: p95_latency > 10s
  action: Slack notification

- name: SageMaker Down
  condition: endpoint_unavailable
  action: PagerDuty + auto-restart

- name: Cost Spike
  condition: hourly_cost > $50
  action: Slack notification
```

---

## Quick Start: Deploy to Production

### 1-Hour Deployment (AWS)

```bash
# Step 1: Prepare (5 min)
cp .env.example .env
# Fill in API keys

# Step 2: Build & Push Image (10 min)
docker build -t rag-api .
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-url>
docker tag rag-api:latest <ecr-url>/rag-api:latest
docker push <ecr-url>/rag-api:latest

# Step 3: Deploy SageMaker (15 min)
poetry install --with aws
poetry poe deploy-inference-endpoint
# Wait for endpoint...

# Step 4: Deploy ECS (20 min)
# Use AWS Console or Terraform
# Create: Cluster → Task → Service → ALB

# Step 5: Test (5 min)
curl -X POST https://api.yourdomain.com/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain RAG systems"}'

# Step 6: Monitor (5 min)
# Check: CloudWatch Logs, Opik Dashboard
```

---

## Best Practices

### Security

✅ **Use AWS Secrets Manager** for API keys
✅ **Enable WAF** (Web Application Firewall)
✅ **Use VPC** (private networking)
✅ **SSL/TLS** (HTTPS only)
✅ **Rate limiting** (prevent abuse)
✅ **Authentication** (API keys, OAuth)

### Performance

✅ **Cache common queries** (Redis)
✅ **Batch requests** to OpenAI
✅ **Connection pooling** for Qdrant
✅ **Async operations** where possible
✅ **CDN** for static content

### Cost Optimization

✅ **Auto-scaling** (scale to 0 when idle)
✅ **Spot instances** (70% discount)
✅ **Reserved instances** (40% discount for committed usage)
✅ **Monitor costs** (set budgets, alerts)
✅ **Optimize prompts** (reduce tokens)

---

## Next Steps

1. **Choose deployment option** (AWS Cloud recommended)
2. **Set up infrastructure** (ECS + SageMaker)
3. **Configure monitoring** (CloudWatch + Opik)
4. **Test thoroughly** (load testing, stress testing)
5. **Deploy to production** 🚀

---

**Questions?** Check the README or ask your team!
