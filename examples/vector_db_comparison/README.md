# Vector Database Comparison Module

This module provides **modular, plug-and-play vector database adapters** that allow you to experiment with different backends (FAISS, ChromaDB, Pinecone, Qdrant) **without modifying your production code**.

## 🎯 Purpose

- **Learn** different vector databases hands-on
- **Compare** performance characteristics
- **Experiment** without touching production Qdrant code
- **Switch** backends easily when needed

## 📦 Installation

### Install Optional Dependencies

```bash
# Install FAISS (CPU version)
poetry install --with vector-db-extra

# Or manually:
pip install faiss-cpu chromadb

# For GPU support (if you have CUDA):
pip install faiss-gpu
```

## 🚀 Quick Start

### 1. Basic Usage (Any Backend)

```python
from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory
import numpy as np

# Create any backend (faiss, chroma, qdrant)
store = VectorStoreFactory.create("faiss")

# Same interface for all backends!
store.create_collection("my_collection", vector_size=384)

# Insert vectors
vectors = [np.random.randn(384) for _ in range(100)]
metadata = [{"author": f"user_{i}", "category": "article"} for i in range(100)]
store.insert("my_collection", vectors=vectors, metadata=metadata)

# Search
query = np.random.randn(384)
results = store.search("my_collection", query_vector=query, top_k=10)

for result in results:
    print(f"ID: {result.id}, Score: {result.score:.4f}")
```

### 2. Run Demo Scripts

```bash
# Demo FAISS
python -m examples.vector_db_comparison.demo_faiss

# Demo ChromaDB
python -m examples.vector_db_comparison.demo_chroma

# Run benchmark comparison
python -m examples.vector_db_comparison.benchmark_comparison
```

## 📊 Benchmark Results (Expected)

On a typical laptop (10K vectors, 384 dimensions):

| Backend   | Insert Rate   | Search Time | Filtered Search |
|-----------|---------------|-------------|-----------------|
| FAISS     | ~50K vec/s    | ~1-2ms      | ~5-10ms         |
| ChromaDB  | ~5K vec/s     | ~5-10ms     | ~10-20ms        |
| Qdrant    | ~10K vec/s    | ~5-15ms     | ~10-20ms        |

**Note**: FAISS is fastest but requires post-filtering for metadata.

## 🏗️ Architecture

```
rag_llm_system/infrastructure/vector_stores/
├── base.py              # Abstract interface (VectorDBAdapter)
├── factory.py           # Factory pattern for backend selection
├── faiss_adapter.py     # FAISS implementation
├── chroma_adapter.py    # ChromaDB implementation
└── qdrant_adapter.py    # Qdrant wrapper (uses existing code)
```

**Design Pattern**: Adapter + Factory
**Principle**: All backends implement the same `VectorDBAdapter` interface

## 🔧 Using with Your RAG System

### Option 1: Keep Qdrant for Production

Your current production code **continues to work unchanged**. Use alternatives only for:
- Development/testing
- Experimentation
- Learning

### Option 2: Integrate Alternative Backend

If you want to use FAISS or ChromaDB in your RAG pipeline:

```python
# In your retriever or preprocessing code
from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory

# Create backend from environment variable
backend = os.getenv("VECTOR_DB_BACKEND", "qdrant")  # default to qdrant
store = VectorStoreFactory.create(backend)

# Rest of your code remains the same!
store.search(collection_name, query_vector, top_k=10, filters={...})
```

### Option 3: A/B Testing

Run two backends side-by-side for comparison:

```python
# Compare Qdrant vs FAISS
qdrant = VectorStoreFactory.create("qdrant")
faiss = VectorStoreFactory.create("faiss")

# Same query on both
results_qdrant = qdrant.search("collection", query, top_k=10)
results_faiss = faiss.search("collection", query, top_k=10)

# Compare results and latency
```

## 📖 Adapter Details

### FAISS Adapter

**Strengths:**
- ⚡ Extremely fast (10-100x faster than cloud)
- 💾 Saves to disk (`.index` files)
- 🎯 Multiple algorithms (Flat, IVF, HNSW, PQ)
- 🆓 Free and open-source

**Limitations:**
- ❌ No native metadata filtering (we add it with post-filtering)
- 📝 Manual persistence (call `save_collection()`)
- 💻 RAM-bound (all data in memory)

**Best For:** Large-scale offline search, prototyping, edge deployment

**Example:**
```python
store = VectorStoreFactory.create("faiss", storage_path="./my_faiss_indices")
store.create_collection("articles", vector_size=384)
store.insert("articles", vectors=[...], metadata=[...])

# Save to disk
store.save_collection("articles")

# Load later
store.load_collection("articles")
```

---

### ChromaDB Adapter

**Strengths:**
- ✅ Simple, intuitive API
- ✅ Built-in metadata filtering (excellent support)
- ✅ Automatic persistence
- ✅ Great for LLM applications

**Limitations:**
- ⚠️ Scalability limits (~1M vectors)
- ⚠️ Single-node only
- ⚠️ Newer/less mature

**Best For:** Development, small-medium production deployments

**Example:**
```python
store = VectorStoreFactory.create("chroma", persist_directory="./my_chroma_db")
store.create_collection("articles", vector_size=384)

# Automatic persistence - no need to save manually!
store.insert("articles", vectors=[...], metadata=[...])

# Excellent filtering
results = store.search(
    "articles",
    query_vector=query,
    top_k=10,
    filters={"author_id": "123", "category": "tech"}
)
```

---

### Qdrant Adapter

**Strengths:**
- ✅ Production-ready (your current choice)
- ✅ Excellent filtering
- ✅ Cloud or self-hosted
- ✅ Good performance at scale

**Note:** This adapter wraps your existing Qdrant implementation for compatibility.

---

## 🧪 Experiment Ideas

1. **Performance Testing**
   ```bash
   python -m examples.vector_db_comparison.benchmark_comparison
   ```

2. **Filter Performance**
   - Test how well each backend handles complex filters
   - Compare FAISS post-filtering vs ChromaDB native filtering

3. **Offline Deployment**
   - Use FAISS to deploy RAG without cloud dependencies
   - Great for demos, prototypes, or edge devices

4. **Cost Analysis**
   - Calculate cost savings of FAISS vs cloud databases
   - Factor in development time vs infrastructure cost

## 🔬 Advanced Usage

### Custom FAISS Index Types

```python
from rag_llm_system.infrastructure.vector_stores.faiss_adapter import FAISSAdapter
import faiss

# Create custom FAISS index
adapter = FAISSAdapter(storage_path="./custom_faiss")

# Use HNSW for faster search
index = faiss.IndexHNSWFlat(384, 32)  # 384-dim, 32 links per node
adapter.indices["custom_collection"] = index
```

### ChromaDB Client Modes

```python
# Persistent mode (saves to disk)
store = VectorStoreFactory.create("chroma", persist_directory="./db", client_mode="persistent")

# Ephemeral mode (in-memory only, for testing)
store = VectorStoreFactory.create("chroma", client_mode="ephemeral")
```

## 🛡️ Safety

**Important**: This module does **NOT** interfere with your production code:

✅ New directory (`vector_stores/`) - doesn't modify existing files
✅ Optional dependencies - Qdrant still works without them
✅ Examples in `examples/` - separate from production code
✅ Factory pattern - easy to switch or remove

Your production RAG system continues using Qdrant as-is!

## 🤝 Contributing

Want to add Pinecone support?

1. Create `pinecone_adapter.py` implementing `VectorDBAdapter`
2. Add to factory in `factory.py`
3. Add example script `demo_pinecone.py`
4. Update dependencies in `pyproject.toml`

## 📚 Further Reading

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Vector Database Comparison](https://qdrant.tech/benchmarks/)

## ❓ FAQ

**Q: Should I switch from Qdrant?**
A: No! Qdrant is excellent for your use case. This module is for learning and experimentation.

**Q: When would I use FAISS?**
A: For offline/edge deployment, massive scale (billions of vectors), or when you need maximum speed and have simple filtering needs.

**Q: Is this production-ready?**
A: The adapters are solid, but this is primarily an educational/experimental module. Your production Qdrant setup is already production-ready.

**Q: Can I use multiple backends simultaneously?**
A: Yes! Create instances of different adapters and use them side-by-side for comparison or A/B testing.

---

**Happy experimenting! 🚀**
