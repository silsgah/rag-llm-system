# RAG Module Execution Trace
## Command: `poetry poe call-rag-retrieval-module`

---

## Execution Flow Diagram

```
poetry poe call-rag-retrieval-module
    ↓
poetry run python -m tools.rag
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ ENTRY POINT: tools/rag.py                                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─→ [1] configure_opik()
    │      ↓
    │   llm_engineering/infrastructure/opik_utils.py
    │      • Loads settings (COMET_API_KEY, COMET_PROJECT)
    │      • Configures Opik tracking for prompt monitoring
    │      • Sets environment variables
    │
    ├─→ [2] ContextRetriever(mock=False)
    │      ↓
    │   llm_engineering/application/rag/retriever.py
    │      • Initializes 3 sub-components:
    │         - QueryExpansion (query_expanison.py)
    │         - SelfQuery (self_query.py)
    │         - Reranker (reranking.py)
    │
    └─→ [3] retriever.search(query, k=9)
           ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ RAG SEARCH PIPELINE                                          │
    └─────────────────────────────────────────────────────────────┘
```

---

## Detailed Module Breakdown

### 1️⃣ **Query Parsing** (Line 35)
**File:** `llm_engineering/domain/queries.py`
```python
query_model = Query.from_str(query)
```

**Dependencies:**
- `llm_engineering/domain/base/vector.py` (VectorBaseDocument)
- `llm_engineering/domain/types.py` (DataCategory)

**What it does:**
- Creates a Query object from the input string
- Sets up basic query structure

---

### 2️⃣ **Metadata Extraction (Self-Query)** (Line 37)
**File:** `llm_engineering/application/rag/self_query.py`
```python
query_model = self._metadata_extractor.generate(query_model)
```

**Class:** `SelfQuery`

**Flow:**
```
SelfQueryTemplate (prompt_templates.py)
    ↓
ChatOpenAI (langchain_openai) → OpenAI API
    ↓
Extract author name from query
    ↓
split_user_full_name() (application/utils/split_user_full_name.py)
    ↓
UserDocument.get_or_create() → MongoDB Query
    ↓
llm_engineering/domain/documents.py (UserDocument)
    ↓
llm_engineering/domain/base/nosql.py (NoSQLBaseDocument)
    ↓
llm_engineering/infrastructure/db/mongo.py (MongoDB connection)
```

**Dependencies:**
- `langchain_openai.ChatOpenAI`
- `llm_engineering/domain/documents.py` (UserDocument)
- `llm_engineering/infrastructure/db/mongo.py` (MongoDB client)
- `llm_engineering/settings.py` (OPENAI_MODEL_ID, OPENAI_API_KEY)

**What it does:**
- Uses GPT to extract author name from query ("Paul Iusztin")
- Queries MongoDB `users` collection
- Attaches `author_id` and `author_full_name` to query

---

### 3️⃣ **Query Expansion** (Line 42)
**File:** `llm_engineering/application/rag/query_expanison.py`
```python
n_generated_queries = self._query_expander.generate(query_model, expand_to_n=3)
```

**Class:** `QueryExpansion`

**Flow:**
```
QueryExpansionTemplate (prompt_templates.py)
    ↓
ChatOpenAI (langchain_openai) → OpenAI API
    ↓
Generate 2 additional query variations
    ↓
Return [original_query, variation1, variation2]
```

**Dependencies:**
- `langchain_openai.ChatOpenAI`
- `llm_engineering/settings.py`

**What it does:**
- Takes 1 query → generates 3 variations
- Improves recall by searching with multiple phrasings

---

### 4️⃣ **Parallel Vector Search** (Lines 47-52)
**File:** `llm_engineering/application/rag/retriever.py`
```python
with concurrent.futures.ThreadPoolExecutor() as executor:
    search_tasks = [executor.submit(self._search, _query_model, k) for _query_model in n_generated_queries]
    n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
```

**Method:** `_search()` (Line 63-97)

**For each expanded query:**

#### 4a. **Embed Query** (Line 89)
**File:** `llm_engineering/application/preprocessing/dispatchers.py`
```python
embedded_query = EmbeddingDispatcher.dispatch(query)
```

**Flow:**
```
EmbeddingDispatcher
    ↓
QueryEmbeddingHandler (embedding_data_handlers.py)
    ↓
EmbeddingModelSingleton (application/networks/embeddings.py)
    ↓
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ↓
Returns 384-dimensional embedding vector
```

**Dependencies:**
- `sentence_transformers.SentenceTransformer`
- `llm_engineering/application/networks/embeddings.py`
- `llm_engineering/settings.py` (TEXT_EMBEDDING_MODEL_ID)

---

#### 4b. **Search Qdrant Collections** (Lines 91-93)

**Three parallel searches in Qdrant:**

##### **Search 1: Posts**
```python
post_chunks = _search_data_category(EmbeddedPostChunk, embedded_query)
```
**File:** `llm_engineering/domain/embedded_chunks.py`
**Collection:** `embedded_posts`

##### **Search 2: Articles**
```python
articles_chunks = _search_data_category(EmbeddedArticleChunk, embedded_query)
```
**Collection:** `embedded_articles`

##### **Search 3: Repositories**
```python
repositories_chunks = _search_data_category(EmbeddedRepositoryChunk, embedded_query)
```
**Collection:** `embedded_repositories`

**Flow for each search:**
```
EmbeddedChunk.search()
    ↓
llm_engineering/domain/base/vector.py (VectorBaseDocument)
    ↓
llm_engineering/infrastructure/db/qdrant.py (QdrantDatabaseConnector)
    ↓
qdrant_client.search()
    ↓
Qdrant Cloud API (QDRANT_CLOUD_URL)
    ↓
Returns top k//3 similar vectors per collection
```

**Dependencies:**
- `qdrant_client`
- `llm_engineering/infrastructure/db/qdrant.py`
- `llm_engineering/domain/base/vector.py`
- `llm_engineering/settings.py` (QDRANT_CLOUD_URL, QDRANT_APIKEY)

**Filtering (Lines 69-81):**
If `author_id` is found, applies Qdrant filter:
```python
Filter(must=[FieldCondition(key="author_id", match=MatchValue(value=author_id))])
```

---

### 5️⃣ **Aggregate & Deduplicate** (Lines 50-52)
**File:** `llm_engineering/application/rag/retriever.py`
```python
n_k_documents = utils.misc.flatten(n_k_documents)  # Flatten list of lists
n_k_documents = list(set(n_k_documents))           # Remove duplicates
```

**Dependencies:**
- `llm_engineering/application/utils/misc.py`

**What it does:**
- Combines results from 3 queries × 3 collections = up to 9 result sets
- Removes duplicate documents

---

### 6️⃣ **Reranking** (Line 57)
**File:** `llm_engineering/application/rag/reranking.py`
```python
k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
```

**Class:** `Reranker`

**Flow:**
```
CrossEncoderModelSingleton
    ↓
llm_engineering/application/networks/embeddings.py
    ↓
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2")
    ↓
Score each (query, document) pair
    ↓
Sort by score (descending)
    ↓
Return top k=9 documents
```

**Dependencies:**
- `sentence_transformers.cross_encoder.CrossEncoder`
- `llm_engineering/application/networks/embeddings.py`
- `llm_engineering/settings.py` (RERANKING_CROSS_ENCODER_MODEL_ID, RAG_MODEL_DEVICE)

**What it does:**
- More accurate relevance scoring than vector similarity
- Cross-encoder computes query-document interaction
- Selects final top-k documents

---

### 7️⃣ **Return Results** (Lines 23-25)
**File:** `tools/rag.py`
```python
logger.info("Retrieved documents:")
for rank, document in enumerate(documents):
    logger.info(f"{rank + 1}: {document}")
```

**What it does:**
- Logs all retrieved documents with their rank
- Each document is an `EmbeddedChunk` instance containing:
  - `content`: Text content
  - `author_full_name`: Author
  - `platform`: Source platform
  - `metadata`: Additional info

---

## Complete File Dependency Tree

```
tools/rag.py
├── llm_engineering/infrastructure/opik_utils.py
│   ├── llm_engineering/settings.py
│   └── opik (external package)
│
├── llm_engineering/application/rag/retriever.py
│   ├── llm_engineering/application/rag/self_query.py
│   │   ├── llm_engineering/application/rag/prompt_templates.py
│   │   ├── llm_engineering/application/utils/split_user_full_name.py
│   │   ├── llm_engineering/domain/documents.py
│   │   │   └── llm_engineering/domain/base/nosql.py
│   │   │       └── llm_engineering/infrastructure/db/mongo.py
│   │   └── langchain_openai (external)
│   │
│   ├── llm_engineering/application/rag/query_expanison.py
│   │   ├── llm_engineering/application/rag/prompt_templates.py
│   │   └── langchain_openai (external)
│   │
│   ├── llm_engineering/application/rag/reranking.py
│   │   └── llm_engineering/application/networks/embeddings.py
│   │       └── sentence_transformers (external)
│   │
│   ├── llm_engineering/application/preprocessing/dispatchers.py
│   │   └── llm_engineering/application/preprocessing/embedding_data_handlers.py
│   │       └── llm_engineering/application/networks/embeddings.py
│   │
│   ├── llm_engineering/domain/queries.py
│   │   └── llm_engineering/domain/base/vector.py
│   │
│   ├── llm_engineering/domain/embedded_chunks.py
│   │   └── llm_engineering/domain/base/vector.py
│   │       └── llm_engineering/infrastructure/db/qdrant.py
│   │
│   └── llm_engineering/application/utils/misc.py
│
└── loguru (external)
```

---

## External Services Called

### 1. **OpenAI API** (2 calls per search)
- **Endpoint:** `https://api.openai.com/v1/chat/completions`
- **Used by:**
  - `SelfQuery` (metadata extraction)
  - `QueryExpansion` (query expansion)
- **Model:** `gpt-4o-mini` (from settings)

### 2. **MongoDB Atlas**
- **URI:** From `DATABASE_HOST` env var
- **Database:** `twin`
- **Collections:**
  - `users` (read by SelfQuery.generate)
- **Used by:** `UserDocument.get_or_create()`

### 3. **Qdrant Cloud**
- **URL:** From `QDRANT_CLOUD_URL` env var
- **Collections queried:**
  - `embedded_posts`
  - `embedded_articles`
  - `embedded_repositories`
- **Operation:** Vector similarity search
- **Used by:** `EmbeddedChunk.search()`

### 4. **Comet ML/Opik**
- **Endpoint:** `https://www.comet.com/opik/api/v1/private/`
- **Used for:** Prompt tracking and monitoring
- **Tracks:**
  - Query inputs
  - LLM responses
  - Retrieval results

---

## Machine Learning Models Loaded

### 1. **Sentence Transformer (Embeddings)**
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose:** Convert text to 384-dim vectors
- **File:** `llm_engineering/application/networks/embeddings.py`
- **Class:** `EmbeddingModelSingleton`

### 2. **Cross-Encoder (Reranking)**
- **Model:** `cross-encoder/ms-marco-MiniLM-L-4-v2`
- **Purpose:** Score query-document relevance
- **File:** `llm_engineering/application/networks/embeddings.py`
- **Class:** `CrossEncoderModelSingleton`

---

## Configuration Files Used

1. **`.env`** - Environment variables:
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL_ID`
   - `DATABASE_HOST` (MongoDB)
   - `QDRANT_CLOUD_URL`
   - `QDRANT_APIKEY`
   - `COMET_API_KEY`
   - `COMET_PROJECT`
   - `TEXT_EMBEDDING_MODEL_ID`
   - `RERANKING_CROSS_ENCODER_MODEL_ID`
   - `RAG_MODEL_DEVICE`

2. **`pyproject.toml`** - Task definition:
   ```toml
   call-rag-retrieval-module = "poetry run python -m tools.rag"
   ```

3. **`llm_engineering/settings.py`** - Loads and validates settings

---

## Error Points & Troubleshooting

### Common Failures:

1. **MongoDB Connection** (Line 32 in self_query.py)
   - **Error:** `OperationFailure: bad auth`
   - **File:** `llm_engineering/domain/base/nosql.py:91`
   - **Fix:** Check `DATABASE_HOST` credentials

2. **Qdrant 404** (Line 91-93 in retriever.py)
   - **Error:** `UnexpectedResponse: 404 Not Found`
   - **File:** `llm_engineering/domain/base/vector.py:142`
   - **Fix:** Verify `QDRANT_CLOUD_URL` and `QDRANT_APIKEY`

3. **Empty Results** (Line 54 in retriever.py)
   - **Warning:** `0 documents retrieved successfully`
   - **Cause:** Collections exist but are empty
   - **Fix:** Run `poetry poe run-feature-engineering-pipeline`

4. **OpenAI API Error**
   - **Error:** `AuthenticationError` or `RateLimitError`
   - **Files:** `self_query.py:25`, `query_expanison.py:26`
   - **Fix:** Check `OPENAI_API_KEY` validity

---

## Performance Characteristics

- **OpenAI API calls:** 2 (serial)
- **MongoDB queries:** 1 (user lookup)
- **Qdrant searches:** 3 queries × 3 collections = 9 searches (parallel)
- **Embedding operations:** 3 queries embedded
- **Reranking operations:** n documents scored

**Typical execution time:** 5-15 seconds

---

## Output Format

Returns a list of `EmbeddedChunk` objects with:
- `content`: str
- `author_full_name`: str
- `platform`: str
- `document_id`: UUID4
- `author_id`: UUID4
- `metadata`: dict
- `embedding`: list[float] (optional)

---

**Generated on:** 2025-10-04
**For:** LLM Engineers Handbook Project
