"""
Using FAISS/ChromaDB with YOUR REAL RAG DATA

This shows how to:
1. Load real embeddings from Qdrant
2. Insert them into FAISS/ChromaDB
3. Search with real queries
4. Compare results with Qdrant

Run:
    python -m examples.vector_db_comparison.use_with_real_data
"""

from loguru import logger

from rag_llm_system.application.preprocessing.dispatchers import EmbeddingDispatcher
from rag_llm_system.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from rag_llm_system.domain.queries import Query
from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory


def load_real_data_from_qdrant(limit: int = 1000):
    """
    Load REAL embeddings from your production Qdrant database.

    This fetches actual article/post/repo embeddings that you've already created
    in your RAG pipeline.

    IMPORTANT: We pass with_vectors=True to load the actual embeddings!
    """
    logger.info(f"Loading up to {limit} real embeddings from Qdrant...")

    try:
        # Load real embedded articles from Qdrant
        # IMPORTANT: with_vectors=True loads the actual embeddings!
        articles, _ = EmbeddedArticleChunk.bulk_find(limit=limit, with_vectors=True)
        logger.info(f"✓ Loaded {len(articles)} article embeddings from Qdrant")

        # Load real embedded posts
        posts, _ = EmbeddedPostChunk.bulk_find(limit=limit, with_vectors=True)
        logger.info(f"✓ Loaded {len(posts)} post embeddings from Qdrant")

        # Load real embedded repositories
        repos, _ = EmbeddedRepositoryChunk.bulk_find(limit=limit, with_vectors=True)
        logger.info(f"✓ Loaded {len(repos)} repository embeddings from Qdrant")

        return articles, posts, repos

    except Exception as e:
        logger.warning(f"Could not load data from Qdrant: {e}")
        logger.info("Make sure:")
        logger.info("  1. Qdrant is running (docker-compose up)")
        logger.info("  2. You've run the feature engineering pipeline")
        logger.info("  3. Data exists in Qdrant collections")
        return [], [], []


def migrate_to_faiss(embedded_chunks, collection_name: str = "articles"):
    """
    Migrate real embeddings from Qdrant to FAISS.

    Use case: Offline deployment, edge computing, cost optimization
    """
    if not embedded_chunks:
        logger.warning("No data to migrate!")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"Migrating {len(embedded_chunks)} embeddings to FAISS")
    logger.info(f"{'='*60}\n")

    # Create FAISS store
    faiss_store = VectorStoreFactory.create("faiss", storage_path="./real_data_faiss")

    # Verify embeddings are loaded
    first_chunk = embedded_chunks[0]
    if not hasattr(first_chunk, "embedding") or first_chunk.embedding is None:
        logger.error("❌ Embeddings not loaded! Make sure to call bulk_find(with_vectors=True)")
        return None

    # Create collection
    vector_size = len(embedded_chunks[0].embedding)
    faiss_store.create_collection(collection_name, vector_size=vector_size)
    logger.info(f"✓ Created FAISS collection '{collection_name}' ({vector_size}D vectors)")

    # Extract vectors and metadata
    vectors = [chunk.embedding for chunk in embedded_chunks]
    metadata = []

    for chunk in embedded_chunks:
        meta = {
            "id": str(chunk.id),
            "author_id": str(chunk.author_id) if hasattr(chunk, "author_id") else None,
            "content": chunk.content[:500] if hasattr(chunk, "content") else "",
            "chunk_type": chunk.__class__.__name__,
        }
        metadata.append(meta)

    # Insert into FAISS
    logger.info(f"Inserting {len(vectors)} vectors into FAISS...")
    faiss_store.insert(collection_name, vectors=vectors, metadata=metadata)

    # Save to disk (for persistence)
    logger.info("Saving FAISS index to disk...")
    faiss_store.save_collection(collection_name)

    stats = faiss_store.get_collection_stats(collection_name)
    logger.info(f"✓ FAISS collection stats: {stats}")
    logger.info(f"✓ Data saved to: ./real_data_faiss/{collection_name}.index\n")

    return faiss_store


def search_real_query(store, collection_name: str, query_text: str):
    """
    Search with a REAL query using your actual embedding model.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Searching for: '{query_text}'")
    logger.info(f"{'='*60}\n")

    # Create query using your existing pipeline
    query = Query.from_str(query_text)

    # Embed query using your actual embedding model (same as RAG pipeline!)
    embedded_query = EmbeddingDispatcher.dispatch(query)

    logger.info(f"✓ Query embedded: {len(embedded_query.embedding)}D vector")

    # Search
    results = store.search(collection_name, query_vector=embedded_query.embedding, top_k=5)

    logger.info(f"\nFound {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        logger.info(f"{i}. Score: {result.score:.4f}")
        logger.info(f"   Content: {result.metadata.get('content', 'N/A')[:200]}...")
        logger.info(f"   Author ID: {result.metadata.get('author_id', 'N/A')}")
        logger.info(f"   Type: {result.metadata.get('chunk_type', 'N/A')}\n")

    return results


def compare_qdrant_vs_faiss(query_text: str):
    """
    Compare search results between Qdrant (production) and FAISS.
    """
    logger.info(f"\n{'='*60}")
    logger.info("COMPARING: Qdrant vs FAISS")
    logger.info(f"{'='*60}\n")

    # Create query
    query = Query.from_str(query_text)
    embedded_query = EmbeddingDispatcher.dispatch(query)

    # Search Qdrant (using existing domain model)
    logger.info("Searching Qdrant (production)...")
    qdrant_results = EmbeddedArticleChunk.search(query_vector=embedded_query.embedding, limit=5)

    # Search FAISS (load from disk first!)
    logger.info("Searching FAISS (test)...")
    faiss_store = VectorStoreFactory.create("faiss", storage_path="./real_data_faiss")

    # Load the saved index from disk
    if not faiss_store.collection_exists("articles"):
        logger.info("Loading FAISS index from disk...")
        faiss_store.load_collection("articles")

    faiss_results = faiss_store.search("articles", query_vector=embedded_query.embedding, top_k=5)

    # Compare
    logger.info("\n--- Qdrant Results ---")
    for i, result in enumerate(qdrant_results[:3], 1):
        logger.info(f"{i}. Content: {result.content[:100]}...")

    logger.info("\n--- FAISS Results ---")
    for i, result in enumerate(faiss_results[:3], 1):
        logger.info(f"{i}. Content: {result.metadata.get('content', '')[:100]}...")

    logger.info("\n🎯 Both should return similar results (same embeddings, same algorithm)!")


def main():
    logger.info("=== Using FAISS/ChromaDB with Real RAG Data ===\n")

    # ============================================
    # Step 1: Load real data from Qdrant
    # ============================================
    articles, posts, repos = load_real_data_from_qdrant(limit=100)

    if not articles and not posts and not repos:
        logger.error("\n❌ No data found in Qdrant!")
        logger.info("\nTo get real data, run:")
        logger.info("  1. poetry poe local-infrastructure-up  # Start Qdrant")
        logger.info("  2. poetry poe run-digital-data-etl     # Scrape data")
        logger.info("  3. poetry poe run-feature-engineering-pipeline  # Generate embeddings")
        logger.info("\nFor now, I'll show you the code structure with a minimal example...")

        # Create minimal example

        from rag_llm_system.application.networks.embeddings import EmbeddingModelSingleton

        logger.info("\nCreating minimal example with embedded text...")

        # Embed some real text using your embedding model
        embedding_model = EmbeddingModelSingleton()

        sample_texts = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language for data science",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing helps computers understand text",
            "Computer vision enables machines to interpret visual information",
        ]

        logger.info(f"Embedding {len(sample_texts)} sample texts...")
        embeddings = [embedding_model.model.encode(text) for text in sample_texts]

        # Create FAISS store
        faiss_store = VectorStoreFactory.create("faiss", storage_path="./example_faiss")
        faiss_store.create_collection("examples", vector_size=len(embeddings[0]))

        # Insert
        metadata = [{"text": text} for text in sample_texts]
        faiss_store.insert("examples", vectors=embeddings, metadata=metadata)

        logger.info("✓ Inserted sample embeddings into FAISS")

        # Search with real query
        query_text = "What is neural network learning?"
        query_embedding = embedding_model.model.encode(query_text)

        logger.info(f"\nSearching for: '{query_text}'")
        results = faiss_store.search("examples", query_vector=query_embedding, top_k=3)

        logger.info("\nResults (using REAL embeddings, not random!):")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. Score: {result.score:.4f}")
            logger.info(f"   Text: {result.metadata.get('text')}\n")

        logger.info("🎯 Notice: Results are semantically relevant!")
        logger.info("   'neural network learning' → found 'deep learning' and 'machine learning'")
        logger.info("\nThis is the REAL use case - semantic search with meaningful embeddings!")

        return

    # ============================================
    # Step 2: Migrate to FAISS
    # ============================================
    if articles:
        logger.info("\nMigrating articles to FAISS...")
        faiss_store = migrate_to_faiss(articles, collection_name="articles")

        # ============================================
        # Step 3: Search with real query
        # ============================================
        logger.info("\nNow let's search with a REAL query!")
        search_real_query(faiss_store, "articles", "machine learning and AI")

        # ============================================
        # Step 4: Compare with Qdrant
        # ============================================
        compare_qdrant_vs_faiss("machine learning and AI")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("""
What we did:
1. ✅ Loaded REAL embeddings from Qdrant (your production data)
2. ✅ Migrated them to FAISS (separate copy)
3. ✅ Searched with REAL queries (using your embedding model)
4. ✅ Got MEANINGFUL results (semantically similar content)

Difference from demo:
- Demo: Random vectors → Random results (just testing API)
- Real:  Actual embeddings → Relevant results (real semantic search!)

When to use this:
- Test FAISS performance with your actual data
- Deploy RAG offline (save FAISS index, no cloud needed)
- Cost optimization (FAISS is free, cloud DBs cost money)
- Edge deployment (run RAG on mobile/IoT devices)

Your production Qdrant is unchanged - this is a separate copy for experimentation!
    """)


if __name__ == "__main__":
    main()
