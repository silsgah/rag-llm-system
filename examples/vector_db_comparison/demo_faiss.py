"""
FAISS Demo Script

This demonstrates how to use the FAISS adapter for vector search.

Run:
    python -m examples.vector_db_comparison.demo_faiss
"""

import numpy as np
from loguru import logger

from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory


def main():
    logger.info("=== FAISS Demo ===")

    # Create FAISS adapter
    store = VectorStoreFactory.create("faiss", storage_path="./demo_faiss_storage")

    # Create collection
    collection_name = "demo_collection"
    vector_size = 384  # Same as your embeddings

    logger.info(f"Creating collection '{collection_name}'...")
    store.create_collection(collection_name, vector_size=vector_size)

    # Generate some sample vectors
    logger.info("Generating sample vectors...")
    n_vectors = 1000
    vectors = [np.random.randn(vector_size).astype("float32") for _ in range(n_vectors)]

    # Add metadata
    metadata = [
        {
            "author_id": f"author_{i % 10}",
            "content": f"Sample content {i}",
            "category": "article" if i % 2 == 0 else "post",
        }
        for i in range(n_vectors)
    ]

    # Insert vectors
    logger.info(f"Inserting {n_vectors} vectors...")
    store.insert(collection_name, vectors=vectors, metadata=metadata)

    # Get stats
    stats = store.get_collection_stats(collection_name)
    logger.info(f"Collection stats: {stats}")

    # Perform search
    logger.info("Performing search...")
    query = np.random.randn(vector_size).astype("float32")

    # Search without filters
    results = store.search(collection_name, query_vector=query, top_k=5)
    logger.info("\n=== Search Results (no filters) ===")
    for i, result in enumerate(results, 1):
        logger.info(f"{i}. ID: {result.id}, Score: {result.score:.4f}, Author: {result.metadata.get('author_id')}")

    # Search with filters
    logger.info("\n=== Search Results (filtered by author) ===")
    results_filtered = store.search(collection_name, query_vector=query, top_k=5, filters={"author_id": "author_1"})
    for i, result in enumerate(results_filtered, 1):
        logger.info(f"{i}. ID: {result.id}, Score: {result.score:.4f}, Author: {result.metadata.get('author_id')}")

    # Save to disk
    logger.info("\nSaving collection to disk...")
    store.save_collection(collection_name)

    logger.info("\n✅ FAISS Demo completed!")
    logger.info("Files saved to: ./demo_faiss_storage/")


if __name__ == "__main__":
    main()
