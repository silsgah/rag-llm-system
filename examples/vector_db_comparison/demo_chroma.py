"""
ChromaDB Demo Script

This demonstrates how to use the ChromaDB adapter for vector search.

Run:
    python -m examples.vector_db_comparison.demo_chroma
"""

import numpy as np
from loguru import logger

from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory


def main():
    logger.info("=== ChromaDB Demo ===")

    # Create ChromaDB adapter
    store = VectorStoreFactory.create("chroma", persist_directory="./demo_chroma_storage")

    # Create collection
    collection_name = "demo_collection"
    vector_size = 384

    logger.info(f"Creating collection '{collection_name}'...")
    store.create_collection(collection_name, vector_size=vector_size)

    # Generate sample vectors
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

    # Search with filters (ChromaDB has excellent filtering!)
    logger.info("\n=== Search Results (filtered by author) ===")
    results_filtered = store.search(collection_name, query_vector=query, top_k=5, filters={"author_id": "author_1"})
    for i, result in enumerate(results_filtered, 1):
        logger.info(f"{i}. ID: {result.id}, Score: {result.score:.4f}, Author: {result.metadata.get('author_id')}")

    # List all collections
    collections = store.list_collections()
    logger.info(f"\nAll collections: {collections}")

    logger.info("\n✅ ChromaDB Demo completed!")
    logger.info("Data persisted to: ./demo_chroma_storage/")


if __name__ == "__main__":
    main()
