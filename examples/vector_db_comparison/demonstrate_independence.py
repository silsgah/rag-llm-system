"""
Demonstrate that FAISS, ChromaDB, and Qdrant are INDEPENDENT.

This script shows:
1. Each backend stores its own data
2. They don't share or fetch from each other
3. You manually insert data into each one

Run:
    python -m examples.vector_db_comparison.demonstrate_independence
"""

import numpy as np
from loguru import logger

from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory


def main():
    logger.info("=== Demonstrating Vector Store Independence ===\n")

    # Generate sample data
    vector_size = 384
    vectors_a = [np.random.randn(vector_size).astype("float32") for _ in range(5)]
    vectors_b = [np.random.randn(vector_size).astype("float32") for _ in range(5)]

    metadata_a = [{"group": "A", "id": i} for i in range(5)]
    metadata_b = [{"group": "B", "id": i} for i in range(5)]

    # ============================================
    # Test 1: Insert different data into each backend
    # ============================================
    logger.info("TEST 1: Inserting DIFFERENT data into each backend\n")

    # Create FAISS store
    faiss_store = VectorStoreFactory.create("faiss", storage_path="./demo_independence_faiss")
    faiss_store.create_collection("test", vector_size=vector_size)
    faiss_store.insert("test", vectors=vectors_a, metadata=metadata_a)
    logger.info("✓ Inserted Group A data into FAISS")

    # Create ChromaDB store
    chroma_store = VectorStoreFactory.create("chroma", persist_directory="./demo_independence_chroma")
    chroma_store.create_collection("test", vector_size=vector_size)
    chroma_store.insert("test", vectors=vectors_b, metadata=metadata_b)
    logger.info("✓ Inserted Group B data into ChromaDB")

    # Check: They should have different data!
    faiss_stats = faiss_store.get_collection_stats("test")
    chroma_stats = chroma_store.get_collection_stats("test")

    logger.info(f"\nFAISS collection: {faiss_stats}")
    logger.info(f"ChromaDB collection: {chroma_stats}")

    # Search both
    query = np.random.randn(vector_size).astype("float32")

    faiss_results = faiss_store.search("test", query_vector=query, top_k=3)
    chroma_results = chroma_store.search("test", query_vector=query, top_k=3)

    logger.info("\n--- FAISS Results (Group A) ---")
    for i, result in enumerate(faiss_results, 1):
        logger.info(f"{i}. Group: {result.metadata.get('group')}, ID: {result.metadata.get('id')}")

    logger.info("\n--- ChromaDB Results (Group B) ---")
    for i, result in enumerate(chroma_results, 1):
        logger.info(f"{i}. Group: {result.metadata.get('group')}, ID: {result.metadata.get('id')}")

    logger.info("\n🎯 Notice: FAISS has Group A, ChromaDB has Group B - they're independent!\n")

    # ============================================
    # Test 2: Insert same data into both
    # ============================================
    logger.info("\nTEST 2: Inserting SAME data into both backends\n")

    # Clean up previous data
    faiss_store.delete_collection("test")
    chroma_store.delete_collection("test")

    # Create fresh collections
    faiss_store.create_collection("test", vector_size=vector_size)
    chroma_store.create_collection("test", vector_size=vector_size)

    # Insert SAME data into both
    shared_vectors = [np.random.randn(vector_size).astype("float32") for _ in range(5)]
    shared_metadata = [{"shared": True, "id": i} for i in range(5)]

    faiss_store.insert("test", vectors=shared_vectors, metadata=shared_metadata)
    chroma_store.insert("test", vectors=shared_vectors, metadata=shared_metadata)

    logger.info("✓ Inserted SAME data into both FAISS and ChromaDB")

    # Both should have 5 vectors now
    logger.info(f"\nFAISS count: {faiss_store.get_collection_stats('test')['count']}")
    logger.info(f"ChromaDB count: {chroma_store.get_collection_stats('test')['count']}")

    # Search both with same query
    query = shared_vectors[0]  # Use first vector as query (should match perfectly)

    faiss_results = faiss_store.search("test", query_vector=query, top_k=1)
    chroma_results = chroma_store.search("test", query_vector=query, top_k=1)

    logger.info("\n--- FAISS Top Result ---")
    logger.info(f"Score: {faiss_results[0].score:.4f}, Metadata: {faiss_results[0].metadata}")

    logger.info("\n--- ChromaDB Top Result ---")
    logger.info(f"Score: {chroma_results[0].score:.4f}, Metadata: {chroma_results[0].metadata}")

    logger.info("\n🎯 Same data, but stored independently in each backend!\n")

    # ============================================
    # Test 3: Modify one backend, other unaffected
    # ============================================
    logger.info("\nTEST 3: Modifying one backend doesn't affect the other\n")

    # Add more data to FAISS only
    new_vectors = [np.random.randn(vector_size).astype("float32") for _ in range(3)]
    new_metadata = [{"new": True, "id": i} for i in range(3)]

    faiss_store.insert("test", vectors=new_vectors, metadata=new_metadata)
    logger.info("✓ Added 3 more vectors to FAISS")

    # Check counts
    faiss_count = faiss_store.get_collection_stats("test")["count"]
    chroma_count = chroma_store.get_collection_stats("test")["count"]

    logger.info(f"\nFAISS count: {faiss_count} (should be 8)")
    logger.info(f"ChromaDB count: {chroma_count} (should still be 5)")

    logger.info("\n🎯 ChromaDB unchanged - they're independent!\n")

    # Clean up
    faiss_store.delete_collection("test")
    chroma_store.delete_collection("test")

    # ============================================
    # Summary
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Vector Store Independence")
    logger.info("=" * 60)
    logger.info("""
Key Findings:

1. Each backend stores its own independent data
   - FAISS data in RAM + ./demo_independence_faiss/
   - ChromaDB data in ./demo_independence_chroma/
   - Qdrant data in Qdrant database (separate)

2. They don't communicate with each other
   - Inserting into FAISS doesn't affect ChromaDB
   - Searching FAISS doesn't query Qdrant
   - Completely separate systems

3. To compare backends:
   - Manually insert same data into both
   - Run same queries on both
   - Compare results/performance

4. For your RAG system:
   - Production data stays in Qdrant
   - Use FAISS/ChromaDB for testing/learning
   - Each maintains its own copy of data

Think of them like different databases:
- PostgreSQL (Qdrant) - Production
- SQLite (FAISS) - Local/embedded
- MySQL (ChromaDB) - Alternative

They're alternatives, not integrated systems!
    """)


if __name__ == "__main__":
    main()
