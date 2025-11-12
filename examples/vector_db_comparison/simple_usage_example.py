"""
Simple usage example showing how easy it is to switch between vector databases.

This demonstrates the power of the adapter pattern - same code works with any backend!

Run:
    python -m examples.vector_db_comparison.simple_usage_example
"""

import numpy as np
from loguru import logger

from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory


def test_backend(backend_name: str):
    """Test a backend with the same code."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {backend_name.upper()}")
    logger.info(f"{'='*60}\n")

    try:
        # Create store (same interface for all!)
        if backend_name == "faiss":
            store = VectorStoreFactory.create("faiss", storage_path=f"./demo_{backend_name}")
        elif backend_name == "chroma":
            store = VectorStoreFactory.create("chroma", persist_directory=f"./demo_{backend_name}")
        else:
            logger.warning(f"Backend {backend_name} not configured in this demo")
            return

        # Same code for all backends from here on!
        collection_name = "test_collection"

        # Create collection
        logger.info("Creating collection...")
        store.create_collection(collection_name, vector_size=384)

        # Insert some vectors
        logger.info("Inserting vectors...")
        vectors = [np.random.randn(384).astype("float32") for _ in range(100)]
        metadata = [{"id": i, "category": f"cat_{i%5}"} for i in range(100)]
        store.insert(collection_name, vectors=vectors, metadata=metadata)

        # Search
        logger.info("Searching...")
        query = np.random.randn(384).astype("float32")
        results = store.search(collection_name, query_vector=query, top_k=5)

        logger.info("Results:")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. Score: {result.score:.4f}, Category: {result.metadata.get('category')}")

        # Get stats
        stats = store.get_collection_stats(collection_name)
        logger.info(f"\nStats: {stats}")

        # Clean up
        store.delete_collection(collection_name)
        logger.info(f"✅ {backend_name.upper()} test completed!")

    except ImportError as e:
        logger.warning(f"❌ {backend_name} not available: {e}")
        logger.info("   Install with: poetry install --with vector-db-extra")
    except Exception as e:
        logger.error(f"❌ Error testing {backend_name}: {e}")


def main():
    logger.info("=== Vector Database Adapter Demo ===")
    logger.info("Same code, different backends!")

    # Check what's available
    available = VectorStoreFactory.list_available_backends()
    logger.info(f"\nAvailable backends: {[k for k, v in available.items() if v]}")

    # Test each backend with THE SAME CODE
    test_backend("faiss")
    test_backend("chroma")

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete! Notice how the same code worked for both backends.")
    logger.info("This is the power of the adapter pattern!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
