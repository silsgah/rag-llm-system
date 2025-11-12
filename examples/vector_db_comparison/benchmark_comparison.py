"""
Benchmark Comparison Script

Compares performance of different vector database backends:
- FAISS
- ChromaDB
- Qdrant (if running)

Metrics:
- Insert time
- Search time (with and without filters)
- Memory usage

Run:
    python -m examples.vector_db_comparison.benchmark_comparison
"""

import time
from typing import List, Tuple

import numpy as np
from loguru import logger

from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory


def generate_test_data(n_vectors: int, vector_size: int) -> Tuple[List[np.ndarray], List[dict]]:
    """Generate test vectors and metadata."""
    vectors = [np.random.randn(vector_size).astype("float32") for _ in range(n_vectors)]
    metadata = [
        {"author_id": f"author_{i % 100}", "category": "article" if i % 2 == 0 else "post", "timestamp": i}
        for i in range(n_vectors)
    ]
    return vectors, metadata


def benchmark_backend(backend_name: str, n_vectors: int = 10000, vector_size: int = 384):
    """Benchmark a specific vector database backend."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {backend_name.upper()}")
    logger.info(f"{'='*60}")

    try:
        # Create store
        if backend_name == "faiss":
            store = VectorStoreFactory.create("faiss", storage_path=f"./bench_{backend_name}")
        elif backend_name == "chroma":
            store = VectorStoreFactory.create("chroma", persist_directory=f"./bench_{backend_name}")
        elif backend_name == "qdrant":
            store = VectorStoreFactory.create("qdrant")
        else:
            logger.error(f"Unknown backend: {backend_name}")
            return None

        collection_name = f"bench_collection_{backend_name}"

        # Create collection
        logger.info("Creating collection...")
        store.create_collection(collection_name, vector_size=vector_size)

        # Generate test data
        logger.info(f"Generating {n_vectors} test vectors...")
        vectors, metadata = generate_test_data(n_vectors, vector_size)

        # Benchmark: Insert
        logger.info("Benchmarking INSERT...")
        start = time.time()
        store.insert(collection_name, vectors=vectors, metadata=metadata)
        insert_time = time.time() - start
        logger.info(f"✓ Insert time: {insert_time:.2f}s ({n_vectors/insert_time:.0f} vectors/sec)")

        # Benchmark: Search (no filter)
        logger.info("Benchmarking SEARCH (no filter)...")
        query = np.random.randn(vector_size).astype("float32")
        n_searches = 100

        start = time.time()
        for _ in range(n_searches):
            _ = store.search(collection_name, query_vector=query, top_k=10)
        search_time = time.time() - start
        avg_search_time = (search_time / n_searches) * 1000  # ms
        logger.info(f"✓ Average search time: {avg_search_time:.2f}ms ({n_searches} searches)")

        # Benchmark: Search (with filter)
        logger.info("Benchmarking SEARCH (with filter)...")
        start = time.time()
        for _ in range(n_searches):
            _ = store.search(collection_name, query_vector=query, top_k=10, filters={"author_id": "author_1"})
        filtered_search_time = time.time() - start
        avg_filtered_search_time = (filtered_search_time / n_searches) * 1000  # ms
        logger.info(f"✓ Average filtered search time: {avg_filtered_search_time:.2f}ms")

        # Get stats
        stats = store.get_collection_stats(collection_name)
        logger.info(f"Collection stats: {stats}")

        # Clean up
        store.delete_collection(collection_name)
        store.close()

        return {
            "backend": backend_name,
            "n_vectors": n_vectors,
            "insert_time": insert_time,
            "insert_rate": n_vectors / insert_time,
            "avg_search_time_ms": avg_search_time,
            "avg_filtered_search_time_ms": avg_filtered_search_time,
        }

    except Exception as e:
        logger.error(f"Benchmark failed for {backend_name}: {e}")
        return None


def main():
    logger.info("=== Vector Database Benchmark ===")

    n_vectors = 10000
    vector_size = 384

    # Check available backends
    available = VectorStoreFactory.list_available_backends()
    logger.info(f"\nAvailable backends: {[k for k, v in available.items() if v]}")

    results = []

    # Benchmark FAISS
    if available.get("faiss"):
        result = benchmark_backend("faiss", n_vectors, vector_size)
        if result:
            results.append(result)
    else:
        logger.warning("FAISS not available (install: pip install faiss-cpu)")

    # Benchmark ChromaDB
    if available.get("chroma"):
        result = benchmark_backend("chroma", n_vectors, vector_size)
        if result:
            results.append(result)
    else:
        logger.warning("ChromaDB not available (install: pip install chromadb)")

    # Benchmark Qdrant (optional - requires running instance)
    # if available.get("qdrant"):
    #     result = benchmark_backend("qdrant", n_vectors, vector_size)
    #     if result:
    #         results.append(result)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)

    if results:
        logger.info(f"\n{'Backend':<15} {'Insert Rate':<20} {'Search (ms)':<20} {'Filtered Search (ms)'}")
        logger.info("-" * 80)
        for r in results:
            logger.info(
                f"{r['backend']:<15} "
                f"{r['insert_rate']:>10,.0f} vec/s    "
                f"{r['avg_search_time_ms']:>10.2f}           "
                f"{r['avg_filtered_search_time_ms']:>10.2f}"
            )

        # Find fastest
        fastest_search = min(results, key=lambda x: x["avg_search_time_ms"])
        fastest_insert = max(results, key=lambda x: x["insert_rate"])

        logger.info("\n🏆 Winners:")
        logger.info(f"  Fastest Search: {fastest_search['backend']} ({fastest_search['avg_search_time_ms']:.2f}ms)")
        logger.info(f"  Fastest Insert: {fastest_insert['backend']} ({fastest_insert['insert_rate']:.0f} vec/s)")

    else:
        logger.warning("No benchmarks completed. Install dependencies: pip install faiss-cpu chromadb")


if __name__ == "__main__":
    main()
