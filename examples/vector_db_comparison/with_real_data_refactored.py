"""
Simple, working streaming migration from Qdrant to FAISS

No fancy estimation, no complex error handling - just works.
"""

import gc
from typing import Iterator, List
from dataclasses import dataclass

from loguru import logger
from tqdm import tqdm


@dataclass
class MigrationConfig:
    """Simple configuration"""

    batch_size: int = 1000
    checkpoint_frequency: int = 10_000
    target_storage_path: str = "./real_data_faiss"


def stream_from_qdrant(chunk_type: type, batch_size: int = 1000) -> Iterator[List]:
    """
    Stream embeddings from Qdrant in batches.

    Keeps yielding until no more data.
    """
    offset = 0
    batch_count = 0

    logger.info(f"Starting stream from {chunk_type.__name__}")

    while True:
        # Fetch batch with vectors
        batch, next_offset = chunk_type.bulk_find(limit=batch_size, offset=offset, with_vectors=True)

        if not batch:
            logger.info(f"Stream complete: {batch_count} batches processed")
            break

        batch_count += 1
        logger.debug(f"Batch {batch_count}: {len(batch)} items")

        yield batch

        # Check if we've exhausted the collection
        if next_offset is None or next_offset == offset:
            logger.info("Reached end of collection")
            break

        offset = next_offset


def migrate_to_faiss_simple(
    chunk_type: type,
    collection_name: str,
    config: MigrationConfig,
):
    """
    Simple streaming migration - no estimation, just works.
    """
    from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory

    logger.info(f"{'='*60}")
    logger.info(f"Migrating {chunk_type.__name__} → FAISS")
    logger.info(f"{'='*60}")

    # Create FAISS store
    faiss_store = VectorStoreFactory.create("faiss", storage_path=config.target_storage_path)

    collection_initialized = False
    total_inserted = 0

    # Progress bar without total (shows items/sec instead)
    pbar = tqdm(desc=f"Migrating {collection_name}", unit="vectors")

    try:
        for batch in stream_from_qdrant(chunk_type, config.batch_size):
            # Initialize collection from first batch
            if not collection_initialized:
                if not batch or not hasattr(batch[0], "embedding"):
                    raise ValueError("No embeddings found!")

                vector_size = len(batch[0].embedding)
                faiss_store.create_collection(collection_name, vector_size=vector_size)
                logger.info(f"Created collection '{collection_name}' ({vector_size}D)")
                collection_initialized = True

            # Extract vectors and metadata
            vectors = [chunk.embedding for chunk in batch]
            metadata = [
                {
                    "id": str(chunk.id),
                    "content": getattr(chunk, "content", "")[:500],
                }
                for chunk in batch
            ]

            # Insert into FAISS
            faiss_store.insert(collection_name, vectors=vectors, metadata=metadata)
            total_inserted += len(batch)
            pbar.update(len(batch))

            # Checkpoint
            if total_inserted % config.checkpoint_frequency == 0:
                logger.info(f"Checkpoint at {total_inserted:,} vectors")
                faiss_store.save_collection(collection_name)

            # Cleanup
            del batch, vectors, metadata
            gc.collect()

        pbar.close()

        # Final save
        logger.info("Saving final index...")
        faiss_store.save_collection(collection_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Success! Migrated {total_inserted:,} vectors")
        logger.info(f"  Saved to: {config.target_storage_path}/{collection_name}.index")
        logger.info(f"{'='*60}\n")

        return faiss_store, total_inserted

    except Exception as e:
        pbar.close()
        logger.error(f"Migration failed: {e}")
        raise


def main():
    """Simple example"""
    from rag_llm_system.domain.embedded_chunks import (
        EmbeddedArticleChunk,
        EmbeddedPostChunk,
        EmbeddedRepositoryChunk,
    )

    config = MigrationConfig(
        batch_size=1000,
        checkpoint_frequency=10_000,
        target_storage_path="./faiss_data",
    )

    logger.info("=== Simple Streaming Migration ===\n")

    # Migrate each collection
    collections = [
        (EmbeddedArticleChunk, "articles"),
        (EmbeddedPostChunk, "posts"),
        (EmbeddedRepositoryChunk, "repositories"),
    ]

    results = {}

    for chunk_type, collection_name in collections:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Processing: {collection_name}")
        logger.info(f"{'#'*60}\n")

        try:
            store, count = migrate_to_faiss_simple(
                chunk_type=chunk_type,
                collection_name=collection_name,
                config=config,
            )
            results[collection_name] = {"success": True, "count": count}

        except Exception as e:
            logger.error(f"Failed: {e}")
            results[collection_name] = {"success": False, "error": str(e)}

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name, result in results.items():
        if result["success"]:
            logger.info(f"✓ {name}: {result['count']:,} vectors")
        else:
            logger.info(f"✗ {name}: {result['error']}")


if __name__ == "__main__":
    main()
