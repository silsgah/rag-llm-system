"""
Quick script to check if you have data in Qdrant.

Run:
    python -m examples.vector_db_comparison.check_qdrant_data
"""

from loguru import logger

from rag_llm_system.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)


def main():
    logger.info("=== Checking Qdrant Data ===\n")

    try:
        # Check articles (without loading vectors, just count)
        articles, _ = EmbeddedArticleChunk.bulk_find(limit=1, with_vectors=False)

        # Get collection stats
        article_stats = EmbeddedArticleChunk.get_collection_stats("embedded_articles")
        post_stats = EmbeddedPostChunk.get_collection_stats("embedded_posts")
        repo_stats = EmbeddedRepositoryChunk.get_collection_stats("embedded_repositories")

        logger.info("📊 Qdrant Collections:\n")
        logger.info(f"embedded_articles:      {article_stats.get('count', 0):,} vectors")
        logger.info(f"embedded_posts:         {post_stats.get('count', 0):,} vectors")
        logger.info(f"embedded_repositories:  {repo_stats.get('count', 0):,} vectors")

        total = article_stats.get("count", 0) + post_stats.get("count", 0) + repo_stats.get("count", 0)

        logger.info(f"\nTotal: {total:,} embeddings in Qdrant")

        if total > 0:
            logger.info("\n✅ You have data! The real usage script will work.")
            logger.info("   Run: python -m examples.vector_db_comparison.use_with_real_data")
        else:
            logger.info("\n⚠️  No data found in Qdrant.")
            logger.info("   The script will run a minimal example instead.")
            logger.info("\n   To populate Qdrant:")
            logger.info("   1. poetry poe local-infrastructure-up")
            logger.info("   2. poetry poe run-digital-data-etl")
            logger.info("   3. poetry poe run-feature-engineering-pipeline")

    except Exception as e:
        logger.error(f"\n❌ Could not connect to Qdrant: {e}")
        logger.info("\nMake sure Qdrant is running:")
        logger.info("  poetry poe local-infrastructure-up")
        logger.info("\nOr check docker-compose:")
        logger.info("  docker-compose up -d")


if __name__ == "__main__":
    main()
