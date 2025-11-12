"""
Qdrant vector database adapter.

This wraps your existing Qdrant implementation to conform to the VectorDBAdapter interface.
It allows seamless switching between Qdrant and other vector databases.

Qdrant is currently your production database - this adapter is for compatibility.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from loguru import logger
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from rag_llm_system.infrastructure.db.qdrant import connection

from .base import SearchResult, VectorDBAdapter


class QdrantAdapter(VectorDBAdapter):
    """
    Qdrant adapter wrapping your existing Qdrant implementation.

    This provides a unified interface while using your production Qdrant setup.
    """

    def __init__(self):
        """Initialize using existing Qdrant connection."""
        self.client = connection
        logger.info("Qdrant adapter initialized (using existing connection)")

    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str = "cosine") -> bool:
        """Create a new Qdrant collection."""
        try:
            # Map distance metrics
            metric_mapping = {"cosine": Distance.COSINE, "euclidean": Distance.EUCLID, "dot": Distance.DOT}

            qdrant_metric = metric_mapping.get(distance_metric, Distance.COSINE)

            # Create collection
            self.client.create_collection(
                collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=qdrant_metric)
            )

            logger.info(f"Created Qdrant collection '{collection_name}' with {vector_size}D vectors")
            return True

        except Exception as e:
            if "already exists" in str(e).lower():
                logger.warning(f"Collection '{collection_name}' already exists")
                return True
            logger.error(f"Failed to create Qdrant collection '{collection_name}': {e}")
            return False

    def insert(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Insert vectors into Qdrant collection."""
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid4()) for _ in range(len(vectors))]

            # Prepare metadata
            if metadata is None:
                metadata = [{} for _ in range(len(vectors))]

            # Create points
            points = []
            for vec_id, vector, meta in zip(ids, vectors, metadata, strict=False):
                # Convert numpy array to list
                vec_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

                points.append(PointStruct(id=vec_id, vector=vec_list, payload=meta))

            # Upsert to Qdrant
            self.client.upsert(collection_name=collection_name, points=points)

            logger.info(f"Inserted {len(vectors)} vectors into Qdrant collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to insert vectors into Qdrant '{collection_name}': {e}")
            return False

    def search(
        self, collection_name: str, query_vector: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors in Qdrant.

        Qdrant has excellent filtering support.
        """
        try:
            # Prepare query vector
            query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

            # Build filter (Qdrant format)
            query_filter = None
            if filters:
                # Convert simple dict filters to Qdrant Filter format
                must_conditions = []
                for key, value in filters.items():
                    must_conditions.append(FieldCondition(key=key, match=MatchValue(value=str(value))))
                query_filter = Filter(must=must_conditions)

            # Search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vec,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_results.append(
                    SearchResult(id=str(result.id), score=result.score, metadata=result.payload or {})
                )

            logger.info(f"Found {len(search_results)} results in Qdrant collection '{collection_name}'")
            return search_results

        except Exception as e:
            logger.error(f"Search failed in Qdrant collection '{collection_name}': {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a Qdrant collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted Qdrant collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete Qdrant collection '{collection_name}': {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a Qdrant collection."""
        try:
            info = self.client.get_collection(collection_name=collection_name)

            return {
                "name": collection_name,
                "count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.name,
            }

        except Exception as e:
            logger.error(f"Failed to get stats for Qdrant collection '{collection_name}': {e}")
            return {}

    def close(self):
        """Close Qdrant connection."""
        # Connection is managed globally, don't close it
        logger.info("Qdrant adapter closed (connection remains active)")
