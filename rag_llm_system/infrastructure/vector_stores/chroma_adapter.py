"""
ChromaDB vector database adapter.

ChromaDB is a modern, Python-native vector database designed for LLM applications.

Pros:
    - Simple, intuitive API
    - Built-in metadata filtering
    - Automatic persistence
    - Multiple deployment modes (embedded, client-server)
    - Great for development and small-medium scale

Cons:
    - Scalability limits (~1M vectors)
    - Single-node (no distributed mode)
    - Newer/less mature than alternatives

Best for: Development, prototyping, small to medium production deployments
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from loguru import logger

from .base import SearchResult, VectorDBAdapter


class ChromaAdapter(VectorDBAdapter):
    """
    ChromaDB adapter implementing the VectorDBAdapter interface.

    ChromaDB has excellent built-in support for metadata, filtering, and persistence.
    """

    def __init__(self, persist_directory: Optional[str] = None, client_mode: str = "persistent"):
        """
        Initialize ChromaDB adapter.

        Args:
            persist_directory: Directory for persistence (default: ./chroma_storage)
            client_mode: "persistent" (with storage) or "ephemeral" (in-memory only)
        """
        try:
            import chromadb

            self.chromadb = chromadb
        except ImportError as e:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb") from e

        self.persist_directory = persist_directory or "./chroma_storage"
        self.client_mode = client_mode

        # Initialize client
        if client_mode == "persistent":
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"ChromaDB client initialized with persistence at {self.persist_directory}")
        elif client_mode == "ephemeral":
            self.client = chromadb.EphemeralClient()
            logger.info("ChromaDB client initialized in ephemeral mode (no persistence)")
        else:
            raise ValueError(f"Invalid client_mode: {client_mode}. Use 'persistent' or 'ephemeral'")

    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str = "cosine") -> bool:
        """Create a new ChromaDB collection."""
        try:
            # Map distance metrics to ChromaDB format
            metric_mapping = {
                "cosine": "cosine",
                "euclidean": "l2",
                "dot": "ip",  # inner product
            }

            chroma_metric = metric_mapping.get(distance_metric, "cosine")

            # Create or get collection
            self.client.create_collection(name=collection_name, metadata={"hnsw:space": chroma_metric})

            logger.info(f"Created ChromaDB collection '{collection_name}' with {chroma_metric} metric")
            return True

        except Exception as e:
            # Collection might already exist
            if "already exists" in str(e).lower():
                logger.warning(f"Collection '{collection_name}' already exists")
                return True
            logger.error(f"Failed to create ChromaDB collection '{collection_name}': {e}")
            return False

    def insert(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Insert vectors into ChromaDB collection."""
        try:
            collection = self.client.get_collection(name=collection_name)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid4()) for _ in range(len(vectors))]

            # Convert numpy arrays to lists
            embeddings = [vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in vectors]

            # Prepare metadata (ChromaDB requires metadata, use empty dict if None)
            if metadata is None:
                metadata = [{} for _ in range(len(vectors))]

            # ChromaDB requires 'documents' parameter (can be empty strings)
            documents = [f"doc_{i}" for i in range(len(vectors))]

            # Add to collection
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadata, documents=documents)

            logger.info(f"Inserted {len(vectors)} vectors into ChromaDB collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to insert vectors into '{collection_name}': {e}")
            return False

    def search(
        self, collection_name: str, query_vector: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors in ChromaDB.

        ChromaDB has excellent built-in filtering support.
        """
        try:
            collection = self.client.get_collection(name=collection_name)

            # Prepare query
            query_embedding = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

            # Build where clause for filtering
            where_clause = None
            if filters:
                # Convert filters to ChromaDB format
                # Simple equality filters: {"author_id": "123"} -> {"author_id": {"$eq": "123"}}
                where_clause = {key: {"$eq": value} for key, value in filters.items()}

            # Query collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances", "embeddings"],
            )

            # Convert to SearchResult objects
            search_results = []
            if results and results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    # ChromaDB returns distance (lower = more similar)
                    # Convert to score (higher = more similar)
                    distance = results["distances"][0][i] if results["distances"] else 0
                    score = 1.0 / (1.0 + distance)  # Convert distance to similarity score

                    search_results.append(
                        SearchResult(
                            id=results["ids"][0][i],
                            score=score,
                            metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                            vector=np.array(results["embeddings"][0][i]) if results["embeddings"] else None,
                        )
                    )

            logger.info(f"Found {len(search_results)} results in ChromaDB collection '{collection_name}'")
            return search_results

        except Exception as e:
            logger.error(f"Search failed in ChromaDB collection '{collection_name}': {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a ChromaDB collection."""
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted ChromaDB collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.get_collection(name=collection_name)
            return True
        except Exception:
            return False

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a ChromaDB collection."""
        try:
            collection = self.client.get_collection(name=collection_name)

            # Get count
            count = collection.count()

            # Get metadata
            metadata = collection.metadata or {}

            return {
                "name": collection_name,
                "count": count,
                "metadata": metadata,
                "distance_metric": metadata.get("hnsw:space", "cosine"),
            }

        except Exception as e:
            logger.error(f"Failed to get stats for '{collection_name}': {e}")
            return {}

    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def close(self):
        """Close ChromaDB client."""
        # ChromaDB handles persistence automatically
        logger.info("ChromaDB adapter closed")
