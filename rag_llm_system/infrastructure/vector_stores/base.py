"""
Base protocol/interface for vector database adapters.

This defines the contract that all vector DB implementations must follow,
allowing them to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SearchResult:
    """
    Unified search result format across all vector databases.

    Attributes:
        id: Unique identifier for the vector
        score: Similarity score (higher = more similar)
        vector: The embedding vector (optional)
        metadata: Additional metadata (author, content, etc.)
    """

    id: str
    score: float
    vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VectorDBAdapter(ABC):
    """
    Abstract base class for vector database adapters.

    All vector database implementations (Qdrant, FAISS, ChromaDB, Pinecone)
    must implement this interface to ensure they can be used interchangeably.
    """

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str = "cosine") -> bool:
        """
        Create a new collection/index.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (e.g., 384)
            distance_metric: Similarity metric ("cosine", "euclidean", "dot")

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def insert(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Insert vectors into a collection.

        Args:
            collection_name: Target collection
            vectors: List of embedding vectors
            ids: Optional list of IDs (auto-generated if None)
            metadata: Optional list of metadata dicts

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def search(
        self, collection_name: str, query_vector: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            collection_name: Collection to search
            query_vector: Query embedding
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"author_id": "123"})

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        pass

    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.

        Returns:
            Dict with keys like: count, vector_size, etc.
        """
        pass

    def close(self):
        """
        Close connections and cleanup resources.

        Default implementation does nothing. Override in subclasses if cleanup is needed.
        """
        return None
