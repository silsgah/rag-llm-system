"""
FAISS vector database adapter.

FAISS is an in-memory, high-performance similarity search library.
This adapter adds metadata support by maintaining a separate metadata store.

Pros:
    - Extremely fast vector search (10-100x faster than cloud solutions)
    - Free and open-source
    - GPU acceleration support
    - Multiple indexing algorithms

Cons:
    - No built-in metadata filtering (we handle it separately)
    - Manual persistence (save/load)
    - RAM-bound (all data in memory)

Best for: Large-scale offline search, prototyping, edge deployment
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from loguru import logger

from .base import SearchResult, VectorDBAdapter


class FAISSAdapter(VectorDBAdapter):
    """
    FAISS adapter with metadata support.

    Uses a hybrid approach:
    - FAISS for fast vector search
    - In-memory dict for metadata storage
    - Optional file persistence
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize FAISS adapter.

        Args:
            storage_path: Directory to save/load indices (optional)
        """
        try:
            import faiss

            self.faiss = faiss
        except ImportError as e:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu (or faiss-gpu)") from e

        self.storage_path = Path(storage_path) if storage_path else Path("./faiss_storage")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Store indices and metadata per collection
        self.indices: Dict[str, Any] = {}  # collection_name -> faiss.Index
        self.metadata_store: Dict[str, Dict[str, Dict]] = {}  # collection -> {id -> metadata}
        self.id_mapping: Dict[str, List[str]] = {}  # collection -> [id1, id2, ...] (maintains order)
        self.vector_sizes: Dict[str, int] = {}  # collection -> vector_size

        logger.info(f"FAISS adapter initialized with storage at {self.storage_path}")

    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str = "cosine") -> bool:
        """Create a new FAISS index."""
        try:
            # Choose index type based on distance metric
            if distance_metric == "cosine":
                # For cosine similarity, normalize vectors and use inner product
                index = self.faiss.IndexFlatIP(vector_size)
            elif distance_metric == "euclidean":
                index = self.faiss.IndexFlatL2(vector_size)
            elif distance_metric == "dot":
                index = self.faiss.IndexFlatIP(vector_size)
            else:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")

            self.indices[collection_name] = index
            self.metadata_store[collection_name] = {}
            self.id_mapping[collection_name] = []
            self.vector_sizes[collection_name] = vector_size

            logger.info(f"Created FAISS collection '{collection_name}' with {vector_size}D vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to create FAISS collection '{collection_name}': {e}")
            return False

    def insert(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Insert vectors into FAISS index."""
        try:
            if collection_name not in self.indices:
                raise ValueError(f"Collection '{collection_name}' does not exist")

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid4()) for _ in range(len(vectors))]

            # Convert to numpy array
            vectors_array = np.array(vectors).astype("float32")

            # Normalize for cosine similarity (if using IndexFlatIP)
            if isinstance(self.indices[collection_name], self.faiss.IndexFlatIP):
                self.faiss.normalize_L2(vectors_array)

            # Add vectors to index
            self.indices[collection_name].add(vectors_array)

            # Store metadata
            for i, vec_id in enumerate(ids):
                self.id_mapping[collection_name].append(vec_id)
                if metadata and i < len(metadata):
                    self.metadata_store[collection_name][vec_id] = metadata[i]
                else:
                    self.metadata_store[collection_name][vec_id] = {}

            logger.info(f"Inserted {len(vectors)} vectors into '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to insert vectors into '{collection_name}': {e}")
            return False

    def search(
        self, collection_name: str, query_vector: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Note: FAISS doesn't support metadata filtering natively.
        We implement it by post-filtering results.
        """
        try:
            if collection_name not in self.indices:
                logger.warning(f"Collection '{collection_name}' does not exist")
                return []

            # Prepare query vector
            query = np.array([query_vector]).astype("float32")

            # Normalize for cosine similarity
            if isinstance(self.indices[collection_name], self.faiss.IndexFlatIP):
                self.faiss.normalize_L2(query)

            # Search (fetch more if we need to filter)
            search_k = top_k * 10 if filters else top_k
            search_k = min(search_k, self.indices[collection_name].ntotal)

            if search_k == 0:
                return []

            distances, indices = self.indices[collection_name].search(query, search_k)

            # Convert to SearchResult objects
            results = []
            for idx, distance in zip(indices[0], distances[0], strict=False):
                if idx == -1:  # FAISS returns -1 for empty results
                    continue

                vec_id = self.id_mapping[collection_name][idx]
                metadata = self.metadata_store[collection_name].get(vec_id, {})

                # Apply filters if provided
                if filters:
                    if not self._matches_filters(metadata, filters):
                        continue

                # Convert distance to score (higher = more similar)
                # For IP (cosine/dot): score = distance (already positive similarity)
                # For L2: score = 1 / (1 + distance)
                if isinstance(self.indices[collection_name], self.faiss.IndexFlatIP):
                    score = float(distance)
                else:
                    score = 1.0 / (1.0 + float(distance))

                results.append(SearchResult(id=vec_id, score=score, metadata=metadata))

                if len(results) >= top_k:
                    break

            logger.info(f"Found {len(results)} results in '{collection_name}'")
            return results

        except Exception as e:
            logger.error(f"Search failed in '{collection_name}': {e}")
            return []

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches all filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            if collection_name in self.indices:
                del self.indices[collection_name]
                del self.metadata_store[collection_name]
                del self.id_mapping[collection_name]
                del self.vector_sizes[collection_name]

                # Delete persisted files
                index_file = self.storage_path / f"{collection_name}.index"
                metadata_file = self.storage_path / f"{collection_name}_metadata.json"
                if index_file.exists():
                    index_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()

                logger.info(f"Deleted collection '{collection_name}'")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        return collection_name in self.indices

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        if collection_name not in self.indices:
            return {}

        return {
            "name": collection_name,
            "count": self.indices[collection_name].ntotal,
            "vector_size": self.vector_sizes.get(collection_name, 0),
            "index_type": type(self.indices[collection_name]).__name__,
        }

    def save_collection(self, collection_name: str) -> bool:
        """Persist collection to disk."""
        try:
            if collection_name not in self.indices:
                return False

            # Save FAISS index
            index_file = self.storage_path / f"{collection_name}.index"
            self.faiss.write_index(self.indices[collection_name], str(index_file))

            # Save metadata
            metadata_file = self.storage_path / f"{collection_name}_metadata.pkl"
            with metadata_file.open("wb") as f:
                pickle.dump(
                    {
                        "metadata": self.metadata_store[collection_name],
                        "id_mapping": self.id_mapping[collection_name],
                        "vector_size": self.vector_sizes[collection_name],
                    },
                    f,
                )

            logger.info(f"Saved collection '{collection_name}' to {self.storage_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save collection '{collection_name}': {e}")
            return False

    def load_collection(self, collection_name: str) -> bool:
        """Load collection from disk."""
        try:
            index_file = self.storage_path / f"{collection_name}.index"
            metadata_file = self.storage_path / f"{collection_name}_metadata.pkl"

            if not index_file.exists() or not metadata_file.exists():
                logger.warning(f"Collection '{collection_name}' files not found")
                return False

            # Load FAISS index
            self.indices[collection_name] = self.faiss.read_index(str(index_file))

            # Load metadata
            with metadata_file.open("rb") as f:
                data = pickle.load(f)
                self.metadata_store[collection_name] = data["metadata"]
                self.id_mapping[collection_name] = data["id_mapping"]
                self.vector_sizes[collection_name] = data["vector_size"]

            logger.info(f"Loaded collection '{collection_name}' from {self.storage_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load collection '{collection_name}': {e}")
            return False

    def save_all(self) -> bool:
        """Save all collections."""
        success = True
        for collection_name in self.indices.keys():
            success = success and self.save_collection(collection_name)
        return success

    def close(self):
        """Clean up resources."""
        logger.info("FAISS adapter closed")
