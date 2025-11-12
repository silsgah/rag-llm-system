"""
Factory for creating vector database adapters.

Usage:
    from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory

    # Create any backend
    store = VectorStoreFactory.create("faiss")
    store = VectorStoreFactory.create("chroma")
    store = VectorStoreFactory.create("qdrant")

    # Use the same interface for all
    store.create_collection("my_collection", vector_size=384)
    store.insert("my_collection", vectors=[...], metadata=[...])
    results = store.search("my_collection", query_vector=[...], top_k=10)
"""

from enum import Enum
from typing import Dict

from loguru import logger

from .base import VectorDBAdapter


class VectorDBBackend(str, Enum):
    """Supported vector database backends."""

    QDRANT = "qdrant"
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"  # Reserved for future implementation


class VectorStoreFactory:
    """
    Factory for creating vector database adapters.

    Handles lazy loading of backends to avoid requiring all dependencies.
    """

    @staticmethod
    def create(backend: str, **kwargs) -> VectorDBAdapter:
        """
        Create a vector database adapter.

        Args:
            backend: Backend type ("qdrant", "faiss", "chroma", "pinecone")
            **kwargs: Backend-specific configuration

        Returns:
            VectorDBAdapter instance

        Raises:
            ValueError: If backend is not supported
            ImportError: If required dependencies are not installed

        Examples:
            # FAISS with custom storage
            store = VectorStoreFactory.create("faiss", storage_path="./my_indices")

            # ChromaDB with persistence
            store = VectorStoreFactory.create("chroma", persist_directory="./my_chroma")

            # Qdrant (uses existing connection)
            store = VectorStoreFactory.create("qdrant")
        """
        backend = backend.lower()

        if backend == VectorDBBackend.QDRANT:
            return VectorStoreFactory._create_qdrant(**kwargs)

        elif backend == VectorDBBackend.FAISS:
            return VectorStoreFactory._create_faiss(**kwargs)

        elif backend == VectorDBBackend.CHROMA:
            return VectorStoreFactory._create_chroma(**kwargs)

        elif backend == VectorDBBackend.PINECONE:
            return VectorStoreFactory._create_pinecone(**kwargs)

        else:
            raise ValueError(
                f"Unsupported backend: {backend}. " f"Supported backends: {[b.value for b in VectorDBBackend]}"
            )

    @staticmethod
    def _create_qdrant(**kwargs) -> VectorDBAdapter:
        """Create Qdrant adapter (uses existing connection)."""
        try:
            from .qdrant_adapter import QdrantAdapter

            logger.info("Creating Qdrant adapter")
            return QdrantAdapter(**kwargs)
        except ImportError as e:
            raise ImportError("Qdrant dependencies not available. " "Ensure qdrant-client is installed.") from e

    @staticmethod
    def _create_faiss(**kwargs) -> VectorDBAdapter:
        """Create FAISS adapter."""
        try:
            from .faiss_adapter import FAISSAdapter

            logger.info("Creating FAISS adapter")
            return FAISSAdapter(**kwargs)
        except ImportError as e:
            raise ImportError(
                "FAISS is not installed. Install with: " "pip install faiss-cpu (or faiss-gpu for GPU support)"
            ) from e

    @staticmethod
    def _create_chroma(**kwargs) -> VectorDBAdapter:
        """Create ChromaDB adapter."""
        try:
            from .chroma_adapter import ChromaAdapter

            logger.info("Creating ChromaDB adapter")
            return ChromaAdapter(**kwargs)
        except ImportError as e:
            raise ImportError("ChromaDB is not installed. Install with: " "pip install chromadb") from e

    @staticmethod
    def _create_pinecone(**kwargs) -> VectorDBAdapter:
        """Create Pinecone adapter (future implementation)."""
        raise NotImplementedError(
            "Pinecone adapter not yet implemented. "
            "You can add it by implementing PineconeAdapter in pinecone_adapter.py"
        )

    @staticmethod
    def list_available_backends() -> Dict[str, bool]:
        """
        Check which backends are available (dependencies installed).

        Returns:
            Dict mapping backend name to availability
        """
        import importlib.util

        availability = {
            "qdrant": importlib.util.find_spec("qdrant_client") is not None,
            "faiss": importlib.util.find_spec("faiss") is not None,
            "chroma": importlib.util.find_spec("chromadb") is not None,
            "pinecone": importlib.util.find_spec("pinecone") is not None,
        }

        return availability
