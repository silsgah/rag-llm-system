"""
Modular vector database adapters.

This module provides a unified interface for different vector database backends
(Qdrant, FAISS, ChromaDB, Pinecone) without modifying existing code.

Usage:
    from rag_llm_system.infrastructure.vector_stores import VectorStoreFactory

    # Create any backend
    store = VectorStoreFactory.create("faiss")
    store = VectorStoreFactory.create("chroma")
    store = VectorStoreFactory.create("qdrant")  # Uses existing implementation
"""

from .base import SearchResult, VectorDBAdapter
from .factory import VectorStoreFactory

__all__ = ["VectorDBAdapter", "SearchResult", "VectorStoreFactory"]
