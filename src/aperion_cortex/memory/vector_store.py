"""Vector Store - Model-agnostic embedding and search interface.

Provides a unified interface for vector search operations, supporting
multiple backends (ChromaDB, FAISS) and embedding providers (HuggingFace,
OpenAI).

This module is the core of the "Cold Storage" tier for semantic retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VectorStoreBackend(str, Enum):
    """Supported vector store backends."""

    CHROMA = "chroma"
    FAISS = "faiss"
    MEMORY = "memory"  # In-memory for testing


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


@dataclass
class VectorHit:
    """A single vector search result."""

    doc_id: str
    score: float
    uri: str
    content: str = ""
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A document to be indexed in the vector store."""

    doc_id: str
    content: str
    uri: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models.
    
    Supports both sync and async embedding operations.
    The async versions use thread pool execution by default,
    but can be overridden for native async support (e.g., OpenAI).
    """

    @abstractmethod
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors (synchronous).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        pass

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors (async).
        
        Default implementation runs sync version in thread pool.
        Override for native async support.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        from aperion_cortex.service.executor import run_in_executor
        return await run_in_executor(self.embed_sync, texts)

    def embed_query_sync(self, text: str) -> list[float]:
        """Embed a single query text (synchronous).

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embed_sync([text])[0]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text (async).

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        results = await self.embed([text])
        return results[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


# Global cache for expensive embedding models (avoids 2-5s reload per instance)
_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}


class HuggingFaceEmbedding(EmbeddingModel):
    """HuggingFace sentence-transformers embedding model.
    
    CPU-bound operations run in thread pool automatically via base class.
    Models are cached globally to avoid expensive reloads.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None

    def _load_model(self):
        if self._model is None:
            # Check global cache first
            if self.model_name in _EMBEDDING_MODEL_CACHE:
                cached = _EMBEDDING_MODEL_CACHE[self.model_name]
                self._model = cached["model"]
                self._dimension = cached["dimension"]
                logger.debug(f"Using cached HuggingFace model {self.model_name}")
                return

            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading HuggingFace model {self.model_name}...")
                model = SentenceTransformer(self.model_name)
                # Get dimension from a sample embedding
                sample = model.encode(["test"])
                dimension = len(sample[0])
                
                # Cache for future instances
                _EMBEDDING_MODEL_CACHE[self.model_name] = {
                    "model": model,
                    "dimension": dimension,
                }
                
                self._model = model
                self._dimension = dimension
                logger.info(
                    f"Loaded and cached HuggingFace model {self.model_name} "
                    f"(dim={self._dimension})"
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for HuggingFace embeddings. "
                    "Install with: pip install sentence-transformers"
                )

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding - runs in thread pool via base class."""
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension or 384


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model with native async support and circuit breaker."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = ""):
        self.model_name = model_name
        self.api_key = api_key
        self._sync_client = None
        self._async_client = None
        self._circuit_breaker = None

    def _get_circuit_breaker(self):
        """Get or create circuit breaker for OpenAI API."""
        if self._circuit_breaker is None:
            try:
                from aperion_cortex.service.circuit_breaker import (
                    CircuitBreaker,
                    CircuitBreakerConfig,
                )
                self._circuit_breaker = CircuitBreaker(
                    name=f"openai-embedding-{self.model_name}",
                    config=CircuitBreakerConfig(
                        failure_threshold=3,
                        timeout_seconds=60.0,
                        # Don't trip on validation errors
                        ignore=(ValueError, TypeError),
                    ),
                )
            except ImportError:
                # Circuit breaker not available, return None
                pass
        return self._circuit_breaker

    def _get_sync_client(self):
        if self._sync_client is None:
            try:
                import openai

                self._sync_client = openai.OpenAI(api_key=self.api_key or None)
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
        return self._sync_client

    def _get_async_client(self):
        if self._async_client is None:
            try:
                import openai

                self._async_client = openai.AsyncOpenAI(api_key=self.api_key or None)
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
        return self._async_client

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding with circuit breaker and retry."""
        client = self._get_sync_client()
        breaker = self._get_circuit_breaker()
        
        def _do_embed():
            response = client.embeddings.create(input=texts, model=self.model_name)
            return [data.embedding for data in response.data]
        
        def _do_embed_with_retry():
            try:
                from aperion_cortex.service.retry import retry_sync, RetryConfig
                
                # Retry on transient errors only
                @retry_sync(config=RetryConfig(
                    max_attempts=3,
                    base_delay=1.0,
                    retry_on=(Exception,),
                    no_retry_on=(ValueError, TypeError, KeyError),
                ))
                def _retryable():
                    return _do_embed()
                
                return _retryable()
            except ImportError:
                return _do_embed()
        
        if breaker:
            return breaker.call(_do_embed_with_retry)
        return _do_embed_with_retry()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Native async embedding with circuit breaker and retry."""
        client = self._get_async_client()
        breaker = self._get_circuit_breaker()
        
        async def _do_embed():
            response = await client.embeddings.create(input=texts, model=self.model_name)
            return [data.embedding for data in response.data]
        
        async def _do_embed_with_retry():
            try:
                from aperion_cortex.service.retry import RetryConfig, calculate_backoff
                import asyncio
                
                config = RetryConfig(
                    max_attempts=3,
                    base_delay=1.0,
                    retry_on=(Exception,),
                    no_retry_on=(ValueError, TypeError, KeyError),
                )
                
                last_exception = None
                for attempt in range(config.max_attempts):
                    try:
                        return await _do_embed()
                    except config.no_retry_on:
                        raise
                    except config.retry_on as e:
                        last_exception = e
                        if attempt < config.max_attempts - 1:
                            delay = calculate_backoff(attempt, config.base_delay)
                            logger.warning(
                                f"OpenAI retry {attempt + 1}/{config.max_attempts} "
                                f"after {delay:.2f}s: {e}"
                            )
                            await asyncio.sleep(delay)
                
                raise last_exception  # type: ignore
            except ImportError:
                return await _do_embed()
        
        if breaker:
            return await breaker.call_async(_do_embed_with_retry)
        return await _do_embed_with_retry()

    @property
    def dimension(self) -> int:
        # text-embedding-3-small: 1536, text-embedding-3-large: 3072
        if "large" in self.model_name:
            return 3072
        return 1536


class VectorStore(ABC):
    """Abstract base class for vector stores.
    
    Supports both sync and async operations. Default async implementations
    run sync methods in thread pool. Override for native async support.
    """

    @abstractmethod
    def add_documents_sync(self, documents: list[Document]) -> list[str]:
        """Add documents to the store (synchronous).

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs
        """
        pass

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the store (async).
        
        Default runs sync version in thread pool.
        """
        from aperion_cortex.service.executor import run_in_executor
        return await run_in_executor(self.add_documents_sync, documents)

    @abstractmethod
    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Search for similar documents (synchronous).

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of VectorHit results
        """
        pass

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Search for similar documents (async).
        
        Default runs sync version in thread pool.
        """
        from aperion_cortex.service.executor import run_in_executor
        return await run_in_executor(self.search_sync, query, top_k, filter_metadata)

    @abstractmethod
    def delete_sync(self, doc_ids: list[str]) -> int:
        """Delete documents by ID (synchronous).

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        pass

    async def delete(self, doc_ids: list[str]) -> int:
        """Delete documents by ID (async)."""
        from aperion_cortex.service.executor import run_in_executor
        return await run_in_executor(self.delete_sync, doc_ids)

    @abstractmethod
    def count(self) -> int:
        """Return total number of documents in the store."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for testing and small datasets."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self._documents: dict[str, Document] = {}
        self._embeddings: dict[str, list[float]] = {}

    def add_documents_sync(self, documents: list[Document]) -> list[str]:
        doc_ids = []
        texts = [doc.content for doc in documents]

        if texts:
            # Use sync embedding for sync method
            embeddings = self.embedding_model.embed_sync(texts)

            for doc, embedding in zip(documents, embeddings):
                self._documents[doc.doc_id] = doc
                self._embeddings[doc.doc_id] = embedding
                doc_ids.append(doc.doc_id)

        logger.debug(f"Added {len(doc_ids)} documents to in-memory store")
        return doc_ids

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Async add with native async embedding."""
        doc_ids = []
        texts = [doc.content for doc in documents]

        if texts:
            embeddings = await self.embedding_model.embed(texts)

            for doc, embedding in zip(documents, embeddings):
                self._documents[doc.doc_id] = doc
                self._embeddings[doc.doc_id] = embedding
                doc_ids.append(doc.doc_id)

        logger.debug(f"Added {len(doc_ids)} documents to in-memory store")
        return doc_ids

    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        if not self._documents:
            return []

        query_embedding = self.embedding_model.embed_query_sync(query)

        # Compute cosine similarity
        scores: list[tuple[str, float]] = []
        for doc_id, doc_embedding in self._embeddings.items():
            doc = self._documents[doc_id]

            # Apply metadata filter
            if filter_metadata:
                match = all(
                    doc.metadata.get(k) == v for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            score = self._cosine_similarity(query_embedding, doc_embedding)
            scores.append((doc_id, score))

        # Sort by score (descending) and take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]

        results = []
        for doc_id, score in top_scores:
            doc = self._documents[doc_id]
            results.append(
                VectorHit(
                    doc_id=doc_id,
                    score=score,
                    uri=doc.uri,
                    content=doc.content[:200],  # Preview
                    summary=doc.metadata.get("summary"),
                    metadata=doc.metadata,
                )
            )

        return results

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Async search with native async embedding."""
        if not self._documents:
            return []

        query_embedding = await self.embedding_model.embed_query(query)

        # Compute cosine similarity (CPU-bound but fast for small datasets)
        scores: list[tuple[str, float]] = []
        for doc_id, doc_embedding in self._embeddings.items():
            doc = self._documents[doc_id]

            if filter_metadata:
                match = all(
                    doc.metadata.get(k) == v for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            score = self._cosine_similarity(query_embedding, doc_embedding)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]

        results = []
        for doc_id, score in top_scores:
            doc = self._documents[doc_id]
            results.append(
                VectorHit(
                    doc_id=doc_id,
                    score=score,
                    uri=doc.uri,
                    content=doc.content[:200],
                    summary=doc.metadata.get("summary"),
                    metadata=doc.metadata,
                )
            )

        return results

    def delete_sync(self, doc_ids: list[str]) -> int:
        count = 0
        for doc_id in doc_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                del self._embeddings[doc_id]
                count += 1
        return count

    def count(self) -> int:
        return len(self._documents)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store backend.
    
    Note: ChromaDB operations are synchronous. The async methods run
    sync operations in thread pool via base class default.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_directory: str | Path | None = None,
        collection_name: str = "cortex_documents",
    ):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _init_client(self):
        if self._client is None:
            try:
                import chromadb

                if self.persist_directory:
                    self._client = chromadb.PersistentClient(
                        path=str(self.persist_directory)
                    )
                else:
                    self._client = chromadb.Client()

                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    f"Initialized ChromaDB collection '{self.collection_name}'"
                )
            except ImportError:
                raise ImportError(
                    "chromadb is required for Chroma backend. "
                    "Install with: pip install chromadb"
                )

    def add_documents_sync(self, documents: list[Document]) -> list[str]:
        self._init_client()

        if not documents:
            return []

        doc_ids = [doc.doc_id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings (sync)
        embeddings = self.embedding_model.embed_sync(texts)

        self._collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.debug(f"Added {len(doc_ids)} documents to ChromaDB")
        return doc_ids

    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        self._init_client()

        query_embedding = self.embedding_model.embed_query_sync(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata if filter_metadata else None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Convert distance to similarity

                hits.append(
                    VectorHit(
                        doc_id=doc_id,
                        score=score,
                        uri=results["metadatas"][0][i].get("uri", "")
                        if results["metadatas"]
                        else "",
                        content=results["documents"][0][i][:200]
                        if results["documents"]
                        else "",
                        summary=results["metadatas"][0][i].get("summary")
                        if results["metadatas"]
                        else None,
                        metadata=results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                    )
                )

        return hits

    def delete_sync(self, doc_ids: list[str]) -> int:
        self._init_client()
        self._collection.delete(ids=doc_ids)
        return len(doc_ids)

    def count(self) -> int:
        self._init_client()
        return self._collection.count()


def create_vector_store(
    backend: VectorStoreBackend = VectorStoreBackend.MEMORY,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE,
    embedding_model: str = "all-MiniLM-L6-v2",
    persist_directory: str | Path | None = None,
    collection_name: str = "cortex_documents",
    openai_api_key: str = "",
) -> VectorStore:
    """Factory function to create a vector store.

    Args:
        backend: Vector store backend (chroma, faiss, memory)
        embedding_provider: Embedding provider (huggingface, openai)
        embedding_model: Model name for embeddings
        persist_directory: Directory for persistent storage
        collection_name: Collection/index name
        openai_api_key: OpenAI API key (required for OpenAI embeddings)

    Returns:
        Configured VectorStore instance
    """
    # Create embedding model
    if embedding_provider == EmbeddingProvider.OPENAI:
        emb_model = OpenAIEmbedding(embedding_model, openai_api_key)
    else:
        emb_model = HuggingFaceEmbedding(embedding_model)

    # Create vector store
    if backend == VectorStoreBackend.CHROMA:
        return ChromaVectorStore(emb_model, persist_directory, collection_name)
    elif backend == VectorStoreBackend.FAISS:
        # FAISS implementation would go here
        raise NotImplementedError("FAISS backend not yet implemented")
    else:
        return InMemoryVectorStore(emb_model)
