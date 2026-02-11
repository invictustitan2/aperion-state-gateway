"""Integration tests for Vector Store functionality."""

import pytest

from aperion_cortex.memory.vector_store import (
    Document,
    InMemoryVectorStore,
    VectorStoreBackend,
    create_vector_store,
)


class TestInMemoryVectorStore:
    """Tests for in-memory vector store."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a simple mock embedding model that implements sync interface."""
        class MockEmbedding:
            def embed_sync(self, texts):
                # Return simple deterministic embeddings based on text length
                return [[len(t) / 100.0] * 384 for t in texts]

            def embed_query_sync(self, text):
                return [len(text) / 100.0] * 384

            @property
            def dimension(self):
                return 384

        return MockEmbedding()

    def test_add_documents(self, mock_embedding_model):
        """Adding documents should increase count."""
        store = InMemoryVectorStore(mock_embedding_model)

        docs = [
            Document(doc_id="doc1", content="Hello world", uri="file://doc1.txt"),
            Document(doc_id="doc2", content="Another document", uri="file://doc2.txt"),
        ]

        ids = store.add_documents_sync(docs)

        assert len(ids) == 2
        assert store.count() == 2

    def test_search_returns_results(self, mock_embedding_model):
        """Search should return matching documents."""
        store = InMemoryVectorStore(mock_embedding_model)

        docs = [
            Document(doc_id="doc1", content="Short", uri="file://doc1.txt"),
            Document(doc_id="doc2", content="A much longer document with more content", uri="file://doc2.txt"),
        ]
        store.add_documents_sync(docs)

        results = store.search_sync("medium length query", top_k=2)

        assert len(results) == 2
        assert all(r.score >= 0 for r in results)

    def test_search_with_metadata_filter(self, mock_embedding_model):
        """Search should filter by metadata."""
        store = InMemoryVectorStore(mock_embedding_model)

        docs = [
            Document(doc_id="doc1", content="First doc", uri="", metadata={"type": "a"}),
            Document(doc_id="doc2", content="Second doc", uri="", metadata={"type": "b"}),
            Document(doc_id="doc3", content="Third doc", uri="", metadata={"type": "a"}),
        ]
        store.add_documents_sync(docs)

        results = store.search_sync("query", top_k=10, filter_metadata={"type": "a"})

        assert len(results) == 2
        assert all(r.metadata.get("type") == "a" for r in results)

    def test_delete_documents(self, mock_embedding_model):
        """Delete should remove documents."""
        store = InMemoryVectorStore(mock_embedding_model)

        docs = [
            Document(doc_id="doc1", content="First", uri=""),
            Document(doc_id="doc2", content="Second", uri=""),
        ]
        store.add_documents_sync(docs)

        deleted = store.delete_sync(["doc1"])

        assert deleted == 1
        assert store.count() == 1

    def test_search_empty_store(self, mock_embedding_model):
        """Search on empty store should return empty list."""
        store = InMemoryVectorStore(mock_embedding_model)

        results = store.search_sync("query")

        assert results == []

    def test_top_k_limit(self, mock_embedding_model):
        """Search should respect top_k limit."""
        store = InMemoryVectorStore(mock_embedding_model)

        docs = [Document(doc_id=f"doc{i}", content=f"Content {i}", uri="") for i in range(10)]
        store.add_documents_sync(docs)

        results = store.search_sync("query", top_k=3)

        assert len(results) == 3


class TestVectorStoreFactory:
    """Tests for vector store factory function."""

    def test_create_memory_store(self):
        """Should create in-memory store."""
        # This would require mocking the embedding model import
        # For now, just test that the function exists
        assert callable(create_vector_store)

    def test_unsupported_backend_raises(self):
        """FAISS backend should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            create_vector_store(backend=VectorStoreBackend.FAISS)


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Document should store all fields."""
        doc = Document(
            doc_id="test-id",
            content="Test content",
            uri="file://test.txt",
            metadata={"key": "value"},
        )

        assert doc.doc_id == "test-id"
        assert doc.content == "Test content"
        assert doc.uri == "file://test.txt"
        assert doc.metadata["key"] == "value"

    def test_document_defaults(self):
        """Document should have sensible defaults."""
        doc = Document(doc_id="id", content="content")

        assert doc.uri == ""
        assert doc.metadata == {}
