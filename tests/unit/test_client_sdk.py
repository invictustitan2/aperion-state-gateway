"""Tests for Cortex client SDK."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aperion_cortex_client.client import (
    ContextPack,
    CortexClientConfig,
    Resource,
    VectorHit,
    VectorSearchResult,
    ContextOptimizeResult,
    CortexClientError,
    CortexAuthError,
    CortexConnectionError,
    CortexNotFoundError,
    CortexRateLimitError,
)


# ---------------------------------------------------------------------------
# Model Tests (no mocking needed)
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for client data models."""

    def test_context_pack_fields(self):
        """ContextPack has all required fields."""
        pack = ContextPack(
            generated_at=datetime.now(timezone.utc),
            ttl_seconds=300,
            summary="Test",
            resources=[],
            hot_facts=[],
            signature="sig",
        )
        assert pack.ttl_seconds == 300
        assert pack.freshness == "fresh"
        assert pack.token_count == 0

    def test_context_pack_with_resources(self):
        """ContextPack properly stores resources."""
        resource = Resource(
            id="r1",
            kind="doc",
            name="Test Doc",
            uri="/test",
            metadata={"tags": ["test"]},
        )
        pack = ContextPack(
            generated_at=datetime.now(timezone.utc),
            ttl_seconds=300,
            summary="Test",
            resources=[resource],
            hot_facts=["Fact 1"],
            signature="sig",
        )
        assert len(pack.resources) == 1
        assert pack.resources[0].name == "Test Doc"
        assert pack.hot_facts == ["Fact 1"]

    def test_resource_fields(self):
        """Resource has all required fields."""
        resource = Resource(
            id="r1",
            kind="doc",
            name="Test",
            uri="/test",
        )
        assert resource.metadata == {}
        assert resource.id == "r1"
        assert resource.kind == "doc"

    def test_resource_with_metadata(self):
        """Resource properly stores metadata."""
        resource = Resource(
            id="r1",
            kind="doc",
            name="Test",
            uri="/test",
            metadata={"author": "test", "version": 1},
        )
        assert resource.metadata["author"] == "test"
        assert resource.metadata["version"] == 1

    def test_vector_hit_fields(self):
        """VectorHit has all required fields."""
        hit = VectorHit(
            doc_id="d1",
            score=0.95,
            uri="/doc",
        )
        assert hit.summary is None
        assert hit.metadata == {}
        assert hit.score == 0.95

    def test_vector_hit_with_summary(self):
        """VectorHit properly stores summary and metadata."""
        hit = VectorHit(
            doc_id="d1",
            score=0.95,
            uri="/doc",
            summary="A test document",
            metadata={"category": "test"},
        )
        assert hit.summary == "A test document"
        assert hit.metadata["category"] == "test"

    def test_vector_search_result(self):
        """VectorSearchResult aggregates hits properly."""
        hits = [
            VectorHit(doc_id="d1", score=0.95, uri="/doc1"),
            VectorHit(doc_id="d2", score=0.80, uri="/doc2"),
        ]
        result = VectorSearchResult(
            hits=hits,
            query_embedding_time_ms=10.5,
            search_time_ms=5.2,
        )
        assert len(result.hits) == 2
        assert result.query_embedding_time_ms == 10.5
        assert result.search_time_ms == 5.2

    def test_context_optimize_result(self):
        """ContextOptimizeResult has correct fields."""
        result = ContextOptimizeResult(
            items=[{"content": "test", "priority": 1}],
            total_tokens=100,
            items_included=1,
            items_dropped=0,
            budget_used_percent=50.0,
        )
        assert result.total_tokens == 100
        assert result.items_included == 1
        assert result.budget_used_percent == 50.0


# ---------------------------------------------------------------------------
# Exception Tests
# ---------------------------------------------------------------------------


class TestExceptions:
    """Tests for client exceptions."""

    def test_cortex_client_error(self):
        """CortexClientError stores status code."""
        err = CortexClientError("Test error", 500)
        assert str(err) == "Test error"
        assert err.status_code == 500

    def test_cortex_connection_error(self):
        """CortexConnectionError is a CortexClientError."""
        err = CortexConnectionError("Connection failed")
        assert isinstance(err, CortexClientError)
        assert err.status_code is None

    def test_cortex_auth_error(self):
        """CortexAuthError has 401 status."""
        err = CortexAuthError("Unauthorized", 401)
        assert err.status_code == 401

    def test_cortex_not_found_error(self):
        """CortexNotFoundError has 404 status."""
        err = CortexNotFoundError("Not found", 404)
        assert err.status_code == 404

    def test_cortex_rate_limit_error(self):
        """CortexRateLimitError stores retry_after."""
        err = CortexRateLimitError("Rate limited", retry_after=60.0)
        assert err.status_code == 429
        assert err.retry_after == 60.0


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Tests for client configuration."""

    def test_default_config(self):
        """CortexClientConfig has sensible defaults."""
        config = CortexClientConfig(base_url="http://localhost:4949")
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_backoff == 1.0
        assert config.connection_pool_size == 10
        assert config.token is None

    def test_config_with_token(self):
        """CortexClientConfig stores token."""
        config = CortexClientConfig(
            base_url="http://localhost:4949",
            token="secret-token",
        )
        assert config.token == "secret-token"

    def test_config_custom_values(self):
        """CortexClientConfig accepts custom values."""
        config = CortexClientConfig(
            base_url="http://localhost:4949",
            timeout=60.0,
            max_retries=5,
            retry_backoff=2.0,
            connection_pool_size=20,
        )
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.retry_backoff == 2.0
        assert config.connection_pool_size == 20
