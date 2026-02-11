"""Integration tests for Aperion compatibility.

These tests verify that Cortex client models and responses are compatible
with the Aperion state gateway models, ensuring seamless integration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Aperion Model Replicas (for compatibility testing)
# These mirror the models in aperion-legendary-ai-main/stack/aperion/state_gateway/models.py
# ---------------------------------------------------------------------------


class AperionResource(BaseModel):
    """Aperion's Resource model."""
    id: str
    kind: str
    name: str
    uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AperionContextPack(BaseModel):
    """Aperion's ContextPack model."""
    generated_at: datetime
    ttl_seconds: int
    summary: str
    resources: list[AperionResource] = Field(default_factory=list)
    hot_facts: list[str] = Field(default_factory=list)
    signature: str


class AperionVectorHit(BaseModel):
    """Aperion's VectorHit model."""
    doc_id: str
    score: float
    uri: str
    summary: str | None = None


class AperionVectorSearchRequest(BaseModel):
    """Aperion's VectorSearchRequest model."""
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class AperionVectorSearchResponse(BaseModel):
    """Aperion's VectorSearchResponse model."""
    hits: list[AperionVectorHit]


# ---------------------------------------------------------------------------
# Cortex Model Imports
# ---------------------------------------------------------------------------

from aperion_cortex_client.client import (
    ContextPack as CortexContextPack,
    Resource as CortexResource,
    VectorHit as CortexVectorHit,
    VectorSearchResult as CortexVectorSearchResult,
)


# ---------------------------------------------------------------------------
# Schema Compatibility Tests
# ---------------------------------------------------------------------------


class TestResourceCompatibility:
    """Tests for Resource schema compatibility."""

    def test_cortex_resource_has_required_fields(self):
        """Cortex Resource has all fields Aperion expects."""
        cortex = CortexResource(
            id="r1",
            kind="document",
            name="Test Doc",
            uri="/docs/test",
            metadata={"author": "test"},
        )
        
        # Should be convertible to Aperion format
        aperion_data = {
            "id": cortex.id,
            "kind": cortex.kind,
            "name": cortex.name,
            "uri": cortex.uri,
            "metadata": cortex.metadata,
        }
        
        aperion = AperionResource(**aperion_data)
        assert aperion.id == cortex.id
        assert aperion.kind == cortex.kind
        assert aperion.name == cortex.name
        assert aperion.uri == cortex.uri
        assert aperion.metadata == cortex.metadata

    def test_resource_json_roundtrip(self):
        """Resource survives JSON serialization between systems."""
        original = {
            "id": "res-123",
            "kind": "file",
            "name": "config.yaml",
            "uri": "/config/config.yaml",
            "metadata": {"size": 1024, "tags": ["config", "yaml"]},
        }
        
        # Cortex receives and parses
        cortex = CortexResource(**original)
        
        # Serialize for Aperion
        json_str = json.dumps({
            "id": cortex.id,
            "kind": cortex.kind,
            "name": cortex.name,
            "uri": cortex.uri,
            "metadata": cortex.metadata,
        })
        
        # Aperion receives and parses
        aperion = AperionResource(**json.loads(json_str))
        
        assert aperion.id == original["id"]
        assert aperion.metadata["tags"] == ["config", "yaml"]


class TestContextPackCompatibility:
    """Tests for ContextPack schema compatibility."""

    def test_cortex_context_pack_has_required_fields(self):
        """Cortex ContextPack has all fields Aperion expects."""
        now = datetime.now(timezone.utc)
        
        cortex = CortexContextPack(
            generated_at=now,
            ttl_seconds=300,
            summary="Test context summary",
            resources=[
                CortexResource(id="r1", kind="doc", name="Doc 1", uri="/doc1"),
            ],
            hot_facts=["Fact 1", "Fact 2"],
            signature="sig-abc123",
            token_count=500,  # Cortex-only field
            freshness="fresh",  # Cortex-only field
        )
        
        # Convert to Aperion format (only required fields)
        aperion_data = {
            "generated_at": cortex.generated_at,
            "ttl_seconds": cortex.ttl_seconds,
            "summary": cortex.summary,
            "resources": [
                {"id": r.id, "kind": r.kind, "name": r.name, "uri": r.uri, "metadata": r.metadata}
                for r in cortex.resources
            ],
            "hot_facts": cortex.hot_facts,
            "signature": cortex.signature,
        }
        
        aperion = AperionContextPack(**aperion_data)
        assert aperion.ttl_seconds == 300
        assert aperion.summary == "Test context summary"
        assert len(aperion.resources) == 1
        assert aperion.hot_facts == ["Fact 1", "Fact 2"]

    def test_context_pack_extra_fields_ignored(self):
        """Aperion ignores Cortex-only fields gracefully."""
        # Cortex returns extra fields
        cortex_response = {
            "generated_at": "2026-02-08T12:00:00Z",
            "ttl_seconds": 120,
            "summary": "Summary",
            "resources": [],
            "hot_facts": [],
            "signature": "sig",
            "token_count": 1000,  # Cortex-only
            "freshness": "cached",  # Cortex-only
        }
        
        # Aperion should parse without error (extra fields ignored by Pydantic v2)
        aperion = AperionContextPack.model_validate(cortex_response)
        assert aperion.ttl_seconds == 120

    def test_context_pack_json_roundtrip(self):
        """ContextPack survives JSON serialization between systems."""
        now = datetime.now(timezone.utc)
        
        cortex = CortexContextPack(
            generated_at=now,
            ttl_seconds=600,
            summary="Multi-resource context",
            resources=[
                CortexResource(id="r1", kind="file", name="main.py", uri="/src/main.py"),
                CortexResource(id="r2", kind="doc", name="README", uri="/README.md"),
            ],
            hot_facts=["User is debugging", "Focus on performance"],
            signature="hmac-sha256:abc123",
        )
        
        # Serialize
        json_str = json.dumps({
            "generated_at": cortex.generated_at.isoformat(),
            "ttl_seconds": cortex.ttl_seconds,
            "summary": cortex.summary,
            "resources": [
                {"id": r.id, "kind": r.kind, "name": r.name, "uri": r.uri, "metadata": r.metadata}
                for r in cortex.resources
            ],
            "hot_facts": cortex.hot_facts,
            "signature": cortex.signature,
        })
        
        # Deserialize as Aperion
        data = json.loads(json_str)
        data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        aperion = AperionContextPack(**data)
        
        assert len(aperion.resources) == 2
        assert aperion.resources[0].name == "main.py"


class TestVectorSearchCompatibility:
    """Tests for VectorSearch schema compatibility."""

    def test_cortex_vector_hit_compatible(self):
        """Cortex VectorHit is compatible with Aperion."""
        cortex = CortexVectorHit(
            doc_id="doc-123",
            score=0.95,
            uri="/docs/guide.md",
            summary="A helpful guide",
            metadata={"category": "docs"},  # Cortex-only field
        )
        
        # Convert to Aperion format
        aperion_data = {
            "doc_id": cortex.doc_id,
            "score": cortex.score,
            "uri": cortex.uri,
            "summary": cortex.summary,
        }
        
        aperion = AperionVectorHit(**aperion_data)
        assert aperion.doc_id == "doc-123"
        assert aperion.score == 0.95
        assert aperion.summary == "A helpful guide"

    def test_vector_search_response_compatible(self):
        """Cortex VectorSearchResult is compatible with Aperion response."""
        cortex = CortexVectorSearchResult(
            hits=[
                CortexVectorHit(doc_id="d1", score=0.9, uri="/d1"),
                CortexVectorHit(doc_id="d2", score=0.8, uri="/d2", summary="Doc 2"),
            ],
            query_embedding_time_ms=15.5,  # Cortex-only
            search_time_ms=5.2,  # Cortex-only
        )
        
        # Convert to Aperion format
        aperion_hits = [
            AperionVectorHit(
                doc_id=h.doc_id,
                score=h.score,
                uri=h.uri,
                summary=h.summary,
            )
            for h in cortex.hits
        ]
        
        aperion = AperionVectorSearchResponse(hits=aperion_hits)
        assert len(aperion.hits) == 2
        assert aperion.hits[0].score == 0.9

    def test_vector_hit_extra_metadata_ignored(self):
        """Aperion ignores Cortex metadata field."""
        cortex_response = {
            "doc_id": "doc-1",
            "score": 0.85,
            "uri": "/test",
            "summary": None,
            "metadata": {"extra": "ignored"},  # Cortex-only
        }
        
        aperion = AperionVectorHit.model_validate(cortex_response)
        assert aperion.doc_id == "doc-1"


# ---------------------------------------------------------------------------
# Event Bridge Integration Tests
# ---------------------------------------------------------------------------


class TestEventBridgeIntegration:
    """Tests for event bridge with mock EventBus."""

    @pytest.mark.asyncio
    async def test_bridge_emits_to_eventbus_interface(self):
        """Bridge correctly calls EventBus.emit() with expected signature."""
        from aperion_cortex_client.integration.event_bridge import CortexEventBridge
        
        # Track emit calls
        emit_calls = []
        
        class MockEventBus:
            def emit(
                self,
                event_type: str,
                payload: dict,
                source: str | None = None,
                correlation_id: str | None = None,
                wait_for_handlers: bool = False,
            ) -> str:
                emit_calls.append({
                    "event_type": event_type,
                    "payload": payload,
                    "source": source,
                    "correlation_id": correlation_id,
                })
                return f"evt-{len(emit_calls)}"
        
        bus = MockEventBus()
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=bus,
        )
        
        # Simulate event processing
        await bridge._handle_event_data(json.dumps({
            "event_type": "state.updated",
            "payload": {"snapshot_id": "snap-1"},
            "correlation_id": "req-123",
        }))
        
        assert len(emit_calls) == 1
        call = emit_calls[0]
        assert call["event_type"] == "cortex.state.updated"
        assert call["source"] == "cortex"
        assert call["correlation_id"] == "req-123"
        assert call["payload"]["snapshot_id"] == "snap-1"

    @pytest.mark.asyncio
    async def test_bridge_enriches_payload(self):
        """Bridge adds metadata to event payload."""
        from aperion_cortex_client.integration.event_bridge import CortexEventBridge
        
        received_payload = None
        
        class MockEventBus:
            def emit(self, event_type, payload, **kwargs):
                nonlocal received_payload
                received_payload = payload
                return "evt-1"
        
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=MockEventBus(),
        )
        
        await bridge._handle_event_data(json.dumps({
            "event_type": "vector.indexed",
            "payload": {"doc_id": "doc-1"},
            "resource_ids": ["r1", "r2"],
        }))
        
        assert received_payload is not None
        assert received_payload["doc_id"] == "doc-1"
        assert received_payload["cortex_event_type"] == "vector.indexed"
        assert received_payload["resource_ids"] == ["r1", "r2"]
        assert "bridge_timestamp" in received_payload


# ---------------------------------------------------------------------------
# End-to-End Simulation Tests
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    """Tests simulating end-to-end flows between Cortex and Aperion."""

    def test_snapshot_flow(self):
        """Simulate: Agent requests snapshot → Cortex returns → Agent parses."""
        # 1. Cortex generates snapshot
        now = datetime.now(timezone.utc)
        cortex_pack = CortexContextPack(
            generated_at=now,
            ttl_seconds=120,
            summary="Agent context: User working on Python project",
            resources=[
                CortexResource(
                    id="r1",
                    kind="file",
                    name="main.py",
                    uri="/src/main.py",
                    metadata={"language": "python"},
                ),
            ],
            hot_facts=[
                "User is debugging a TypeError",
                "Project uses FastAPI",
            ],
            signature="hmac-sha256:validated",
            token_count=250,
        )
        
        # 2. Serialize as API response
        api_response = {
            "generated_at": cortex_pack.generated_at.isoformat(),
            "ttl_seconds": cortex_pack.ttl_seconds,
            "summary": cortex_pack.summary,
            "resources": [
                {
                    "id": r.id,
                    "kind": r.kind,
                    "name": r.name,
                    "uri": r.uri,
                    "metadata": r.metadata,
                }
                for r in cortex_pack.resources
            ],
            "hot_facts": cortex_pack.hot_facts,
            "signature": cortex_pack.signature,
            "token_count": cortex_pack.token_count,
            "freshness": cortex_pack.freshness,
        }
        
        # 3. Agent receives and parses (using Aperion model)
        json_str = json.dumps(api_response)
        data = json.loads(json_str)
        data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        
        aperion_pack = AperionContextPack.model_validate(data)
        
        # 4. Verify agent can use the data
        assert aperion_pack.summary.startswith("Agent context:")
        assert len(aperion_pack.resources) == 1
        assert aperion_pack.resources[0].name == "main.py"
        assert "FastAPI" in aperion_pack.hot_facts[1]

    def test_vector_search_flow(self):
        """Simulate: Agent searches → Cortex returns hits → Agent processes."""
        # 1. Agent constructs search request (Aperion format)
        aperion_request = AperionVectorSearchRequest(
            query="How to handle async errors in FastAPI?",
            top_k=5,
        )
        
        # 2. Request is valid for Cortex (Cortex accepts top_k up to 100)
        assert aperion_request.top_k <= 100
        
        # 3. Cortex returns results
        cortex_result = CortexVectorSearchResult(
            hits=[
                CortexVectorHit(
                    doc_id="doc-async-errors",
                    score=0.92,
                    uri="/docs/async-errors.md",
                    summary="Guide to async error handling",
                    metadata={"category": "errors"},
                ),
                CortexVectorHit(
                    doc_id="doc-fastapi-deps",
                    score=0.85,
                    uri="/docs/fastapi-deps.md",
                    summary="FastAPI dependency injection",
                ),
            ],
            query_embedding_time_ms=12.5,
            search_time_ms=3.2,
        )
        
        # 4. Agent parses response (using Aperion models)
        aperion_hits = [
            AperionVectorHit(
                doc_id=h.doc_id,
                score=h.score,
                uri=h.uri,
                summary=h.summary,
            )
            for h in cortex_result.hits
        ]
        
        aperion_response = AperionVectorSearchResponse(hits=aperion_hits)
        
        # 5. Agent uses results
        assert len(aperion_response.hits) == 2
        top_hit = aperion_response.hits[0]
        assert top_hit.score > 0.9
        assert "async" in top_hit.uri
