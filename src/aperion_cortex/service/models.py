"""Pydantic models backing the Cortex API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Core Resource Models
# ---------------------------------------------------------------------------


class Resource(BaseModel):
    """A discoverable resource in the system."""

    id: str
    kind: str
    name: str
    uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPack(BaseModel):
    """A signed, token-optimized context snapshot for LLM consumption.

    The ContextPack is the primary output of the Cortex's "Warm Context" tier.
    It contains a summary, resources, and hot facts that fit within the
    LLM's context window, signed for integrity verification.
    """

    generated_at: datetime
    ttl_seconds: int
    summary: str
    resources: list[Resource] = Field(default_factory=list)
    hot_facts: list[str] = Field(default_factory=list)
    signature: str
    token_count: int = Field(default=0, description="Estimated token count")

    @property
    def expires_at(self) -> datetime:
        return self.generated_at + timedelta(seconds=self.ttl_seconds)

    @field_validator("hot_facts")
    @classmethod
    def _validate_hot_facts(cls, values: list[str]) -> list[str]:
        for value in values:
            if len(value.encode("utf-8")) > 2048:
                raise ValueError("Hot fact entries must remain under 2KB")
        return values


class SnapshotResponse(ContextPack):
    """Extended ContextPack with freshness indicator."""

    freshness: str = Field(description="Indicator: fresh, cached, restored, stale")


# ---------------------------------------------------------------------------
# Vector Search Models
# ---------------------------------------------------------------------------


class VectorHit(BaseModel):
    """A single vector search result."""

    doc_id: str
    score: float
    uri: str
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchRequest(BaseModel):
    """Request for semantic vector search."""

    query: str = Field(..., min_length=1, max_length=10000)  # 10KB max
    top_k: int = Field(default=5, ge=1, le=100)
    filter_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def _validate_query_size(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 10 * 1024:
            raise ValueError("Query must be under 10KB")
        return v


class VectorSearchResponse(BaseModel):
    """Response from vector search endpoint."""

    hits: list[VectorHit]
    query_embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------


class Database(BaseModel):
    """Database metadata."""

    name: str
    engine: str
    dsn_ref: str
    table_count: int | None = None
    row_count: int | None = None
    last_migration: str | None = None


class DatabaseSchema(BaseModel):
    """Database schema introspection result."""

    tables: list[dict[str, Any]] = Field(default_factory=list)
    schema_hash: str
    computed_at: datetime


class DatabaseSample(BaseModel):
    """Sample rows from a database table."""

    database: str
    table: str
    limit: int
    rows: list[dict[str, Any]]
    truncated: bool = False


# ---------------------------------------------------------------------------
# Service Health Models
# ---------------------------------------------------------------------------


class Service(BaseModel):
    """Registered service metadata."""

    name: str
    kind: str
    endpoint: str
    version: str | None = None
    git_sha: str | None = None
    deployed_at: datetime | None = None


class ServiceHealth(BaseModel):
    """Health status for a service."""

    name: str
    status: str
    latency_ms: int
    last_checked: datetime


class ServiceHealthResponse(BaseModel):
    """Response for service health endpoint."""

    services: list[ServiceHealth]


# ---------------------------------------------------------------------------
# Filesystem Models
# ---------------------------------------------------------------------------


class FileEntry(BaseModel):
    """Filesystem entry metadata."""

    path: str
    type: str
    size: int
    modified_at: datetime | None = None


class FileReadResponse(BaseModel):
    """Response from file read operation."""

    path: str
    bytes_returned: int
    text: str
    truncated: bool


# ---------------------------------------------------------------------------
# Audit & Event Models
# ---------------------------------------------------------------------------


class AuditLogEntry(BaseModel):
    """Single audit log entry for tool invocations."""

    tool_name: str
    scope: str
    input_bytes: int
    output_bytes: int
    latency_ms: float
    resource_ids: list[str] = Field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    timestamp: datetime


class AuditRecordRequest(BaseModel):
    """Request to record an agent execution to the audit ledger.

    This is the primary interface for agents to log their execution results.
    The Cortex is the single writer to the training ledger.
    """

    run_id: str = Field(description="Unique run identifier")
    correlation_id: str | None = Field(default=None, description="Request correlation")
    agent_name: str = Field(description="Name of the executing agent")
    method_name: str = Field(description="Method/function that was called")
    prompt: str = Field(description="Input prompt to the agent")
    raw_output: str = Field(description="Raw LLM output")
    parsed_data: dict[str, Any] | None = Field(default=None)
    parse_ok: bool = Field(default=False)
    parse_strategy: str | None = Field(default=None)
    parse_confidence: float | None = Field(default=None)
    parse_error: str | None = Field(default=None)
    provider_name: str = Field(default="")
    model_name: str = Field(default="")
    latency_ms: float = Field(default=0.0)
    context_refs: list[str] = Field(default_factory=list)
    policy_decisions: list[dict[str, Any]] = Field(default_factory=list)


class AuditRecordResponse(BaseModel):
    """Response from audit record endpoint."""

    run_id: str
    recorded_at: datetime
    ledger_path: str


class EventPayload(BaseModel):
    """Event payload for SSE streaming."""

    event_type: str
    payload: dict[str, Any]
    timestamp: float
    resource_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery Models
# ---------------------------------------------------------------------------


class DiscoverResponse(BaseModel):
    """Response from state discovery endpoint."""

    resources: list[Resource]


# ---------------------------------------------------------------------------
# Error Models
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    reason: str | None = None
    correlation_id: str | None = None


# ---------------------------------------------------------------------------
# Context Optimization Models
# ---------------------------------------------------------------------------


class ContextItem(BaseModel):
    """An item to include in context packing."""

    content: str = Field(..., max_length=100000)  # 100KB max per item
    priority: int = Field(default=0, description="Higher = more important")
    category: str = Field(default="general", max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextOptimizeRequest(BaseModel):
    """Request to optimize context for LLM consumption."""

    items: list[ContextItem] = Field(..., max_length=500)  # Max 500 items
    max_tokens: int = Field(default=4096, ge=100, le=128000)
    reserve_tokens: int = Field(default=0, ge=0, description="Tokens to reserve")


class ContextOptimizeResponse(BaseModel):
    """Response from context optimization."""

    items: list[ContextItem]
    total_tokens: int
    items_included: int
    items_dropped: int
    budget_used_percent: float
