"""FastAPI router for the Cortex service.

Implements all API endpoints for:
- State management (/state/*)
- Vector search (/vector/*)
- Database introspection (/db/*)
- Filesystem access (/fs/*)
- Audit logging (/audit/*)
- Event streaming (/events/*)
"""

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from .auth import TokenInfo, make_scope_dependency
from .models import (
    AuditLogEntry,
    AuditRecordRequest,
    AuditRecordResponse,
    ContextOptimizeRequest,
    ContextOptimizeResponse,
    Database,
    DatabaseSample,
    DatabaseSchema,
    DiscoverResponse,
    FileEntry,
    FileReadResponse,
    Service,
    ServiceHealth,
    SnapshotResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)


def build_router(service: "CortexService") -> APIRouter:
    """Build the Cortex API router.

    Args:
        service: The CortexService instance

    Returns:
        Configured APIRouter
    """
    router = APIRouter()

    # Scope dependencies
    context_scope = make_scope_dependency("context")
    vector_scope = make_scope_dependency("vector")
    database_scope = make_scope_dependency("database")
    filesystem_scope = make_scope_dependency("filesystem")
    services_scope = make_scope_dependency("services")
    state_scope = make_scope_dependency("state")
    audit_scope = make_scope_dependency("audit")

    # -----------------------------------------------------------------------
    # State Endpoints
    # -----------------------------------------------------------------------

    @router.get("/state/discover", response_model=DiscoverResponse)
    def state_discover(
        auth: Annotated[TokenInfo, Depends(state_scope)],
    ) -> DiscoverResponse:
        """Discover available resources."""
        return service.state_discover()

    @router.get("/state/snapshot", response_model=SnapshotResponse)
    def state_snapshot(
        auth: Annotated[TokenInfo, Depends(context_scope)],
        since: str | None = Query(default=None),
    ) -> SnapshotResponse:
        """Get signed context snapshot.

        Args:
            since: ISO timestamp - return fresh if unchanged since this time
        """
        return service.assemble_context_pack(since=since)

    # -----------------------------------------------------------------------
    # Vector Search Endpoints
    # -----------------------------------------------------------------------

    @router.post("/vector/search", response_model=VectorSearchResponse)
    def vector_search(
        request: VectorSearchRequest,
        auth: Annotated[TokenInfo, Depends(vector_scope)],
    ) -> VectorSearchResponse:
        """Perform semantic vector search."""
        return service.vector_search(request)

    # -----------------------------------------------------------------------
    # Database Endpoints
    # -----------------------------------------------------------------------

    @router.get("/db/list")
    def db_list(
        auth: Annotated[TokenInfo, Depends(database_scope)],
    ) -> list[Database]:
        """List registered databases."""
        return service.list_databases()

    @router.get("/db/schema/{name}")
    def db_schema(
        name: str,
        auth: Annotated[TokenInfo, Depends(database_scope)],
    ) -> DatabaseSchema:
        """Get database schema."""
        return service.describe_database(name)

    @router.get("/db/sample", response_model=DatabaseSample)
    def db_sample(
        name: str,
        table: str,
        auth: Annotated[TokenInfo, Depends(database_scope)],
        limit: int = Query(default=10, ge=1, le=1000),
    ) -> DatabaseSample:
        """Sample rows from a database table."""
        return service.sample_database(name, table, limit)

    # -----------------------------------------------------------------------
    # Service Endpoints
    # -----------------------------------------------------------------------

    @router.get("/svc/list")
    def svc_list(
        auth: Annotated[TokenInfo, Depends(services_scope)],
    ) -> list[Service]:
        """List registered services."""
        return service.list_services()

    @router.get("/svc/health")
    def svc_health(
        auth: Annotated[TokenInfo, Depends(services_scope)],
    ) -> list[ServiceHealth]:
        """Get health status for all services."""
        return service.service_health()

    # -----------------------------------------------------------------------
    # Filesystem Endpoints
    # -----------------------------------------------------------------------

    @router.get("/fs/list")
    def fs_list(
        auth: Annotated[TokenInfo, Depends(filesystem_scope)],
        path: str | None = Query(default=None),
    ) -> list[FileEntry]:
        """List filesystem entries."""
        return service.list_files(path)

    @router.get("/fs/read_text", response_model=FileReadResponse)
    def fs_read_text(
        path: str,
        auth: Annotated[TokenInfo, Depends(filesystem_scope)],
        num_bytes: int = Query(default=4096, ge=256, le=8192),
    ) -> FileReadResponse:
        """Read text file content."""
        return service.read_file_text(path, num_bytes)

    # -----------------------------------------------------------------------
    # Audit Endpoints (Constitution E - Centralized Ledger)
    # -----------------------------------------------------------------------

    @router.get("/audit/log")
    def audit_log(
        auth: Annotated[TokenInfo, Depends(state_scope)],
    ) -> dict[str, Any]:
        """Get recent audit log entries."""
        entries = [entry.model_dump() for entry in service.audit_log()]
        return {"entries": entries}

    @router.post("/audit/record", response_model=AuditRecordResponse)
    def audit_record(
        request: AuditRecordRequest,
        auth: Annotated[TokenInfo, Depends(audit_scope)],
    ) -> AuditRecordResponse:
        """Record an agent execution to the audit ledger.

        This is the primary interface for agents to log their execution
        results. The Cortex is the single writer to the training ledger.
        """
        return service.record_audit_entry(request)

    @router.get("/audit/stats")
    def audit_stats(
        auth: Annotated[TokenInfo, Depends(audit_scope)],
    ) -> dict[str, Any]:
        """Get audit ledger statistics."""
        return service.audit_stats()

    # -----------------------------------------------------------------------
    # Context Optimization Endpoints
    # -----------------------------------------------------------------------

    @router.post("/context/optimize", response_model=ContextOptimizeResponse)
    def context_optimize(
        request: ContextOptimizeRequest,
        auth: Annotated[TokenInfo, Depends(context_scope)],
    ) -> ContextOptimizeResponse:
        """Optimize context items for LLM consumption.

        Accepts a list of items with content and priority, returns
        an optimized list that fits within the token budget.
        """
        return service.optimize_context(request)

    # -----------------------------------------------------------------------
    # Event Streaming Endpoints
    # -----------------------------------------------------------------------

    @router.get("/events/stream")
    async def events_stream(
        auth: Annotated[TokenInfo, Depends(state_scope)],
    ) -> StreamingResponse:
        """Server-Sent Events stream for real-time updates."""

        async def event_iterator() -> AsyncIterator[str]:
            async for payload in service.subscribe_events():
                yield payload

        return StreamingResponse(event_iterator(), media_type="text/event-stream")

    return router


# Type hint for forward reference
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import CortexService
