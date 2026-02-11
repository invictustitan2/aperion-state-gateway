"""Core Cortex service - business logic and state management."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from fastapi import HTTPException, status

from ..memory.context import ContextEngine, ContextItem, PackingResult
from ..memory.hot_state import HotStateManager
from ..memory.vector_store import VectorStore, VectorHit as MemoryVectorHit
from ..persistence.database import StateRepository
from ..persistence.ledger import AuditLedger, ParseResult
from .config import CortexConfig
from .models import (
    AuditLogEntry,
    AuditRecordRequest,
    AuditRecordResponse,
    ContextOptimizeRequest,
    ContextOptimizeResponse,
    ContextItem as ModelContextItem,
    ContextPack,
    Database,
    DatabaseSample,
    DatabaseSchema,
    DiscoverResponse,
    FileEntry,
    FileReadResponse,
    Resource,
    Service,
    ServiceHealth,
    SnapshotResponse,
    VectorHit,
    VectorSearchRequest,
    VectorSearchResponse,
)


class CortexService:
    """Core Cortex service implementing business logic.

    Provides:
    - Context pack assembly and signing
    - Vector search integration
    - Database/filesystem access
    - Audit logging
    - Event streaming
    """

    def __init__(
        self,
        config: CortexConfig,
        *,
        resource_provider: Callable[[], list[Resource]] | None = None,
        hot_fact_provider: Callable[[], list[str]] | None = None,
        summary_provider: Callable[[list[Resource]], str] | None = None,
        vector_store: VectorStore | None = None,
        database_provider: Callable[[], list[Database]] | None = None,
        schema_provider: Callable[[str], dict[str, Any]] | None = None,
        database_sampler: Callable[[str, str, int], DatabaseSample] | None = None,
        service_provider: Callable[[], list[Service]] | None = None,
        service_health_provider: Callable[[], list[ServiceHealth]] | None = None,
        fs_list_provider: Callable[[str | None], list[FileEntry]] | None = None,
        fs_read_provider: Callable[[str, int], FileReadResponse] | None = None,
        state_repository: StateRepository | None = None,
        audit_ledger: AuditLedger | None = None,
        hot_state: HotStateManager | None = None,
        time_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config

        # Adapters (with defaults)
        self._resource_provider = resource_provider or (lambda: [])
        self._hot_fact_provider = hot_fact_provider or (lambda: [])
        self._summary_provider = summary_provider or (
            lambda r: f"{len(r)} resources registered."
        )
        self._vector_store = vector_store
        self._database_provider = database_provider or (lambda: [])
        self._schema_provider = schema_provider or (lambda n: {"tables": []})
        self._database_sampler = database_sampler or (
            lambda n, t, l: DatabaseSample(
                database=n, table=t, limit=l, rows=[], truncated=False
            )
        )
        self._service_provider = service_provider or (lambda: [])
        self._service_health_provider = service_health_provider or (
            lambda: [
                ServiceHealth(
                    name="cortex",
                    status="ok",
                    latency_ms=0,
                    last_checked=datetime.now(timezone.utc),
                )
            ]
        )
        self._fs_list_provider = fs_list_provider or (lambda p: [])
        self._fs_read_provider = fs_read_provider or (
            lambda p, b: FileReadResponse(
                path=p, bytes_returned=0, text="", truncated=False
            )
        )

        # Persistence
        self._repository = state_repository
        self._ledger = audit_ledger or AuditLedger()
        self._hot_state = hot_state or HotStateManager()

        # Time provider
        self._time_provider = time_provider or (lambda: datetime.now(timezone.utc))

        # Context caching
        self._context_cache: tuple[ContextPack, float] | None = None

        # In-memory audit log (recent entries)
        self._audit_log: list[AuditLogEntry] = []

        # Event subscribers
        self._event_queues: list[asyncio.Queue[str]] = []

    # -----------------------------------------------------------------------
    # Context Pack Assembly
    # -----------------------------------------------------------------------

    def _serialize_and_sign(self, context: dict[str, Any]) -> tuple[str, str]:
        """Serialize and sign a context dictionary."""
        body = json.dumps(context, separators=(",", ":"), sort_keys=True, default=str)
        signature = hashlib.sha256(
            (body + self.config.signing_key).encode("utf-8")
        ).hexdigest()
        return body, signature

    def assemble_context_pack(
        self, force_refresh: bool = False, since: str | None = None
    ) -> SnapshotResponse:
        """Assemble a signed context pack.

        Args:
            force_refresh: Force regeneration even if cached
            since: ISO timestamp - force refresh if cached is older

        Returns:
            Signed SnapshotResponse
        """
        now = self._time_provider()

        # Check cache
        if not force_refresh and self._context_cache:
            cached, cached_at = self._context_cache
            expires_at = cached.generated_at + timedelta(seconds=cached.ttl_seconds)

            if now < expires_at:
                # Check 'since' parameter
                if since:
                    try:
                        since_dt = self._parse_datetime(since)
                        if cached.generated_at > since_dt:
                            return SnapshotResponse(
                                **cached.model_dump(), freshness="cached"
                            )
                    except ValueError:
                        pass
                else:
                    return SnapshotResponse(**cached.model_dump(), freshness="cached")

        # Build fresh context
        resources = self._resource_provider()
        summary = self._summary_provider(resources)
        hot_facts = self._hot_fact_provider()

        context_dict = {
            "generated_at": now.isoformat(timespec="seconds"),
            "ttl_seconds": self.config.default_ttl_seconds,
            "summary": summary,
            "resources": [r.model_dump() for r in resources],
            "hot_facts": hot_facts,
        }

        serialized, signature = self._serialize_and_sign(context_dict)
        context_size = len(serialized.encode("utf-8"))

        # Trim if over size cap
        if context_size > self.config.context_size_cap:
            context_dict = self._trim_context(context_dict)
            serialized, signature = self._serialize_and_sign(context_dict)
            context_size = len(serialized.encode("utf-8"))

            if context_size > self.config.context_size_cap:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Context pack exceeds size budget even after trimming",
                )

        # Build typed models
        typed_resources = [Resource(**r) for r in context_dict.get("resources", [])]

        context_pack = ContextPack(
            generated_at=now,
            ttl_seconds=self.config.default_ttl_seconds,
            summary=context_dict["summary"],
            resources=typed_resources,
            hot_facts=context_dict.get("hot_facts", []),
            signature=signature,
        )

        # Cache it
        self._context_cache = (context_pack, time.time())

        # Emit event
        self._emit_event("state.snapshot_generated", {"signature": signature})

        return SnapshotResponse(**context_pack.model_dump(), freshness="fresh")

    def _trim_context(self, context_dict: dict[str, Any]) -> dict[str, Any]:
        """Trim context to fit within size cap."""
        trimmed = dict(context_dict)
        hot_facts = list(trimmed.get("hot_facts", []))
        resources = list(trimmed.get("resources", []))

        def get_size(obj: dict) -> int:
            return len(
                json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str).encode()
            )

        while hot_facts and get_size(trimmed) > self.config.context_size_cap:
            hot_facts.pop()
            trimmed["hot_facts"] = hot_facts

        while resources and get_size(trimmed) > self.config.context_size_cap:
            resources.pop()
            trimmed["resources"] = resources

        return trimmed

    def _parse_datetime(self, value: str) -> datetime:
        """Parse ISO datetime, handling trailing 'Z'."""
        if value.endswith("Z"):
            value = value[:-1]
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    # -----------------------------------------------------------------------
    # Context Optimization
    # -----------------------------------------------------------------------

    def optimize_context(self, request: ContextOptimizeRequest) -> ContextOptimizeResponse:
        """Optimize context items for LLM consumption."""
        engine = ContextEngine(
            max_tokens=request.max_tokens,
            reserve_tokens=request.reserve_tokens,
        )

        for item in request.items:
            engine.add(
                content=item.content,
                priority=item.priority,
                category=item.category,
                metadata=item.metadata,
            )

        result = engine.pack()

        # Convert back to response model
        response_items = [
            ModelContextItem(
                content=item.content,
                priority=item.priority,
                category=item.category,
                metadata=item.metadata,
            )
            for item in result.items
        ]

        return ContextOptimizeResponse(
            items=response_items,
            total_tokens=result.total_tokens,
            items_included=result.items_included,
            items_dropped=result.items_dropped,
            budget_used_percent=result.budget_used_percent,
        )

    # -----------------------------------------------------------------------
    # Vector Search
    # -----------------------------------------------------------------------

    def vector_search(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """Perform vector search."""
        if self._vector_store is None:
            return VectorSearchResponse(hits=[])

        start = time.time()
        hits = self._vector_store.search_sync(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata or None,
        )
        search_time = (time.time() - start) * 1000

        # Convert to response model
        response_hits = [
            VectorHit(
                doc_id=hit.doc_id,
                score=hit.score,
                uri=hit.uri,
                summary=hit.summary,
                metadata=hit.metadata,
            )
            for hit in hits
        ]

        # Log the invocation
        self._log_tool_invocation(
            "vector.search",
            "vector",
            {"query": request.query, "top_k": request.top_k},
            {"hits": len(response_hits)},
        )

        return VectorSearchResponse(
            hits=response_hits,
            search_time_ms=search_time,
        )

    # -----------------------------------------------------------------------
    # Database Methods
    # -----------------------------------------------------------------------

    def list_databases(self) -> list[Database]:
        """List registered databases."""
        return self._database_provider()

    def describe_database(self, name: str) -> DatabaseSchema:
        """Get database schema."""
        start = time.time()
        result = self._schema_provider(name)
        tables = result.get("tables", [])

        schema_hash = hashlib.sha256(
            json.dumps(tables, sort_keys=True, default=str).encode()
        ).hexdigest()

        self._log_tool_invocation(
            "db.describe",
            "database",
            {"name": name},
            {"table_count": len(tables)},
        )

        return DatabaseSchema(
            tables=tables,
            schema_hash=schema_hash,
            computed_at=self._time_provider(),
        )

    def sample_database(self, name: str, table: str, limit: int) -> DatabaseSample:
        """Sample rows from a database table."""
        sample = self._database_sampler(name, table, limit)

        self._log_tool_invocation(
            "db.sample",
            "database",
            {"name": name, "table": table, "limit": limit},
            {"row_count": len(sample.rows)},
        )

        return sample

    # -----------------------------------------------------------------------
    # Service Methods
    # -----------------------------------------------------------------------

    def list_services(self) -> list[Service]:
        """List registered services."""
        return self._service_provider()

    def service_health(self) -> list[ServiceHealth]:
        """Get service health status."""
        return self._service_health_provider()

    # -----------------------------------------------------------------------
    # Filesystem Methods
    # -----------------------------------------------------------------------

    def list_files(self, path: str | None) -> list[FileEntry]:
        """List filesystem entries."""
        entries = self._fs_list_provider(path)

        self._log_tool_invocation(
            "fs.list",
            "filesystem",
            {"path": path or ""},
            {"entry_count": len(entries)},
        )

        return entries

    def read_file_text(self, path: str, byte_limit: int) -> FileReadResponse:
        """Read text file content."""
        response = self._fs_read_provider(path, byte_limit)

        self._log_tool_invocation(
            "fs.read_text",
            "filesystem",
            {"path": path, "bytes": byte_limit},
            {"bytes_returned": response.bytes_returned},
        )

        return response

    # -----------------------------------------------------------------------
    # Discovery
    # -----------------------------------------------------------------------

    def state_discover(self) -> DiscoverResponse:
        """Discover available resources."""
        return DiscoverResponse(resources=self._resource_provider())

    # -----------------------------------------------------------------------
    # Audit Logging
    # -----------------------------------------------------------------------

    def _log_tool_invocation(
        self,
        name: str,
        scope: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        error: str | None = None,
    ) -> None:
        """Log a tool invocation to the audit log."""
        entry = AuditLogEntry(
            tool_name=name,
            scope=scope,
            input_bytes=len(json.dumps(input_payload).encode()),
            output_bytes=len(json.dumps(output_payload).encode()),
            latency_ms=0.0,
            resource_ids=[],
            status="error" if error else "ok",
            error=error,
            timestamp=self._time_provider(),
        )

        self._audit_log.append(entry)
        if len(self._audit_log) > self.config.audit_log_limit:
            self._audit_log = self._audit_log[-self.config.audit_log_limit:]

        self._emit_event(
            "state.tool_invoked",
            {"tool_name": name, "scope": scope, "status": entry.status},
        )

    def audit_log(self) -> list[AuditLogEntry]:
        """Get recent audit log entries."""
        return list(self._audit_log)

    def record_audit_entry(self, request: AuditRecordRequest) -> AuditRecordResponse:
        """Record an agent execution to the permanent audit ledger."""
        entry = self._ledger.record_run(
            agent_name=request.agent_name,
            method_name=request.method_name,
            prompt=request.prompt,
            raw_output=request.raw_output,
            parsed_data=request.parsed_data,
            parse_ok=request.parse_ok,
            parse_strategy=request.parse_strategy,
            parse_confidence=request.parse_confidence,
            parse_error=request.parse_error,
            provider_name=request.provider_name,
            model_name=request.model_name,
            correlation_id=request.correlation_id,
            latency_ms=request.latency_ms,
            context_refs=request.context_refs,
            policy_decisions=request.policy_decisions,
        )

        self._emit_event(
            "audit.entry_recorded",
            {
                "run_id": entry.run_id,
                "agent_name": entry.agent_name,
                "method_name": entry.method_name,
            },
        )

        return AuditRecordResponse(
            run_id=entry.run_id,
            recorded_at=self._time_provider(),
            ledger_path=str(self._ledger.ledger_path),
        )

    def audit_stats(self) -> dict[str, Any]:
        """Get audit ledger statistics."""
        return self._ledger.stats()

    # -----------------------------------------------------------------------
    # Event Streaming
    # -----------------------------------------------------------------------

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit an event to all subscribers."""
        event_data = json.dumps({
            "event_type": event_type,
            "payload": payload,
            "timestamp": time.time(),
        })
        message = f"data: {event_data}\n\n"

        for queue in list(self._event_queues):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

    async def subscribe_events(self) -> AsyncIterator[str]:
        """Subscribe to SSE event stream."""
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._event_queues.append(queue)

        try:
            while True:
                message = await queue.get()
                yield message
        finally:
            self._event_queues.remove(queue)
