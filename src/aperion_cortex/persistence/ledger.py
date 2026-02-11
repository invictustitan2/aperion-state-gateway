"""Audit Ledger - Immutable record of all agent runs.

The Training Ledger captures every agent execution for:
- Training data collection
- Debugging and audit trails
- Performance analysis
- Policy compliance verification

The Cortex is the SINGLE WRITER to this ledger, ensuring data immutability.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Captured ParseResult for ledger entries."""

    ok: bool
    strategy: str | None = None
    confidence: float | None = None
    error: str | None = None


@dataclass
class LedgerEntry:
    """A single training ledger entry.

    Captures all relevant information about an agent execution:
    - Identity (run_id, correlation_id)
    - Timing (timestamps, latency)
    - Agent context (agent name, method)
    - Input (prompt, context refs)
    - Output (raw output, parsed data)
    - Provider info (model, usage)
    - Policy decisions
    - Training labels (for future annotation)
    """

    # Identity
    run_id: str
    correlation_id: str | None = None

    # Timing
    timestamp_start: str = ""
    timestamp_end: str = ""
    latency_ms: float = 0.0

    # Agent context
    agent_name: str = ""
    method_name: str = ""

    # Input
    prompt: str = ""
    context_refs: list[str] = field(default_factory=list)
    input_metadata: dict[str, Any] = field(default_factory=dict)

    # Output
    raw_output: str = ""
    parsed_data: dict[str, Any] | None = None
    parse_result: ParseResult | None = None

    # Provider
    provider_name: str = ""
    model_name: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    # Policy
    policy_decisions: list[dict[str, Any]] = field(default_factory=list)

    # Training labels (empty until human review)
    label: str | None = None  # e.g., "good", "bad", "partial"
    gold: dict[str, Any] | None = None  # expected output for comparison
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert ParseResult to dict if present
        if self.parse_result:
            d["parse_result"] = asdict(self.parse_result)
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLedger:
    """Immutable audit ledger for agent execution records.

    The Cortex is the single writer to this ledger, ensuring:
    - Data immutability (append-only JSONL)
    - Centralized audit trail
    - Training data collection
    - Policy compliance verification

    Example:
        ledger = AuditLedger()

        # Start a run
        entry = ledger.start_run(
            agent_name="AnalystAgent",
            method_name="detect_code_smells",
            prompt="...",
            correlation_id="corr-123"
        )

        # ... agent executes ...

        # Complete the run
        ledger.complete_run(
            entry,
            raw_output="...",
            parsed_data={...},
            parse_result=ParseResult(ok=True, strategy="json_loads"),
            provider_name="workers_ai",
            model_name="llama-3.3-70b"
        )
    """

    def __init__(
        self,
        ledger_path: Path | str | None = None,
        auto_flush: bool = True,
    ):
        """Initialize the audit ledger.

        Args:
            ledger_path: Path to JSONL file. Defaults to data/audit_ledger.jsonl
            auto_flush: Whether to flush after each write
        """
        if ledger_path is None:
            ledger_path = Path.cwd() / "data" / "audit_ledger.jsonl"
        else:
            ledger_path = Path(ledger_path)

        self._ledger_path = ledger_path
        self._auto_flush = auto_flush
        self._entries: list[LedgerEntry] = []

        # Ensure directory exists
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def ledger_path(self) -> Path:
        """Get the ledger file path."""
        return self._ledger_path

    def start_run(
        self,
        agent_name: str,
        method_name: str,
        prompt: str,
        correlation_id: str | None = None,
        context_refs: list[str] | None = None,
        input_metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        """Start a new ledger entry for an agent run.

        Args:
            agent_name: Name of the executing agent
            method_name: Method/function being called
            prompt: Input prompt to the agent
            correlation_id: Request correlation ID
            context_refs: List of context references
            input_metadata: Additional input metadata

        Returns:
            LedgerEntry with run_id and start timestamp
        """
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        entry = LedgerEntry(
            run_id=run_id,
            correlation_id=correlation_id or f"corr_{uuid.uuid4().hex[:8]}",
            timestamp_start=datetime.now(timezone.utc).isoformat(),
            agent_name=agent_name,
            method_name=method_name,
            prompt=prompt,
            context_refs=context_refs or [],
            input_metadata=input_metadata or {},
        )

        return entry

    def complete_run(
        self,
        entry: LedgerEntry,
        raw_output: str,
        parsed_data: dict[str, Any] | None = None,
        parse_result: ParseResult | None = None,
        provider_name: str = "",
        model_name: str = "",
        usage: dict[str, Any] | None = None,
        policy_decisions: list[dict[str, Any]] | None = None,
    ) -> LedgerEntry:
        """Complete a ledger entry and write to file.

        Args:
            entry: The entry from start_run()
            raw_output: Raw LLM output string
            parsed_data: Parsed/structured data (if successful)
            parse_result: ParseResult record
            provider_name: LLM provider name
            model_name: Model identifier
            usage: Token usage stats
            policy_decisions: Any policy checks that ran

        Returns:
            The completed LedgerEntry
        """
        entry.timestamp_end = datetime.now(timezone.utc).isoformat()

        # Calculate latency
        start = datetime.fromisoformat(entry.timestamp_start)
        end = datetime.fromisoformat(entry.timestamp_end)
        entry.latency_ms = (end - start).total_seconds() * 1000

        entry.raw_output = raw_output
        entry.parsed_data = parsed_data
        entry.parse_result = parse_result
        entry.provider_name = provider_name
        entry.model_name = model_name
        entry.usage = usage or {}
        entry.policy_decisions = policy_decisions or []

        # Write to ledger
        self._write_entry(entry)
        self._entries.append(entry)

        logger.debug(f"Ledger entry written: {entry.run_id}")
        return entry

    def record_run(
        self,
        agent_name: str,
        method_name: str,
        prompt: str,
        raw_output: str,
        parsed_data: dict[str, Any] | None = None,
        parse_ok: bool = False,
        parse_strategy: str | None = None,
        parse_confidence: float | None = None,
        parse_error: str | None = None,
        provider_name: str = "",
        model_name: str = "",
        correlation_id: str | None = None,
        latency_ms: float | None = None,
        context_refs: list[str] | None = None,
        policy_decisions: list[dict[str, Any]] | None = None,
    ) -> LedgerEntry:
        """Convenience method to record a complete run in one call.

        This is the primary API for agents to log their executions
        via the POST /audit/record endpoint.
        """
        entry = self.start_run(
            agent_name=agent_name,
            method_name=method_name,
            prompt=prompt,
            correlation_id=correlation_id,
            context_refs=context_refs,
        )

        parse_result = ParseResult(
            ok=parse_ok,
            strategy=parse_strategy,
            confidence=parse_confidence,
            error=parse_error,
        )

        completed = self.complete_run(
            entry,
            raw_output=raw_output,
            parsed_data=parsed_data,
            parse_result=parse_result,
            provider_name=provider_name,
            model_name=model_name,
            policy_decisions=policy_decisions,
        )

        # Override latency if provided (for cases where timing is external)
        if latency_ms is not None:
            completed.latency_ms = latency_ms

        return completed

    def _write_entry(self, entry: LedgerEntry) -> None:
        """Write a single entry to the ledger file."""
        with open(self._ledger_path, "a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")
            if self._auto_flush:
                f.flush()

    def get_entry(self, run_id: str) -> LedgerEntry | None:
        """Get an entry by run_id (searches file).

        Args:
            run_id: Run identifier to search for

        Returns:
            LedgerEntry if found, None otherwise
        """
        if not self._ledger_path.exists():
            return None

        with open(self._ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("run_id") == run_id:
                    return self._dict_to_entry(data)
        return None

    def get_recent(self, n: int = 10) -> list[LedgerEntry]:
        """Get the N most recent entries.

        Args:
            n: Number of entries to return

        Returns:
            List of most recent LedgerEntries
        """
        if not self._ledger_path.exists():
            return []

        entries = []
        with open(self._ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entries.append(json.loads(line))

        return [self._dict_to_entry(d) for d in entries[-n:]]

    def get_by_agent(self, agent_name: str, limit: int = 100) -> list[LedgerEntry]:
        """Get entries for a specific agent.

        Args:
            agent_name: Agent name to filter by
            limit: Maximum entries to return

        Returns:
            List of matching LedgerEntries
        """
        if not self._ledger_path.exists():
            return []

        entries = []
        with open(self._ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("agent_name") == agent_name:
                    entries.append(self._dict_to_entry(data))
                    if len(entries) >= limit:
                        break

        return entries

    def _dict_to_entry(self, data: dict[str, Any]) -> LedgerEntry:
        """Convert a dict back to LedgerEntry."""
        # Handle parse_result specially
        pr_data = data.pop("parse_result", None)
        parse_result = None
        if pr_data:
            parse_result = ParseResult(**pr_data)

        entry = LedgerEntry(**data)
        entry.parse_result = parse_result
        return entry

    def count(self) -> int:
        """Count total entries in ledger."""
        if not self._ledger_path.exists():
            return 0
        with open(self._ledger_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def stats(self) -> dict[str, Any]:
        """Get ledger statistics.

        Returns:
            Dict with entry counts, agent breakdown, success rate, etc.
        """
        if not self._ledger_path.exists():
            return {
                "total_entries": 0,
                "agents": {},
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
            }

        entries = []
        with open(self._ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        if not entries:
            return {
                "total_entries": 0,
                "agents": {},
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
            }

        # Compute stats
        agents: dict[str, int] = {}
        successful = 0
        total_latency = 0.0

        for entry in entries:
            agent = entry.get("agent_name", "unknown")
            agents[agent] = agents.get(agent, 0) + 1

            pr = entry.get("parse_result", {})
            if pr and pr.get("ok"):
                successful += 1

            total_latency += entry.get("latency_ms", 0)

        return {
            "total_entries": len(entries),
            "agents": agents,
            "success_rate": successful / len(entries) if entries else 0.0,
            "avg_latency_ms": total_latency / len(entries) if entries else 0.0,
        }
