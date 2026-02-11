# Event Sourcing & Audit Logging Best Practices

> Reference documentation for immutable event ledgers
> Sources: eventsourcing.readthedocs.io, learn.microsoft.com, dzone.com, softwarepatternslexicon.com

---

## 1. Core Principles

### 1.1 Immutability

Events are **append-only**. Never update or delete events.

```python
class AuditLedger:
    """Append-only audit ledger."""
    
    def record(self, entry: LedgerEntry) -> None:
        """Append entry. Never modifies existing entries."""
        with open(self._log_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
    
    def delete(self, entry_id: str) -> None:
        """NEVER IMPLEMENT THIS."""
        raise NotImplementedError("Ledger is immutable")
```

### 1.2 Event as Source of Truth

The event log is the canonical record. All other state is derived.

```
Events ─────▶ Current State
       ─────▶ Analytics
       ─────▶ Audit Reports
       ─────▶ Training Data
```

### 1.3 Meaningful Event Design

Events should describe **what happened**, not **what to do**:

```python
# ✅ Good: Describes state change
@dataclass
class AgentCompletedTask:
    agent_name: str
    task_id: str
    result: str
    latency_ms: float
    timestamp: datetime

# ❌ Bad: Command, not event
@dataclass
class RunAgent:
    agent_name: str
    input: str
```

---

## 2. Event Structure

### 2.1 Cortex LedgerEntry

```python
@dataclass
class LedgerEntry:
    # Identity
    run_id: str                    # Unique ID for this entry
    correlation_id: str | None     # Links related events
    
    # Timing
    timestamp_start: str           # ISO8601
    timestamp_end: str             # ISO8601
    latency_ms: float
    
    # Agent Context
    agent_name: str
    method_name: str
    
    # Input
    prompt: str
    context_refs: list[str]        # References to context used
    input_metadata: dict
    
    # Output
    raw_output: str
    parsed_data: dict | None
    parse_result: ParseResult | None
    
    # Provider
    provider_name: str
    model_name: str
    usage: dict                    # Token counts, costs
    
    # Policy
    policy_decisions: list[dict]   # RBAC, rate limits, etc.
    
    # Training
    training_labels: dict          # For future annotation
```

### 2.2 Required Fields for Audit

At minimum, every entry must have:

| Field | Purpose |
|-------|---------|
| `run_id` | Unique identifier |
| `timestamp` | When it happened |
| `actor` | Who/what caused it |
| `action` | What happened |
| `outcome` | Success/failure |

---

## 3. Storage Patterns

### 3.1 JSONL (Current)

Simple, human-readable, append-friendly:

```jsonl
{"run_id": "r1", "agent_name": "analyst", "timestamp": "2026-02-08T04:00:00Z", ...}
{"run_id": "r2", "agent_name": "coder", "timestamp": "2026-02-08T04:00:01Z", ...}
```

**Pros**: Simple, no dependencies, grep-friendly
**Cons**: No indexing, linear scan for queries

### 3.2 SQLite (Better for Queries)

```sql
CREATE TABLE audit_log (
    id TEXT PRIMARY KEY,
    correlation_id TEXT,
    timestamp TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    action TEXT NOT NULL,
    payload TEXT NOT NULL,  -- JSON
    signature TEXT,         -- For tamper detection
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_agent ON audit_log(agent_name);
CREATE INDEX idx_audit_correlation ON audit_log(correlation_id);
```

### 3.3 PostgreSQL (Production)

```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id UUID,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_name TEXT NOT NULL,
    action TEXT NOT NULL,
    payload JSONB NOT NULL,
    signature TEXT,
    
    -- Partitioning for scale
    created_at DATE NOT NULL DEFAULT CURRENT_DATE
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE audit_log_2026_02 PARTITION OF audit_log
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
```

---

## 4. Tamper-Proofing

### 4.1 Hash Chaining

Each entry includes hash of previous entry:

```python
import hashlib

def compute_entry_hash(entry: dict, previous_hash: str) -> str:
    """Compute hash including previous entry's hash."""
    content = json.dumps(entry, sort_keys=True) + previous_hash
    return hashlib.sha256(content.encode()).hexdigest()

def append_entry(entry: dict, ledger_path: Path) -> str:
    """Append entry with hash chain."""
    # Get previous hash
    previous_hash = get_last_hash(ledger_path) or "GENESIS"
    
    # Add hash to entry
    entry["previous_hash"] = previous_hash
    entry["hash"] = compute_entry_hash(entry, previous_hash)
    
    # Append
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    return entry["hash"]
```

### 4.2 Verification

```python
def verify_ledger_integrity(ledger_path: Path) -> bool:
    """Verify hash chain is intact."""
    previous_hash = "GENESIS"
    
    with open(ledger_path) as f:
        for line in f:
            entry = json.loads(line)
            
            # Verify previous hash matches
            if entry["previous_hash"] != previous_hash:
                return False
            
            # Verify entry hash
            expected_hash = compute_entry_hash(
                {k: v for k, v in entry.items() if k != "hash"},
                previous_hash
            )
            if entry["hash"] != expected_hash:
                return False
            
            previous_hash = entry["hash"]
    
    return True
```

### 4.3 Digital Signatures

For highest assurance:

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

def sign_entry(entry: dict, private_key: rsa.RSAPrivateKey) -> str:
    """Sign entry with RSA private key."""
    content = json.dumps(entry, sort_keys=True).encode()
    signature = private_key.sign(
        content,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode()
```

---

## 5. Snapshotting

### 5.1 Purpose

Avoid replaying all events for state reconstruction:

```python
class SnapshotStore:
    def save_snapshot(self, state: dict, as_of_event_id: str) -> None:
        """Save state snapshot at a point in time."""
        snapshot = {
            "state": state,
            "as_of_event_id": as_of_event_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._store.save(snapshot)
    
    def load_state(self) -> dict:
        """Load state from snapshot + replay recent events."""
        snapshot = self._store.get_latest_snapshot()
        if snapshot:
            state = snapshot["state"]
            events_after = self._ledger.get_events_after(snapshot["as_of_event_id"])
        else:
            state = {}
            events_after = self._ledger.get_all_events()
        
        # Replay events to update state
        for event in events_after:
            state = apply_event(state, event)
        
        return state
```

### 5.2 Snapshot Frequency

Balance between storage cost and replay time:

| Scenario | Frequency |
|----------|-----------|
| High write volume | Every 1000 events |
| Low write volume | Daily |
| Large state | More frequent |
| Small state | Less frequent |

---

## 6. Retention & Rotation

### 6.1 Log Rotation

```python
import logging.handlers

def setup_rotating_ledger(path: Path, max_bytes: int = 100_000_000) -> None:
    """Set up rotating ledger with 100MB files."""
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=10,
    )
    return handler
```

### 6.2 Retention Policy

```python
@dataclass
class RetentionPolicy:
    hot_days: int = 7      # Full detail, fast access
    warm_days: int = 30    # Compressed, slower access
    cold_days: int = 365   # Archived, restore on demand
    
    def apply(self, ledger: AuditLedger) -> None:
        now = datetime.now(timezone.utc)
        
        for entry in ledger.scan():
            age_days = (now - entry.timestamp).days
            
            if age_days > self.cold_days:
                self._archive_to_cold(entry)
            elif age_days > self.warm_days:
                self._compress_to_warm(entry)
```

---

## 7. Cortex Implementation Gaps

### Current State
- ✅ Append-only JSONL
- ✅ Structured LedgerEntry
- ✅ Single writer enforcement

### Missing
- ❌ Hash chaining (tamper-proofing)
- ❌ Snapshotting
- ❌ Retention/rotation
- ❌ Verification API
- ❌ Index for queries

### Recommended Additions

```python
class AuditLedger:
    # Add these methods:
    
    def verify_integrity(self) -> bool:
        """Verify hash chain is intact."""
        ...
    
    def get_entries_by_correlation(self, correlation_id: str) -> list[LedgerEntry]:
        """Query entries by correlation ID."""
        ...
    
    def create_snapshot(self) -> str:
        """Create state snapshot, return snapshot ID."""
        ...
    
    def rotate(self) -> None:
        """Rotate log files per retention policy."""
        ...
```

---

## References

- https://eventsourcing.readthedocs.io/en/stable/
- https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing
- https://dzone.com/articles/event-sourcing-explained-building-robust-systems
- https://softwarepatternslexicon.com/bitemporal-modeling/audit-logging-patterns/event-sourcing/
- https://www.kurrent.io/blog/event-sourcing-audit
