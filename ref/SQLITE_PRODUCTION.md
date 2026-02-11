# SQLite Production Best Practices

> Reference documentation for production SQLite deployment
> Sources: sqlite.org, markaicode.com, thelinuxcode.com, nerdleveltech.com

---

## 1. WAL Mode Configuration

### 1.1 Enable WAL Mode

WAL (Write-Ahead Logging) is essential for concurrent read/write:

```python
import sqlite3

def create_production_connection(db_path: str) -> sqlite3.Connection:
    """Create production-ready SQLite connection."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    
    # Enable WAL mode
    conn.execute("PRAGMA journal_mode=WAL;")
    
    # Normal sync is safe for most workloads
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    # Busy timeout in milliseconds
    conn.execute("PRAGMA busy_timeout=5000;")
    
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys=ON;")
    
    # Memory-mapped I/O for performance
    conn.execute("PRAGMA mmap_size=268435456;")  # 256MB
    
    return conn
```

### 1.2 WAL Benefits

| Mode | Readers | Writers | Durability |
|------|---------|---------|------------|
| DELETE | Blocked during write | 1 | High |
| WAL | Not blocked | 1 | High |
| MEMORY | Not blocked | 1 | None |

### 1.3 WAL Considerations

- WAL files can grow large under heavy writes
- Must checkpoint periodically
- All access must be on same filesystem (no NFS)
- Keep `.db`, `.db-wal`, `.db-shm` files together

---

## 2. Connection Management

### 2.1 Connection Pooling

For multi-threaded applications:

```python
from queue import Queue
from threading import Lock
from contextlib import contextmanager

class SQLitePool:
    """Simple connection pool for SQLite."""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self._db_path = db_path
        self._pool = Queue(maxsize=pool_size)
        self._lock = Lock()
        
        # Pre-create connections
        for _ in range(pool_size):
            conn = create_production_connection(db_path)
            self._pool.put(conn)
    
    @contextmanager
    def get_connection(self):
        conn = self._pool.get(timeout=5.0)
        try:
            yield conn
        finally:
            self._pool.put(conn)
    
    def close_all(self):
        while not self._pool.empty():
            conn = self._pool.get_nowait()
            conn.close()
```

### 2.2 Async with aiosqlite

For async applications:

```python
import aiosqlite

async def get_async_connection(db_path: str) -> aiosqlite.Connection:
    """Create async SQLite connection."""
    conn = await aiosqlite.connect(db_path)
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA synchronous=NORMAL;")
    await conn.execute("PRAGMA busy_timeout=5000;")
    return conn
```

---

## 3. Transaction Patterns

### 3.1 Keep Transactions Short

```python
# ✅ Good: Short transaction
def insert_record(conn, data):
    with conn:  # Auto-commit/rollback
        conn.execute("INSERT INTO records VALUES (?)", (data,))

# ❌ Bad: Long transaction
def bad_insert(conn, data_list):
    cursor = conn.cursor()
    cursor.execute("BEGIN")
    for data in data_list:  # Could take minutes!
        cursor.execute("INSERT INTO records VALUES (?)", (data,))
    cursor.execute("COMMIT")
```

### 3.2 Batch Operations

```python
def batch_insert(conn, records: list[dict]) -> None:
    """Batch insert with single transaction."""
    with conn:
        conn.executemany(
            "INSERT INTO records (id, data) VALUES (:id, :data)",
            records
        )
```

### 3.3 Retry on Lock

```python
import time
from sqlite3 import OperationalError

def execute_with_retry(
    conn,
    sql: str,
    params: tuple = (),
    max_retries: int = 3,
    backoff: float = 0.1
) -> None:
    """Execute with exponential backoff on lock."""
    for attempt in range(max_retries):
        try:
            with conn:
                conn.execute(sql, params)
            return
        except OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise
```

---

## 4. Checkpointing

### 4.1 Manual Checkpoint

```python
def checkpoint(conn) -> dict:
    """Checkpoint WAL to main database."""
    result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
    return {
        "blocked": result[0],
        "pages_written": result[1],
        "pages_remaining": result[2],
    }
```

### 4.2 Auto-Checkpoint Tuning

```python
# Checkpoint every 1000 pages (default is 1000)
conn.execute("PRAGMA wal_autocheckpoint=1000;")

# Disable auto-checkpoint (manual only)
conn.execute("PRAGMA wal_autocheckpoint=0;")
```

### 4.3 Scheduled Checkpointing

```python
import threading

class CheckpointScheduler:
    """Periodic checkpoint scheduler."""
    
    def __init__(self, conn, interval_seconds: int = 300):
        self._conn = conn
        self._interval = interval_seconds
        self._timer = None
    
    def start(self):
        self._schedule_next()
    
    def _schedule_next(self):
        self._timer = threading.Timer(self._interval, self._checkpoint)
        self._timer.daemon = True
        self._timer.start()
    
    def _checkpoint(self):
        try:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
        finally:
            self._schedule_next()
    
    def stop(self):
        if self._timer:
            self._timer.cancel()
```

---

## 5. Indexing

### 5.1 Index Strategy

```sql
-- Primary lookup patterns
CREATE INDEX idx_sessions_active ON sessions(last_active);
CREATE INDEX idx_snapshots_session ON state_snapshots(session_id);

-- Composite for common queries
CREATE INDEX idx_snapshots_session_time 
    ON state_snapshots(session_id, created_at DESC);

-- Partial index for active records only
CREATE INDEX idx_active_sessions 
    ON sessions(id) 
    WHERE last_active > datetime('now', '-1 hour');
```

### 5.2 Query Analysis

```python
def analyze_query(conn, sql: str) -> list[dict]:
    """Get query plan for optimization."""
    plan = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    return [{"id": r[0], "parent": r[1], "detail": r[3]} for r in plan]
```

---

## 6. Monitoring

### 6.1 Database Statistics

```python
def get_db_stats(conn) -> dict:
    """Get database statistics."""
    stats = {}
    
    # Page count and size
    stats["page_count"] = conn.execute("PRAGMA page_count;").fetchone()[0]
    stats["page_size"] = conn.execute("PRAGMA page_size;").fetchone()[0]
    stats["size_bytes"] = stats["page_count"] * stats["page_size"]
    
    # WAL size
    stats["wal_pages"] = conn.execute(
        "PRAGMA wal_checkpoint(PASSIVE);"
    ).fetchone()[2]
    
    # Fragmentation
    stats["freelist_count"] = conn.execute(
        "PRAGMA freelist_count;"
    ).fetchone()[0]
    
    return stats
```

### 6.2 Health Check

```python
def check_sqlite_health(conn) -> dict:
    """Health check for SQLite database."""
    health = {"status": "healthy", "checks": {}}
    
    # Can execute queries?
    try:
        conn.execute("SELECT 1").fetchone()
        health["checks"]["query"] = "ok"
    except Exception as e:
        health["checks"]["query"] = str(e)
        health["status"] = "unhealthy"
    
    # Integrity check (expensive, run periodically)
    # result = conn.execute("PRAGMA integrity_check;").fetchone()[0]
    # health["checks"]["integrity"] = result
    
    # WAL file size
    stats = get_db_stats(conn)
    if stats["wal_pages"] > 10000:
        health["checks"]["wal"] = "checkpoint needed"
        health["status"] = "degraded"
    else:
        health["checks"]["wal"] = "ok"
    
    return health
```

---

## 7. Backup

### 7.1 Online Backup

```python
import shutil
from pathlib import Path

def backup_database(conn, backup_path: Path) -> None:
    """Create online backup using SQLite backup API."""
    backup_conn = sqlite3.connect(str(backup_path))
    conn.backup(backup_conn)
    backup_conn.close()
```

### 7.2 WAL Checkpoint Before Backup

```python
def safe_backup(db_path: Path, backup_path: Path) -> None:
    """Backup with WAL checkpoint."""
    conn = sqlite3.connect(str(db_path))
    
    # Checkpoint to ensure all data in main file
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    
    # Backup
    backup_conn = sqlite3.connect(str(backup_path))
    conn.backup(backup_conn)
    backup_conn.close()
    conn.close()
```

---

## 8. Cortex Implementation

### Current State (`src/cortex/persistence/database.py`)

```python
class StateRepository:
    def __init__(self, db_path: str | Path | None = None):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
```

### Recommended Improvements

```python
class StateRepository:
    def __init__(self, db_path: str | Path | None = None):
        self._conn = sqlite3.connect(
            str(db_path),
            timeout=5.0,
            check_same_thread=False,  # For multi-threaded access
        )
        self._conn.row_factory = sqlite3.Row
        
        # Production pragmas
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        
        self._create_schema()
    
    def checkpoint(self) -> dict:
        """Checkpoint WAL to main database."""
        result = self._conn.execute(
            "PRAGMA wal_checkpoint(TRUNCATE);"
        ).fetchone()
        return {
            "blocked": result[0],
            "pages_written": result[1],
            "pages_remaining": result[2],
        }
    
    def health_check(self) -> dict:
        """Check database health."""
        ...
```

---

## 9. Limitations

When to use PostgreSQL/MySQL instead:

| Scenario | SQLite | PostgreSQL |
|----------|--------|------------|
| Single server | ✅ | ✅ |
| Multiple servers | ❌ | ✅ |
| High write concurrency | ❌ | ✅ |
| > 100 writes/sec | ⚠️ | ✅ |
| > 1TB database | ⚠️ | ✅ |
| Complex queries | ⚠️ | ✅ |

---

## References

- https://sqlite.org/wal.html
- https://sqlite.org/pragma.html
- https://markaicode.com/sqlite-4-production-database-benchmarks-pitfalls/
- https://thelinuxcode.com/sqlite-transactions-in-practice-acid-concurrency-savepoints-and-production-patterns/
- https://nerdleveltech.com/sqlite-in-2025-the-unsung-hero-powering-modern-apps
