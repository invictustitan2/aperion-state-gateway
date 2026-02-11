"""State Repository - SQLite persistence for state snapshots.

Provides persistent storage for State Gateway context packs,
enabling state survival across service restarts and multi-user
session isolation.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# SQL schema for state snapshots
SCHEMA_SQL = """
-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_active TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

-- State snapshots table
CREATE TABLE IF NOT EXISTS state_snapshots (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    snapshot_data TEXT NOT NULL,
    signature TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    ttl_seconds INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_snapshots_session_id ON state_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_created_at ON state_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_generated_at ON state_snapshots(generated_at);
"""


class StateRepository:
    """Repository for persistent state storage.

    Provides persistence layer for Cortex context packs, enabling:
    - State survival across service restarts
    - Multi-user session isolation
    - Historical snapshot retrieval
    - Automatic cleanup of expired snapshots

    Example:
        with StateRepository("data/cortex.db") as repo:
            # Save a snapshot
            snapshot_id = repo.save_snapshot(
                session_id="sess_123",
                snapshot_data={"summary": "...", "resources": [...]},
                signature="sha256:abc...",
                generated_at=datetime.now(timezone.utc),
                ttl_seconds=300,
            )

            # Retrieve latest
            snapshot = repo.get_latest_snapshot("sess_123")
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize repository with database connection.

        Args:
            db_path: Path to SQLite database file. Defaults to data/cortex.db
        """
        if db_path is None:
            db_path = Path.cwd() / "data" / "cortex.db"
        else:
            db_path = Path(db_path)

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_connection()
        self._ensure_schema()

    def _ensure_connection(self) -> None:
        """Ensure database connection is established."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode = WAL")
            # Sync only at critical moments (WAL provides durability)
            self._conn.execute("PRAGMA synchronous = NORMAL")
            # Wait up to 5 seconds if database is locked
            self._conn.execute("PRAGMA busy_timeout = 5000")

    def _ensure_schema(self) -> None:
        """Ensure database schema exists."""
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection, ensuring it's established."""
        self._ensure_connection()
        assert self._conn is not None
        return self._conn

    def ensure_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Ensure a session exists, creating if necessary.

        Args:
            session_id: Session identifier
            metadata: Optional session metadata
        """
        conn = self._get_conn()
        metadata_json = json.dumps(metadata or {})

        conn.execute(
            """
            INSERT INTO sessions (id, metadata)
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_active = datetime('now'),
                metadata = ?
            """,
            (session_id, metadata_json, metadata_json),
        )
        conn.commit()

    def save_snapshot(
        self,
        session_id: str,
        snapshot_data: dict[str, Any],
        signature: str,
        generated_at: datetime,
        ttl_seconds: int,
    ) -> str:
        """Save a state snapshot to the database.

        Args:
            session_id: Session identifier
            snapshot_data: Complete snapshot payload (JSON-serializable dict)
            signature: HMAC signature for integrity verification
            generated_at: Timestamp when snapshot was generated
            ttl_seconds: Time-to-live for this snapshot

        Returns:
            Snapshot ID (generated UUID)
        """
        # Ensure session exists
        self.ensure_session(session_id)

        snapshot_id = str(uuid.uuid4())
        snapshot_json = json.dumps(snapshot_data, separators=(",", ":"), default=str)
        generated_at_iso = generated_at.isoformat()

        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO state_snapshots
            (id, session_id, snapshot_data, signature, generated_at, ttl_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                session_id,
                snapshot_json,
                signature,
                generated_at_iso,
                ttl_seconds,
            ),
        )
        conn.commit()

        return snapshot_id

    def get_latest_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve the most recent snapshot for a session.

        Args:
            session_id: Session identifier

        Returns:
            Snapshot data dict with keys: id, session_id, snapshot_data (parsed),
            signature, generated_at (datetime), ttl_seconds, created_at (datetime).
            Returns None if no snapshots exist.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, session_id, snapshot_data, signature, generated_at,
                   ttl_seconds, created_at
            FROM state_snapshots
            WHERE session_id = ?
            ORDER BY generated_at DESC, created_at DESC, id DESC
            LIMIT 1
            """,
            (session_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "snapshot_data": json.loads(row["snapshot_data"]),
            "signature": row["signature"],
            "generated_at": datetime.fromisoformat(row["generated_at"]),
            "ttl_seconds": row["ttl_seconds"],
            "created_at": datetime.fromisoformat(row["created_at"]),
        }

    def get_snapshot_by_id(self, snapshot_id: str) -> dict[str, Any] | None:
        """Retrieve a specific snapshot by ID.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            Snapshot data dict or None if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, session_id, snapshot_data, signature, generated_at,
                   ttl_seconds, created_at
            FROM state_snapshots
            WHERE id = ?
            """,
            (snapshot_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "snapshot_data": json.loads(row["snapshot_data"]),
            "signature": row["signature"],
            "generated_at": datetime.fromisoformat(row["generated_at"]),
            "ttl_seconds": row["ttl_seconds"],
            "created_at": datetime.fromisoformat(row["created_at"]),
        }

    def list_snapshots(
        self,
        session_id: str,
        since: datetime | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List snapshots for a session with optional filtering.

        Args:
            session_id: Session identifier
            since: Only return snapshots created after this timestamp
            limit: Maximum number of snapshots to return

        Returns:
            List of snapshot dicts, ordered by created_at descending
        """
        conn = self._get_conn()

        if since is not None:
            since_iso = since.isoformat()
            cursor = conn.execute(
                """
                SELECT id, session_id, snapshot_data, signature, generated_at,
                       ttl_seconds, created_at
                FROM state_snapshots
                WHERE session_id = ? AND created_at > ?
                ORDER BY generated_at DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (session_id, since_iso, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, session_id, snapshot_data, signature, generated_at,
                       ttl_seconds, created_at
                FROM state_snapshots
                WHERE session_id = ?
                ORDER BY generated_at DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (session_id, limit),
            )

        snapshots = []
        for row in cursor.fetchall():
            snapshots.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "snapshot_data": json.loads(row["snapshot_data"]),
                    "signature": row["signature"],
                    "generated_at": datetime.fromisoformat(row["generated_at"]),
                    "ttl_seconds": row["ttl_seconds"],
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
            )

        return snapshots

    def cleanup_expired(self, max_age_days: int = 30) -> int:
        """Remove snapshots older than specified age.

        Args:
            max_age_days: Maximum age in days before snapshot is deleted

        Returns:
            Number of snapshots deleted
        """
        conn = self._get_conn()
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cutoff_iso = cutoff.isoformat()

        cursor = conn.execute(
            "DELETE FROM state_snapshots WHERE created_at < ?",
            (cutoff_iso,),
        )
        conn.commit()

        return cursor.rowcount

    def cleanup_by_ttl(self) -> int:
        """Remove snapshots that have exceeded their TTL.

        Returns:
            Number of snapshots deleted
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc)

        # Fetch and check in Python (SQLite date arithmetic is limited)
        cursor = conn.execute(
            "SELECT id, generated_at, ttl_seconds FROM state_snapshots"
        )

        expired_ids = []
        for row in cursor.fetchall():
            generated = datetime.fromisoformat(row["generated_at"])
            expires_at = generated + timedelta(seconds=row["ttl_seconds"])
            if now > expires_at:
                expired_ids.append(row["id"])

        if not expired_ids:
            return 0

        placeholders = ",".join("?" * len(expired_ids))
        conn.execute(
            f"DELETE FROM state_snapshots WHERE id IN ({placeholders})",
            expired_ids,
        )
        conn.commit()

        return len(expired_ids)

    def count_by_session(self, session_id: str) -> int:
        """Count total snapshots for a session."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM state_snapshots WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return row["count"] if row else 0

    def count_all(self) -> int:
        """Count total snapshots in database."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) as count FROM state_snapshots")
        row = cursor.fetchone()
        return row["count"] if row else 0

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> StateRepository:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes connection."""
        self.close()
