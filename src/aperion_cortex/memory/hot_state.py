"""Hot State Manager - Fast session state and caching.

The "Hot State" tier provides low-latency access to active session data.
Implements an in-memory cache with TTL support and optional Redis backend
for distributed deployments.

Key Features:
- In-memory LRU cache with TTL
- Session isolation
- Optional Redis backend for distributed state
- Thread-safe operations
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""

    value: T
    created_at: float
    expires_at: float | None = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self, now: float | None = None) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        now = now or time.time()
        return now >= self.expires_at


class HotStateManager:
    """In-memory hot state manager with TTL and LRU eviction.

    Provides fast key-value storage for session state with:
    - Automatic TTL-based expiration
    - LRU eviction when max size reached
    - Session-scoped namespacing
    - Thread-safe operations

    Example:
        state = HotStateManager(max_size=1000, default_ttl=300)

        # Set session state
        state.set("session:123", "user_prefs", {"theme": "dark"})

        # Get session state
        prefs = state.get("session:123", "user_prefs")

        # Get with namespace prefix
        state.set_session("123", "context", {"messages": [...]})
        ctx = state.get_session("123", "context")
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float | None = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,
    ):
        """Initialize the hot state manager.

        Args:
            max_size: Maximum number of entries before LRU eviction
            default_ttl: Default time-to-live in seconds (None = no expiration)
            cleanup_interval: Interval for background cleanup (seconds)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, CacheEntry[Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

    def _make_key(self, namespace: str, key: str) -> str:
        """Create a namespaced cache key."""
        return f"{namespace}:{key}"

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        """Set a value in the cache.

        Args:
            namespace: Namespace (e.g., session ID)
            key: Key within namespace
            value: Value to store
            ttl: Time-to-live in seconds (None = use default)
        """
        cache_key = self._make_key(namespace, key)
        now = time.time()

        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = now + effective_ttl if effective_ttl else None

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=expires_at,
            access_count=0,
            last_accessed=now,
        )

        with self._lock:
            # Remove if exists (to update position in OrderedDict)
            if cache_key in self._cache:
                del self._cache[cache_key]

            # Add new entry
            self._cache[cache_key] = entry

            # Evict if over max size
            self._evict_if_needed()

            # Periodic cleanup
            self._maybe_cleanup()

    def get(
        self,
        namespace: str,
        key: str,
        default: T | None = None,
    ) -> Any | T | None:
        """Get a value from the cache.

        Args:
            namespace: Namespace (e.g., session ID)
            key: Key within namespace
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        cache_key = self._make_key(namespace, key)
        now = time.time()

        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                return default

            if entry.is_expired(now):
                del self._cache[cache_key]
                return default

            # Update access metadata
            entry.access_count += 1
            entry.last_accessed = now

            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)

            return entry.value

    def delete(self, namespace: str, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            namespace: Namespace
            key: Key within namespace

        Returns:
            True if deleted, False if not found
        """
        cache_key = self._make_key(namespace, key)

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
            return False

    def exists(self, namespace: str, key: str) -> bool:
        """Check if a key exists and is not expired.

        Args:
            namespace: Namespace
            key: Key within namespace

        Returns:
            True if exists and not expired
        """
        cache_key = self._make_key(namespace, key)
        now = time.time()

        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                return False
            if entry.is_expired(now):
                del self._cache[cache_key]
                return False
            return True

    def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of entries cleared
        """
        prefix = f"{namespace}:"
        count = 0

        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
                count += 1

        logger.debug(f"Cleared {count} entries from namespace '{namespace}'")
        return count

    def clear_all(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()

        logger.debug(f"Cleared all {count} cache entries")
        return count

    # Convenience methods for session-scoped access
    def set_session(self, session_id: str, key: str, value: Any, ttl: float | None = None) -> None:
        """Set a value in a session's namespace."""
        self.set(f"session:{session_id}", key, value, ttl)

    def get_session(self, session_id: str, key: str, default: T | None = None) -> Any | T | None:
        """Get a value from a session's namespace."""
        return self.get(f"session:{session_id}", key, default)

    def delete_session(self, session_id: str, key: str) -> bool:
        """Delete a value from a session's namespace."""
        return self.delete(f"session:{session_id}", key)

    def clear_session(self, session_id: str) -> int:
        """Clear all data for a session."""
        return self.clear_namespace(f"session:{session_id}")

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over max size."""
        while len(self._cache) > self.max_size:
            # Remove oldest (first) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Evicted oldest entry: {oldest_key}")

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup >= self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now

    def _cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items() if entry.is_expired(now)
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

        return len(expired_keys)

    @property
    def size(self) -> int:
        """Current number of entries in cache."""
        return len(self._cache)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with size, max_size, hit/miss stats, etc.
        """
        with self._lock:
            total_accesses = sum(e.access_count for e in self._cache.values())
            expired_count = sum(
                1 for e in self._cache.values() if e.is_expired()
            )

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "expired_pending": expired_count,
                "default_ttl": self.default_ttl,
            }
