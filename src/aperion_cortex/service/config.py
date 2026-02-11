"""Configuration primitives for the Cortex service."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class VectorBackend(str, Enum):
    """Supported vector store backends."""

    CHROMA = "chroma"
    FAISS = "faiss"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


@dataclass(slots=True)
class ScopedToken:
    """Represents a scoped access token for the Cortex."""

    token: str
    scopes: Sequence[str]
    expires_at: datetime | None = None

    def is_expired(self, now: datetime) -> bool:
        if self.expires_at is None:
            return False
        return now >= self.expires_at


@dataclass(slots=True)
class CortexConfig:
    """Runtime configuration for the Cortex service.

    Configuration Sources (priority order):
    1. Direct constructor arguments
    2. Environment variables (CORTEX_*)
    3. Default values

    Attributes:
        signing_key: Secret key for signing ContextPacks (REQUIRED)
        port: Service port (default: 4949)
        default_ttl_seconds: TTL for context packs (default: 120)
        context_size_cap: Max context pack size in bytes (default: 30KB)
        tool_payload_cap: Max tool response size in bytes (default: 100KB)
        audit_log_limit: In-memory audit log entries (default: 500)
        db_path: SQLite database path (default: data/cortex.db)
        vector_backend: Vector store backend (default: chroma)
        embedding_provider: Embedding model provider (default: huggingface)
        embedding_model: Model identifier for embeddings
        max_token_budget: Maximum tokens for context packing (default: 4096)
    """

    signing_key: str
    port: int = 4949
    default_ttl_seconds: int = 120
    context_size_cap: int = 30 * 1024  # 30KB
    tool_payload_cap: int = 100 * 1024  # 100KB
    audit_log_limit: int = 500
    db_path: str = "data/cortex.db"
    vector_backend: VectorBackend = VectorBackend.CHROMA
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE
    embedding_model: str = "all-MiniLM-L6-v2"
    max_token_budget: int = 4096
    default_scopes: Sequence[str] = (
        "context",
        "vector",
        "database",
        "filesystem",
        "services",
        "state",
        "audit",
    )
    tokens: list[ScopedToken] = field(default_factory=list)
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    @classmethod
    def from_env(cls) -> CortexConfig:
        """Create configuration from environment variables.

        Required:
            CORTEX_SIGNING_KEY: Signing key for ContextPacks

        Optional:
            CORTEX_PORT: Service port (default: 4949)
            CORTEX_DB_PATH: SQLite database path
            CORTEX_VECTOR_BACKEND: 'chroma' or 'faiss'
            CORTEX_EMBEDDING_PROVIDER: 'huggingface' or 'openai'
            CORTEX_EMBEDDING_MODEL: Model identifier
            CORTEX_MAX_TOKEN_BUDGET: Max tokens for context (default: 4096)
            CORTEX_DEFAULT_TTL: Context pack TTL in seconds
            CORTEX_CONTEXT_SIZE_CAP: Max context size in bytes
            CORTEX_TOKENS: Comma-separated list of tokens (all get default scopes)
        """
        signing_key = os.environ.get("CORTEX_SIGNING_KEY")
        if not signing_key:
            raise ValueError("CORTEX_SIGNING_KEY environment variable is required")

        config = cls(
            signing_key=signing_key,
            port=int(os.environ.get("CORTEX_PORT", "4949")),
            db_path=os.environ.get("CORTEX_DB_PATH", "data/cortex.db"),
            vector_backend=VectorBackend(
                os.environ.get("CORTEX_VECTOR_BACKEND", "chroma").lower()
            ),
            embedding_provider=EmbeddingProvider(
                os.environ.get("CORTEX_EMBEDDING_PROVIDER", "huggingface").lower()
            ),
            embedding_model=os.environ.get(
                "CORTEX_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            max_token_budget=int(os.environ.get("CORTEX_MAX_TOKEN_BUDGET", "4096")),
            default_ttl_seconds=int(os.environ.get("CORTEX_DEFAULT_TTL", "120")),
            context_size_cap=int(
                os.environ.get("CORTEX_CONTEXT_SIZE_CAP", str(30 * 1024))
            ),
        )
        
        # Load tokens from environment
        tokens_str = os.environ.get("CORTEX_TOKENS", "")
        if tokens_str:
            for token in tokens_str.split(","):
                token = token.strip()
                if token:
                    config.tokens.append(
                        ScopedToken(token=token, scopes=config.default_scopes)
                    )
        
        return config

    def with_tokens(self, tokens: Iterable[ScopedToken]) -> CortexConfig:
        """Add tokens to the configuration."""
        self.tokens = list(tokens)
        return self

    def issue_ephemeral_token(
        self,
        token: str,
        scopes: Sequence[str],
        lifetime: timedelta | None = None,
    ) -> None:
        """Issue an ephemeral token with optional expiration."""
        expires_at = None
        if lifetime is not None:
            expires_at = datetime.now(timezone.utc) + lifetime
        self.tokens.append(
            ScopedToken(token=token, scopes=scopes, expires_at=expires_at)
        )


@dataclass(slots=True)
class AdapterConfig:
    """Configuration for external adapters/providers.

    Used to configure callbacks for resource, database, and filesystem
    providers when the Cortex is integrated with external systems.
    """

    resource_provider: Any = None
    hot_fact_provider: Any = None
    summary_provider: Any = None
    vector_searcher: Any = None
    database_provider: Any = None
    schema_provider: Any = None
    database_sampler: Any = None
    service_provider: Any = None
    service_health_provider: Any = None
    fs_list_provider: Any = None
    fs_read_provider: Any = None
