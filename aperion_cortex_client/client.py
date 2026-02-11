"""Cortex client implementation with async/sync interfaces."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CortexError(Exception):
    """Base exception for Cortex client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CortexConfig:
    """Configuration for Cortex client."""

    url: str = "http://localhost:4949"
    token: str | None = None
    timeout: float = 30.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "CortexConfig":
        """Load configuration from environment variables."""
        return cls(
            url=os.environ.get("CORTEX_URL", "http://localhost:4949"),
            token=os.environ.get("CORTEX_TOKEN"),
            timeout=float(os.environ.get("CORTEX_TIMEOUT", "30.0")),
            max_retries=int(os.environ.get("CORTEX_MAX_RETRIES", "3")),
        )


# ---------------------------------------------------------------------------
# Response Models (mirrors server models for type safety)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Resource:
    """A discoverable resource."""
    
    id: str
    kind: str
    name: str
    uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextPack:
    """Signed, token-optimized context snapshot."""
    
    generated_at: datetime
    ttl_seconds: int
    summary: str
    resources: list[Resource]
    hot_facts: list[str]
    signature: str
    token_count: int = 0
    freshness: str = "fresh"


@dataclass(slots=True)
class VectorHit:
    """A vector search result."""
    
    doc_id: str
    score: float
    uri: str
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorSearchResult:
    """Result from vector search."""
    
    hits: list[VectorHit]
    query_embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0


@dataclass(slots=True)
class ContextOptimizeResult:
    """Result from context optimization."""
    
    items: list[dict[str, Any]]
    total_tokens: int
    items_included: int
    items_dropped: int
    budget_used_percent: float


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CortexClientError(Exception):
    """Base exception for Cortex client errors."""
    
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CortexConnectionError(CortexClientError):
    """Connection to Cortex failed."""
    pass


class CortexAuthError(CortexClientError):
    """Authentication failed."""
    pass


class CortexNotFoundError(CortexClientError):
    """Resource not found."""
    pass


class CortexRateLimitError(CortexClientError):
    """Rate limit exceeded."""
    
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, 429)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Client Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CortexClientConfig:
    """Configuration for CortexClient."""
    
    base_url: str
    token: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    connection_pool_size: int = 10


# ---------------------------------------------------------------------------
# Async Client
# ---------------------------------------------------------------------------


class CortexClient:
    """Async client for Cortex state gateway.
    
    Example:
        >>> async with CortexClient("http://localhost:4949") as client:
        ...     pack = await client.get_snapshot()
        ...     print(pack.summary)
    """
    
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        self._config = CortexClientConfig(
            base_url=base_url.rstrip("/"),
            token=token,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self._client: httpx.AsyncClient | None = None
    
    async def __aenter__(self) -> "CortexClient":
        await self._ensure_client()
        return self
    
    async def __aexit__(self, *args) -> None:
        await self.close()
    
    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                limits=httpx.Limits(
                    max_connections=self._config.connection_pool_size,
                    max_keepalive_connections=5,
                ),
            )
        return self._client
    
    async def close(self) -> None:
        """Close the client connection."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    def _build_headers(self, correlation_id: str | None = None) -> dict[str, str]:
        headers = {}
        if self._config.token:
            headers["Authorization"] = f"Bearer {self._config.token}"
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        return headers
    
    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Make request with retry logic."""
        client = await self._ensure_client()
        headers = self._build_headers(correlation_id)
        
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                response = await client.request(
                    method,
                    path,
                    json=json,
                    params=params,
                    headers=headers,
                )
                
                # Handle specific status codes
                if response.status_code == 401:
                    raise CortexAuthError("Authentication failed", 401)
                elif response.status_code == 404:
                    raise CortexNotFoundError("Resource not found", 404)
                elif response.status_code == 429:
                    retry_after = response.headers.get("X-RateLimit-Reset")
                    raise CortexRateLimitError(
                        "Rate limit exceeded",
                        retry_after=float(retry_after) if retry_after else None,
                    )
                
                response.raise_for_status()
                return response.json()
                
            except httpx.ConnectError as e:
                last_error = CortexConnectionError(f"Connection failed: {e}")
            except httpx.TimeoutException as e:
                last_error = CortexConnectionError(f"Request timed out: {e}")
            except CortexClientError:
                raise  # Don't retry auth/notfound/ratelimit errors
            except httpx.HTTPStatusError as e:
                last_error = CortexClientError(
                    f"HTTP {e.response.status_code}: {e.response.text}",
                    e.response.status_code,
                )
            
            # Exponential backoff before retry
            if attempt < self._config.max_retries - 1:
                delay = self._config.retry_backoff * (2 ** attempt)
                logger.debug(f"Retry {attempt + 1}/{self._config.max_retries} after {delay}s")
                await asyncio.sleep(delay)
        
        if last_error:
            raise last_error
        raise CortexConnectionError("Request failed after retries")
    
    # -----------------------------------------------------------------------
    # State API
    # -----------------------------------------------------------------------
    
    async def get_snapshot(
        self,
        *,
        correlation_id: str | None = None,
    ) -> ContextPack:
        """Get current state snapshot (ContextPack).
        
        Returns:
            ContextPack with resources, hot facts, and signature.
        """
        data = await self._request("GET", "/state/snapshot", correlation_id=correlation_id)
        return ContextPack(
            generated_at=datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00")),
            ttl_seconds=data["ttl_seconds"],
            summary=data["summary"],
            resources=[
                Resource(**r) for r in data.get("resources", [])
            ],
            hot_facts=data.get("hot_facts", []),
            signature=data["signature"],
            token_count=data.get("token_count", 0),
            freshness=data.get("freshness", "fresh"),
        )
    
    async def discover(
        self,
        *,
        correlation_id: str | None = None,
    ) -> list[Resource]:
        """Discover available resources.
        
        Returns:
            List of discoverable resources.
        """
        data = await self._request("GET", "/state/discover", correlation_id=correlation_id)
        return [Resource(**r) for r in data.get("resources", [])]
    
    # -----------------------------------------------------------------------
    # Vector Search API
    # -----------------------------------------------------------------------
    
    async def vector_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> VectorSearchResult:
        """Perform semantic vector search.
        
        Args:
            query: Search query text.
            top_k: Number of results to return (1-100).
            filter_metadata: Optional metadata filters.
            correlation_id: Optional request correlation ID.
        
        Returns:
            VectorSearchResult with hits and timing info.
        """
        payload = {
            "query": query,
            "top_k": top_k,
            "filter_metadata": filter_metadata or {},
        }
        data = await self._request(
            "POST",
            "/vector/search",
            json=payload,
            correlation_id=correlation_id,
        )
        return VectorSearchResult(
            hits=[VectorHit(**h) for h in data.get("hits", [])],
            query_embedding_time_ms=data.get("query_embedding_time_ms", 0.0),
            search_time_ms=data.get("search_time_ms", 0.0),
        )
    
    # -----------------------------------------------------------------------
    # Context Optimization API
    # -----------------------------------------------------------------------
    
    async def optimize_context(
        self,
        items: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        reserve_tokens: int = 0,
        correlation_id: str | None = None,
    ) -> ContextOptimizeResult:
        """Optimize context items for LLM consumption.
        
        Args:
            items: List of context items with content, priority, category.
            max_tokens: Maximum token budget.
            reserve_tokens: Tokens to reserve for response.
            correlation_id: Optional request correlation ID.
        
        Returns:
            ContextOptimizeResult with optimized items and stats.
        """
        payload = {
            "items": items,
            "max_tokens": max_tokens,
            "reserve_tokens": reserve_tokens,
        }
        data = await self._request(
            "POST",
            "/context/optimize",
            json=payload,
            correlation_id=correlation_id,
        )
        return ContextOptimizeResult(
            items=data.get("items", []),
            total_tokens=data.get("total_tokens", 0),
            items_included=data.get("items_included", 0),
            items_dropped=data.get("items_dropped", 0),
            budget_used_percent=data.get("budget_used_percent", 0.0),
        )
    
    # -----------------------------------------------------------------------
    # Audit API
    # -----------------------------------------------------------------------
    
    async def record_audit(
        self,
        run_id: str,
        agent_name: str,
        method_name: str,
        prompt: str,
        raw_output: str,
        *,
        correlation_id: str | None = None,
        parsed_data: dict[str, Any] | None = None,
        parse_ok: bool = False,
        parse_strategy: str | None = None,
        parse_confidence: float | None = None,
        parse_error: str | None = None,
        provider_name: str = "",
        model_name: str = "",
        latency_ms: float = 0.0,
        context_refs: list[str] | None = None,
        policy_decisions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Record agent execution to the audit ledger.
        
        Args:
            run_id: Unique run identifier.
            agent_name: Name of the executing agent.
            method_name: Method/function that was called.
            prompt: Input prompt to the agent.
            raw_output: Raw LLM output.
            ... additional optional fields for parse results, timing, etc.
        
        Returns:
            Response with run_id, recorded_at, and ledger_path.
        """
        payload = {
            "run_id": run_id,
            "correlation_id": correlation_id,
            "agent_name": agent_name,
            "method_name": method_name,
            "prompt": prompt,
            "raw_output": raw_output,
            "parsed_data": parsed_data,
            "parse_ok": parse_ok,
            "parse_strategy": parse_strategy,
            "parse_confidence": parse_confidence,
            "parse_error": parse_error,
            "provider_name": provider_name,
            "model_name": model_name,
            "latency_ms": latency_ms,
            "context_refs": context_refs or [],
            "policy_decisions": policy_decisions or [],
        }
        return await self._request(
            "POST",
            "/audit/record",
            json=payload,
            correlation_id=correlation_id,
        )
    
    # -----------------------------------------------------------------------
    # Health API
    # -----------------------------------------------------------------------
    
    async def health(self) -> dict[str, Any]:
        """Get service health status."""
        return await self._request("GET", "/healthz")
    
    async def ready(self) -> bool:
        """Check if service is ready."""
        try:
            data = await self._request("GET", "/ready")
            return data.get("ready", False)
        except CortexClientError:
            return False


# ---------------------------------------------------------------------------
# Sync Client Wrapper
# ---------------------------------------------------------------------------


class CortexClientSync:
    """Synchronous wrapper for CortexClient.
    
    Use this when you need to call Cortex from synchronous code.
    
    Example:
        >>> client = CortexClientSync("http://localhost:4949")
        >>> pack = client.get_snapshot()
        >>> print(pack.summary)
    """
    
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self._async_client = CortexClient(
            base_url,
            token=token,
            timeout=timeout,
            max_retries=max_retries,
        )
    
    def _run(self, coro):
        """Run coroutine in sync context."""
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - use thread-safe scheduling
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=self._async_client._config.timeout)
        except RuntimeError:
            # No running loop - create one
            return asyncio.run(coro)
    
    def close(self) -> None:
        """Close the client."""
        self._run(self._async_client.close())
    
    def get_snapshot(self, *, correlation_id: str | None = None) -> ContextPack:
        """Get current state snapshot."""
        return self._run(self._async_client.get_snapshot(correlation_id=correlation_id))
    
    def discover(self, *, correlation_id: str | None = None) -> list[Resource]:
        """Discover available resources."""
        return self._run(self._async_client.discover(correlation_id=correlation_id))
    
    def vector_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> VectorSearchResult:
        """Perform semantic vector search."""
        return self._run(
            self._async_client.vector_search(
                query,
                top_k=top_k,
                filter_metadata=filter_metadata,
                correlation_id=correlation_id,
            )
        )
    
    def optimize_context(
        self,
        items: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        reserve_tokens: int = 0,
        correlation_id: str | None = None,
    ) -> ContextOptimizeResult:
        """Optimize context items for LLM consumption."""
        return self._run(
            self._async_client.optimize_context(
                items,
                max_tokens=max_tokens,
                reserve_tokens=reserve_tokens,
                correlation_id=correlation_id,
            )
        )
    
    def record_audit(
        self,
        run_id: str,
        agent_name: str,
        method_name: str,
        prompt: str,
        raw_output: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Record agent execution to the audit ledger."""
        return self._run(
            self._async_client.record_audit(
                run_id,
                agent_name,
                method_name,
                prompt,
                raw_output,
                **kwargs,
            )
        )
    
    def health(self) -> dict[str, Any]:
        """Get service health status."""
        return self._run(self._async_client.health())
    
    def ready(self) -> bool:
        """Check if service is ready."""
        return self._run(self._async_client.ready())
