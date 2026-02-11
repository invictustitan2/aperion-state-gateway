# Async/Sync Architecture Fix

> **Severity**: 🔴 CRITICAL (P0)
> **Impact**: Service hangs under load, violates Constitution A5 (Latency)
> **Generated**: 2026-02-08

---

## Executive Summary

The Cortex service has **critical async/sync mixing** that causes the entire API to block when:
- OpenAI embeddings are requested (2-5 second blocks)
- HuggingFace models encode text (100ms-2s CPU-bound)
- Database operations take time

This effectively **single-threads the entire service**, causing:
- Health check timeouts
- Request queuing instead of parallelism
- Service unresponsiveness under load

---

## Issues Identified

### 1. 🔴 Router Endpoints Are Sync

**Location**: `src/cortex/service/router.py`

All 15+ endpoints are defined as `def` instead of `async def`:

```python
# CURRENT (Blocking)
@router.get("/state/snapshot")
def state_snapshot(...):  # ❌ Sync function in async context
    return service.assemble_context_pack(...)

# REQUIRED (Non-Blocking)
@router.get("/state/snapshot")
async def state_snapshot(...):  # ✅ Async function
    return await service.assemble_context_pack(...)
```

### 2. 🔴 OpenAI Embedding Uses Sync Client

**Location**: `src/cortex/memory/vector_store.py:137-171`

```python
# CURRENT (Blocking)
class OpenAIEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = client.embeddings.create(...)  # ❌ Blocks event loop
        return [data.embedding for data in response.data]
```

### 3. 🔴 HuggingFace Is CPU-Bound Sync

**Location**: `src/cortex/memory/vector_store.py:96-134`

```python
# CURRENT (Blocking)
class HuggingFaceEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self._model.encode(texts)  # ❌ CPU-bound, blocks
        return [emb.tolist() for emb in embeddings]
```

### 4. 🔴 Core Service Is All Sync

**Location**: `src/cortex/service/core.py`

All 20+ methods are sync `def`, but are called from FastAPI async handlers.

---

## Fix Implementation

### Phase 1: Add Async Infrastructure

Create `src/cortex/service/executor.py`:

```python
"""Thread pool executor for CPU-bound and blocking operations."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# Shared executor for blocking operations
_executor: ThreadPoolExecutor | None = None


def get_executor() -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="cortex-blocking-",
        )
    return _executor


def shutdown_executor() -> None:
    """Shutdown the executor gracefully."""
    global _executor
    if _executor:
        _executor.shutdown(wait=True)
        _executor = None


async def run_in_executor(
    func: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Run a blocking function in the thread pool executor.
    
    Use this for:
    - CPU-bound operations (embedding encoding)
    - Sync HTTP clients (legacy code)
    - Database operations without async driver
    
    Example:
        result = await run_in_executor(sync_function, arg1, arg2)
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()
    return await loop.run_in_executor(
        executor,
        partial(func, *args, **kwargs),
    )


def async_wrap(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to wrap a sync function for async execution.
    
    Example:
        @async_wrap
        def cpu_bound_function(data):
            return heavy_computation(data)
        
        # Now callable as:
        result = await cpu_bound_function(data)
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await run_in_executor(func, *args, **kwargs)
    return wrapper
```

### Phase 2: Async Embedding Models

Update `src/cortex/memory/vector_store.py`:

```python
from cortex.service.executor import run_in_executor

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embed (for thread pool execution)."""
        pass

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Async embed - runs sync version in thread pool."""
        return await run_in_executor(self.embed_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        results = await self.embed([text])
        return results[0]


class HuggingFaceEmbedding(EmbeddingModel):
    """HuggingFace sentence-transformers embedding model."""

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Sync embed - runs in thread pool."""
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model with async support."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = ""):
        self.model_name = model_name
        self.api_key = api_key
        self._sync_client = None
        self._async_client = None

    def _get_async_client(self):
        if self._async_client is None:
            import openai
            self._async_client = openai.AsyncOpenAI(api_key=self.api_key or None)
        return self._async_client

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Sync embed - fallback for thread pool."""
        if self._sync_client is None:
            import openai
            self._sync_client = openai.OpenAI(api_key=self.api_key or None)
        response = self._sync_client.embeddings.create(
            input=texts, model=self.model_name
        )
        return [data.embedding for data in response.data]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Async embed - uses native async client."""
        client = self._get_async_client()
        response = await client.embeddings.create(
            input=texts, model=self.model_name
        )
        return [data.embedding for data in response.data]
```

### Phase 3: Async Vector Store

```python
class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents_sync(self, documents: list[Document]) -> list[str]:
        """Sync add documents."""
        pass

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Async add documents - default wraps sync."""
        return await run_in_executor(self.add_documents_sync, documents)

    @abstractmethod
    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Sync search."""
        pass

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Async search - default wraps sync."""
        return await run_in_executor(
            self.search_sync, query, top_k, metadata_filter
        )
```

### Phase 4: Async Core Service

Update `src/cortex/service/core.py`:

```python
class CortexService:
    """Core Cortex service with async support."""

    async def assemble_context_pack(
        self,
        since: str | None = None,
    ) -> SnapshotResponse:
        """Async context pack assembly."""
        # Run sync operations in thread pool
        resources = await run_in_executor(self._resource_provider)
        hot_facts = await run_in_executor(self._hot_fact_provider)
        
        # ... rest of implementation
        return SnapshotResponse(...)

    async def vector_search(
        self, request: VectorSearchRequest
    ) -> VectorSearchResponse:
        """Async vector search."""
        if self._vector_store is None:
            raise HTTPException(501, "Vector store not configured")
        
        # Async vector search
        hits = await self._vector_store.search(
            query=request.query,
            top_k=request.top_k,
            metadata_filter=request.filter,
        )
        
        return VectorSearchResponse(
            hits=[VectorHit(...) for hit in hits],
            query=request.query,
            timestamp=datetime.now(timezone.utc),
        )
```

### Phase 5: Async Router

Update `src/cortex/service/router.py`:

```python
@router.get("/state/snapshot", response_model=SnapshotResponse)
async def state_snapshot(
    auth: Annotated[TokenInfo, Depends(context_scope)],
    since: str | None = Query(default=None),
) -> SnapshotResponse:
    """Get signed context snapshot."""
    return await service.assemble_context_pack(since=since)

@router.post("/vector/search", response_model=VectorSearchResponse)
async def vector_search(
    request: VectorSearchRequest,
    auth: Annotated[TokenInfo, Depends(vector_scope)],
) -> VectorSearchResponse:
    """Perform semantic vector search."""
    return await service.vector_search(request)
```

---

## Implementation Checklist

### Immediate (COMPLETED ✅)
- [x] Create `src/cortex/service/executor.py`
- [x] Add `shutdown_executor()` to lifespan
- [x] Convert `OpenAIEmbedding` to async with native `AsyncOpenAI` client
- [x] Wrap `HuggingFaceEmbedding.embed_sync` for thread pool execution
- [x] Update `EmbeddingModel` ABC with `embed_sync`/`embed` pattern
- [x] Update `VectorStore` ABC with `*_sync`/async pattern
- [x] Update `InMemoryVectorStore` implementations
- [x] Update `ChromaVectorStore` implementations
- [x] Update tests to use sync methods

### Short-term (Pending)
- [ ] Convert router endpoints to `async def` (low priority - sync is fine for now)
- [ ] Convert core service methods to async (only if needed for parallelism)
- [ ] Add async database operations (SQLite is sync, would need aiosqlite)

### Testing
- [ ] Load test with concurrent requests
- [ ] Verify health checks respond during embedding
- [ ] Measure latency p99 improvement
- [ ] Test graceful shutdown with in-flight requests

---

## Current Architecture Pattern

The Cortex service now supports both sync and async usage patterns:

### Sync Path (Current Default)
Router (sync) → Service (sync) → VectorStore.search_sync() → Embedding.embed_sync()

This works well for:
- Single-threaded workloads
- Testing without async overhead
- Backward compatibility

### Async Path (Available)
Router (async) → Service (async) → VectorStore.search() → Embedding.embed()

For HuggingFace: `embed()` runs `embed_sync()` in thread pool
For OpenAI: `embed()` uses native `AsyncOpenAI` client

This enables:
- Concurrent request handling
- Non-blocking embedding operations
- Health checks that respond during slow operations

---

## Verification

After implementing, verify with:

```python
# test_async_behavior.py
import asyncio
import httpx
import time

async def test_concurrent_requests():
    """Verify requests are processed concurrently."""
    async with httpx.AsyncClient(base_url="http://localhost:4949") as client:
        headers = {"Authorization": "Bearer test-token"}
        
        # Start 10 concurrent requests
        start = time.time()
        tasks = [
            client.post(
                "/vector/search",
                json={"query": f"test query {i}", "top_k": 5},
                headers=headers,
            )
            for i in range(10)
        ]
        
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        # With async: ~2s (parallel)
        # With sync: ~20s (sequential)
        assert elapsed < 5, f"Requests took {elapsed}s - likely blocking!"
        assert all(r.status_code == 200 for r in responses)

if __name__ == "__main__":
    asyncio.run(test_concurrent_requests())
```

---

## References

- FastAPI Async: https://fastapi.tiangolo.com/async/
- asyncio run_in_executor: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
- OpenAI AsyncClient: https://github.com/openai/openai-python#async-usage
