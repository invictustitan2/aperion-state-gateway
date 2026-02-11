# Integration Guide: Cortex ↔ Aperion-Legendary-AI

> Guide for integrating aperion-state-gateway with aperion-legendary-ai-main
> Path: /home/dreamboat/projects/aperion/aperion-legendary-ai-main

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    aperion-legendary-ai-main                     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Agents     │  │   Chat/CLI   │  │   EventBus           │   │
│  │              │  │              │  │   (stack/aperion/    │   │
│  │  - Analyst   │  │  - TUI       │  │    foundation/       │   │
│  │  - Coder     │  │  - API       │  │    event_bus.py)     │   │
│  │  - Reviewer  │  │  - SSE       │  │                      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│         └─────────────────┼──────────────────────┘               │
│                           │                                      │
│                    HTTP/gRPC                                     │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    aperion-state-gateway                         │
│                    ("The Cortex")                                │
│                    Port: 4949                                    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Memory     │  │   Vector     │  │   Audit Ledger       │   │
│  │   Hot/Warm   │  │   Search     │  │   (Training Data)    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  SQLite + ChromaDB                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Current Connection Points

### 2.1 In aperion-legendary-ai-main

| Component | Path | Integration |
|-----------|------|-------------|
| State Gateway (old) | `stack/aperion/state_gateway/` | → Replace with Cortex client |
| Doc RAG | `stack/aperion/foundation/doc_rag.py` | → Migrate to Cortex vector search |
| Training Ledger | `stack/aperion/training/ledger.py` | → Migrate to Cortex `/audit/record` |
| EventBus | `stack/aperion/foundation/event_bus.py` | → Bridge to Cortex SSE |

### 2.2 API Endpoints (Cortex)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/state/snapshot` | GET | Get context pack for LLM |
| `/vector/search` | POST | Semantic search |
| `/audit/record` | POST | Log agent execution |
| `/context/optimize` | POST | Token-optimize content |
| `/events/stream` | GET (SSE) | Real-time updates |

---

## 3. Client SDK (Recommended)

Create `stack/aperion/clients/cortex_client.py`:

```python
"""Cortex client for aperion-legendary-ai integration."""

from __future__ import annotations

import httpx
from dataclasses import dataclass
from typing import Any

@dataclass
class CortexConfig:
    """Cortex client configuration."""
    base_url: str = "http://localhost:4949"
    token: str = ""
    timeout: float = 30.0

class CortexClient:
    """HTTP client for Cortex service."""
    
    def __init__(self, config: CortexConfig | None = None):
        config = config or CortexConfig()
        self._base_url = config.base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {config.token}"}
        self._timeout = config.timeout
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        )
    
    def get_snapshot(self, since: str | None = None) -> dict:
        """Get context snapshot for LLM consumption."""
        params = {"since": since} if since else {}
        response = self._client.get("/state/snapshot", params=params)
        response.raise_for_status()
        return response.json()
    
    def vector_search(
        self,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[dict]:
        """Perform semantic vector search."""
        response = self._client.post(
            "/vector/search",
            json={
                "query": query,
                "top_k": top_k,
                "namespace": namespace,
            },
        )
        response.raise_for_status()
        return response.json()["hits"]
    
    def record_audit(
        self,
        agent_name: str,
        method_name: str,
        prompt: str,
        raw_output: str,
        latency_ms: float,
        *,
        correlation_id: str | None = None,
        provider_name: str = "",
        model_name: str = "",
        usage: dict | None = None,
    ) -> str:
        """Record agent execution to audit ledger."""
        response = self._client.post(
            "/audit/record",
            json={
                "agent_name": agent_name,
                "method_name": method_name,
                "prompt": prompt,
                "raw_output": raw_output,
                "latency_ms": latency_ms,
                "correlation_id": correlation_id,
                "provider_name": provider_name,
                "model_name": model_name,
                "usage": usage or {},
            },
        )
        response.raise_for_status()
        return response.json()["run_id"]
    
    def optimize_context(
        self,
        items: list[dict],
        max_tokens: int = 4096,
    ) -> dict:
        """Optimize context items for token budget."""
        response = self._client.post(
            "/context/optimize",
            json={
                "items": items,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if Cortex is healthy."""
        try:
            response = self._client.get("/healthz")
            return response.json().get("status") == "ok"
        except Exception:
            return False
    
    def close(self):
        """Close HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Async version
class AsyncCortexClient:
    """Async HTTP client for Cortex service."""
    
    def __init__(self, config: CortexConfig | None = None):
        config = config or CortexConfig()
        self._base_url = config.base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {config.token}"}
        self._timeout = config.timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        )
    
    async def get_snapshot(self, since: str | None = None) -> dict:
        """Get context snapshot for LLM consumption."""
        params = {"since": since} if since else {}
        response = await self._client.get("/state/snapshot", params=params)
        response.raise_for_status()
        return response.json()
    
    async def vector_search(
        self,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[dict]:
        """Perform semantic vector search."""
        response = await self._client.post(
            "/vector/search",
            json={
                "query": query,
                "top_k": top_k,
                "namespace": namespace,
            },
        )
        response.raise_for_status()
        return response.json()["hits"]
    
    # ... async versions of other methods ...
    
    async def close(self):
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
```

---

## 4. Migration Steps

### Phase 1: Parallel Operation

Run both old State Gateway and new Cortex side-by-side:

```python
# In agent code
from stack.aperion.clients.cortex_client import CortexClient

# Feature flag
USE_CORTEX = os.getenv("USE_CORTEX", "false").lower() == "true"

if USE_CORTEX:
    cortex = CortexClient(CortexConfig(
        base_url="http://localhost:4949",
        token=os.getenv("CORTEX_TOKEN"),
    ))
    context = cortex.get_snapshot()
else:
    # Old path
    from stack.aperion.state_gateway import get_context
    context = get_context()
```

### Phase 2: Audit Logging Migration

Replace direct ledger writes with Cortex API:

```python
# Before (in agent code)
from stack.aperion.training.ledger import record_run
record_run(agent_name, prompt, output, ...)

# After
cortex.record_audit(
    agent_name=agent_name,
    method_name=method_name,
    prompt=prompt,
    raw_output=output,
    latency_ms=elapsed_ms,
    correlation_id=correlation_id,
)
```

### Phase 3: Vector Search Migration

Replace doc_rag with Cortex vector search:

```python
# Before
from stack.aperion.foundation.doc_rag import DocumentationRAG
rag = DocumentationRAG(docs_path)
results = rag.search_by_keyword(query)

# After
hits = cortex.vector_search(query, top_k=10)
```

### Phase 4: EventBus Bridge

Connect EventBus to Cortex SSE:

```python
import asyncio
import httpx
from stack.aperion.foundation.event_bus import EventBus

async def bridge_cortex_events(
    cortex_url: str,
    event_bus: EventBus,
    token: str,
):
    """Bridge Cortex SSE events to local EventBus."""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            f"{cortex_url}/events/stream",
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:])
                    event_bus.emit(
                        f"cortex.{data['type']}",
                        data["payload"],
                        source="cortex",
                    )
```

---

## 5. Configuration

### Environment Variables

```bash
# In aperion-legendary-ai-main
export CORTEX_URL="http://localhost:4949"
export CORTEX_TOKEN="your-token-here"
export USE_CORTEX="true"

# In aperion-state-gateway
export CORTEX_PORT=4949
export CORTEX_SIGNING_KEY="your-signing-key"
export CORTEX_LOG_LEVEL="INFO"
```

### docker-compose.yml (Development)

```yaml
version: "3.8"

services:
  cortex:
    build: ../aperion-state-gateway
    ports:
      - "4949:4949"
    volumes:
      - cortex_data:/app/data
    environment:
      - CORTEX_SIGNING_KEY=dev-key
      - CORTEX_LOG_LEVEL=DEBUG
  
  aperion:
    build: .
    depends_on:
      - cortex
    environment:
      - CORTEX_URL=http://cortex:4949
      - CORTEX_TOKEN=dev-token
      - USE_CORTEX=true

volumes:
  cortex_data:
```

---

## 6. Testing Integration

```python
# tests/integration/test_cortex_integration.py

import pytest
from stack.aperion.clients.cortex_client import CortexClient, CortexConfig

@pytest.fixture
def cortex_client():
    """Create Cortex client for testing."""
    config = CortexConfig(
        base_url=os.getenv("CORTEX_URL", "http://localhost:4949"),
        token=os.getenv("CORTEX_TOKEN", "test-token"),
    )
    client = CortexClient(config)
    yield client
    client.close()

def test_cortex_health(cortex_client):
    """Verify Cortex is healthy."""
    assert cortex_client.health_check() is True

def test_snapshot_retrieval(cortex_client):
    """Verify snapshot can be retrieved."""
    snapshot = cortex_client.get_snapshot()
    assert "context_pack" in snapshot
    assert "signature" in snapshot

def test_audit_recording(cortex_client):
    """Verify audit records are created."""
    run_id = cortex_client.record_audit(
        agent_name="test-agent",
        method_name="test-method",
        prompt="test prompt",
        raw_output="test output",
        latency_ms=100.0,
    )
    assert run_id is not None
```

---

## 7. Rollback Plan

If issues occur, revert to old implementation:

```bash
# Disable Cortex
export USE_CORTEX=false

# Restart services
make restart
```

Keep old code paths available until Cortex is proven stable.

---

## 8. Monitoring

### Health Dashboard Endpoints

| Service | Health | Ready |
|---------|--------|-------|
| Cortex | `GET /healthz` | `GET /ready` |
| Aperion | `GET /api/health` | `GET /api/ready` |

### Key Metrics to Monitor

- Cortex response latency (p50, p95, p99)
- Vector search query times
- Audit log write throughput
- Error rates by endpoint
- Memory usage trends

---

## References

- Cortex API: `/home/dreamboat/projects/aperion-state-gateway/README.md`
- Gap Analysis: `/home/dreamboat/projects/aperion-state-gateway/ref/GAP_ANALYSIS.md`
- Aperion Architecture: `/home/dreamboat/projects/aperion/aperion-legendary-ai-main/docs/ARCHITECTURE.md`
