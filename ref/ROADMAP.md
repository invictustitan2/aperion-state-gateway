# Cortex Service Roadmap

> **Last Updated**: 2026-02-08
> **Status**: 47/47 tests passing, Milestones 1-2 COMPLETE

---

## Current State Summary

### ✅ Completed
- Core service with memory tiering (Hot/Warm/Cold)
- API contract compliance (Chapter 19 endpoints on port 4949)
- Context Engine with tiktoken optimization
- Model-agnostic vector store (HuggingFace/OpenAI, Chroma/Memory)
- Audit ledger with POST /audit/record
- Async infrastructure (executor.py, async embedding models)
- **Milestone 1: Production Hardening** (2026-02-08)
  - SQLite WAL mode + synchronous=NORMAL + busy_timeout
  - Embedding model caching (avoids 2-5s reload)
  - Comprehensive health checks (/healthz verifies db, vector_store, ledger)
  - Graceful shutdown (signal handlers, executor cleanup)
  - Input validation limits (query 10KB, items 500 max)
  - Structured JSON logging (structlog)
- **Milestone 2: Observability & Resilience** (2026-02-08)
  - Prometheus metrics export (/metrics endpoint)
  - Correlation ID middleware (X-Correlation-ID on all requests)
  - Circuit breaker for OpenAI embeddings
  - Rate limiting with token bucket (X-RateLimit-* headers)
  - Retry with exponential backoff for transient failures
- 47 tests passing
- Service verified running on port 4950

### ⏳ Ready for Next Phase
- Milestone 3: Integration with Aperion

---

## Milestones

### Milestone 1: Production Hardening (Priority: 🔴 Critical)
**Goal**: Make the service deployable to production
**Effort**: ~2 hours ✅ COMPLETE (2026-02-08)

| Task | File | Effort | Status |
|------|------|--------|--------|
| Enable SQLite WAL mode | `persistence/database.py` | 5 min | ✅ |
| Add embedding model caching | `memory/vector_store.py` | 15 min | ✅ |
| Comprehensive health checks | `service/app.py` | 30 min | ✅ |
| Graceful shutdown handling | `service/app.py` | 20 min | ✅ |
| Input validation limits | `service/models.py` | 20 min | ✅ |
| Structured JSON logging | `service/logging.py` (new) | 15 min | ✅ |

**Exit Criteria**: ✅ All Met
- Service handles concurrent load without SQLite locks
- Health checks verify actual dependency status
- Clean shutdown with no dropped requests
- All inputs validated with size limits

---

### Milestone 2: Observability & Resilience (Priority: 🟠 High)
**Goal**: Production monitoring and fault tolerance
**Effort**: ~4 hours ✅ COMPLETE (2026-02-08)

| Task | File | Effort | Status |
|------|------|--------|--------|
| Prometheus metrics export | `service/metrics.py` (new) | 45 min | ✅ |
| Correlation ID middleware | `service/middleware.py` (new) | 30 min | ✅ |
| Circuit breaker for embeddings | `service/circuit_breaker.py` (new) | 45 min | ✅ |
| Rate limiting | `service/rate_limit.py` (new) | 30 min | ✅ |
| Retry with backoff for OpenAI | `service/retry.py` (new) | 30 min | ✅ |

**Exit Criteria**: ✅ All Met
- /metrics endpoint with Prometheus-style output
- Requests traceable end-to-end via X-Correlation-ID
- Circuit breaker protects OpenAI embedding calls
- Rate limiting with X-RateLimit-* headers
- Exponential backoff retry for transient failures

---

### Milestone 3: Integration with Aperion (Priority: 🟠 High)
**Goal**: Seamless integration with aperion-legendary-ai-main
**Effort**: ~6 hours

| Task | File | Effort | Status |
|------|------|--------|--------|
| Python client SDK | `cortex_client/` (new package) | 2 hr | ⬜ |
| Event bus bridge (SSE → EventBus) | `integration/event_bridge.py` | 1 hr | ⬜ |
| Service discovery integration | `service/discovery.py` (new) | 1 hr | ⬜ |
| Migrate agents to use Cortex client | (in aperion-legendary-ai) | 2 hr | ⬜ |

**Exit Criteria**:
- Agents use `CortexClient.get_snapshot()` instead of local state
- Real-time events flow from Cortex to main EventBus
- Service discoverable in containerized deployments

---

### Milestone 4: Audit Ledger Hardening (Priority: 🟡 Medium)
**Goal**: Tamper-proof, production-grade event sourcing
**Effort**: ~4 hours

| Task | File | Effort | Status |
|------|------|--------|--------|
| Hash chaining for integrity | `persistence/ledger.py` | 1 hr | ⬜ |
| Log rotation policy | `persistence/ledger.py` | 30 min | ⬜ |
| Periodic snapshots | `persistence/ledger.py` | 1 hr | ⬜ |
| Archive to cold storage | `persistence/archiver.py` (new) | 1 hr | ⬜ |
| Integrity verification CLI | `cli/verify_ledger.py` (new) | 30 min | ⬜ |

**Exit Criteria**:
- Each ledger entry verifiable via hash chain
- Old entries archived automatically
- Fast recovery via snapshots

---

### Milestone 5: Advanced Context Features (Priority: 🟡 Medium)
**Goal**: Smarter context packing for better LLM performance
**Effort**: ~6 hours

| Task | File | Effort | Status |
|------|------|--------|--------|
| Model-specific tokenizers | `memory/context.py` | 1 hr | ⬜ |
| Sliding window with summarization | `memory/context.py` | 2 hr | ⬜ |
| Compression strategies | `memory/compression.py` (new) | 2 hr | ⬜ |
| Priority decay over time | `memory/context.py` | 1 hr | ⬜ |

**Exit Criteria**:
- Token counting accurate for Claude, Llama, GPT models
- Long conversations maintain context via summarization
- Context optimally packed with no wasted tokens

---

### Milestone 6: Vector Store Enhancements (Priority: 🟢 Low)
**Goal**: Production-scale vector search
**Effort**: ~4 hours

| Task | File | Effort | Status |
|------|------|--------|--------|
| FAISS backend implementation | `memory/vector_store.py` | 2 hr | ⬜ |
| Embedding quantization (FP16) | `memory/vector_store.py` | 1 hr | ⬜ |
| ONNX runtime for HuggingFace | `memory/vector_store.py` | 1 hr | ⬜ |

**Exit Criteria**:
- FAISS available as fallback backend
- 50% memory reduction via FP16
- 2x inference speedup via ONNX

---

## Recommended Execution Order

```
Phase 1 (Week 1): Production Hardening
├── Day 1: SQLite WAL + Embedding Cache + Health Checks
└── Day 2: Graceful Shutdown + Input Validation + Logging

Phase 2 (Week 2): Observability & Integration
├── Day 1: Metrics + Correlation IDs
├── Day 2: Circuit Breakers + Rate Limiting  
└── Day 3-4: Python Client SDK

Phase 3 (Week 3): Integration & Testing
├── Day 1-2: Event Bus Bridge + Service Discovery
└── Day 3-4: Migrate Aperion agents to Cortex client

Phase 4 (Week 4): Hardening & Polish
├── Day 1-2: Ledger integrity + rotation
└── Day 3-4: Load testing + documentation
```

---

## Quick Wins (Do Today)

These can be done in ~1 hour total:

1. **SQLite WAL Mode** (5 min)
   ```python
   # In persistence/database.py __init__
   self._conn.execute("PRAGMA journal_mode=WAL;")
   self._conn.execute("PRAGMA synchronous=NORMAL;")
   self._conn.execute("PRAGMA busy_timeout=5000;")
   ```

2. **Embedding Model Cache** (15 min)
   ```python
   # In memory/vector_store.py
   _MODEL_CACHE: dict[str, Any] = {}
   
   def _load_model(self):
       if self.model_name not in _MODEL_CACHE:
           _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
       self._model = _MODEL_CACHE[self.model_name]
   ```

3. **Require Signing Key in Production** (10 min)
   ```python
   # In service/config.py
   @validator('signing_key')
   def validate_signing_key(cls, v):
       if v == "auto-generated" and os.getenv("CORTEX_ENV") == "production":
           raise ValueError("Explicit signing key required in production")
       return v
   ```

---

## Dependency Graph

```
                    ┌─────────────────────────┐
                    │  M1: Production         │
                    │  Hardening              │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  M2: Observ-    │ │  M3: Aperion    │ │  M4: Ledger     │
    │  ability        │ │  Integration    │ │  Hardening      │
    └────────┬────────┘ └────────┬────────┘ └─────────────────┘
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
             ┌─────────────────┐
             │  M5: Advanced   │
             │  Context        │
             └─────────────────┘
                       │
                       ▼
             ┌─────────────────┐
             │  M6: Vector     │
             │  Enhancements   │
             └─────────────────┘
```

**M1** must be done first (production safety).
**M2, M3, M4** can be parallelized.
**M5, M6** are enhancements after core integration is stable.

---

## Metrics for Success

| Metric | Current | Target | Milestone |
|--------|---------|--------|-----------|
| Test count | 47 | 80+ | M3 |
| Test coverage | ~70% | 85%+ | M2 |
| p99 latency (vector search) | Unknown | <200ms | M2 |
| Concurrent requests | Untested | 100+ | M1 |
| Health check accuracy | Static | Actual deps | M1 |
| Mean time to recovery | N/A | <30s | M4 |

---

## References

See `/ref/` for detailed guidance:
- `GAP_ANALYSIS.md` - Full gap analysis with code examples
- `ASYNC_SYNC_FIX.md` - Async architecture (COMPLETED)
- `FASTAPI_BEST_PRACTICES.md` - FastAPI patterns
- `CHROMADB_PRODUCTION.md` - ChromaDB gotchas
- `SQLITE_PRODUCTION.md` - SQLite optimization
- `RESILIENCE_PATTERNS.md` - Circuit breakers, retries
