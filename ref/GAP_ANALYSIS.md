# Cortex Gap Analysis - Updated 2026-02-08

> **Status**: Milestones 1-2 Complete | 47 tests passing | 23 source files
> **Last Analysis**: 2026-02-08 11:13 UTC

---

## Completed Items (No Longer Gaps)

| Original Gap | Resolution | File |
|--------------|------------|------|
| ~~SQLite WAL Mode~~ | ✅ WAL + synchronous=NORMAL + busy_timeout | `database.py` |
| ~~Embedding Cache~~ | ✅ Global `_EMBEDDING_MODEL_CACHE` | `vector_store.py` |
| ~~Graceful Shutdown~~ | ✅ Signal handlers + executor cleanup | `app.py` |
| ~~Health Checks~~ | ✅ /healthz verifies all dependencies | `app.py` |
| ~~Metrics Export~~ | ✅ /metrics Prometheus endpoint | `metrics.py` |
| ~~Correlation ID~~ | ✅ X-Correlation-ID middleware | `middleware.py` |
| ~~Circuit Breaker~~ | ✅ OpenAI embedding protection | `circuit_breaker.py` |
| ~~Rate Limiting~~ | ✅ Token bucket with headers | `rate_limit.py` |
| ~~Retry with Backoff~~ | ✅ Exponential backoff for OpenAI | `retry.py` |
| ~~Structured Logging~~ | ✅ structlog with JSON/console | `logging.py` |
| ~~Input Validation~~ | ✅ Max sizes on all inputs | `models.py` |

---

## Remaining Gaps by Priority

### 🔴 Critical (Blocks Integration)

#### Gap 1: No Python Client SDK
**Impact**: Agents must use raw HTTP calls; integration is fragile
**Current**: No client library exists
**Required**: 
```python
from cortex_client import CortexClient
client = CortexClient("http://localhost:4949", token="...")
pack = await client.get_snapshot()
hits = await client.vector_search("query")
```
**Effort**: 2 hours
**Milestone**: 3

#### Gap 2: No Event Bus Bridge
**Impact**: State changes don't propagate to main Aperion EventBus
**Current**: SSE endpoint exists but isn't connected
**Required**: Bridge that subscribes to Cortex SSE and emits to EventBus
**Effort**: 1 hour
**Milestone**: 3

---

### 🟠 High (Production Quality)

#### Gap 3: FAISS Backend Not Implemented
**Impact**: No alternative to ChromaDB; vendor lock-in
**Current**: `raise NotImplementedError("FAISS backend not yet implemented")`
**Required**: Working FAISS implementation for environments without ChromaDB
**Effort**: 2 hours
**Milestone**: 6

#### Gap 4: No Token Expiration Cleanup
**Impact**: Memory leak from stale tokens over time
**Current**: Tokens checked on use but never cleaned
**Required**: Periodic background task to purge expired tokens
**Effort**: 30 min
**Milestone**: 4

#### Gap 5: Audit Ledger Not Tamper-Proof
**Impact**: Entries can be modified without detection
**Current**: Plain JSONL append
**Required**: Hash chaining where each entry includes hash of previous
**Effort**: 1 hour
**Milestone**: 4

---

### 🟡 Medium (Best Practices)

#### Gap 6: Single Tokenizer (cl100k_base only)
**Impact**: Inaccurate token counts for Claude, Llama, Mistral
**Current**: Always uses GPT-4 tokenizer
**Required**: Model-specific tokenizer selection
**Effort**: 1 hour
**Milestone**: 5

#### Gap 7: No Context Summarization
**Impact**: Long conversations lose early context entirely
**Current**: Truncation-based packing (drops items)
**Required**: Sliding window with LLM summarization
**Effort**: 2 hours
**Milestone**: 5

#### Gap 8: No Ledger Rotation
**Impact**: Disk exhaustion over time
**Current**: Single JSONL file grows forever
**Required**: Time-based rotation with archival
**Effort**: 30 min
**Milestone**: 4

#### Gap 9: No Service Discovery
**Impact**: Hardcoded port 4949; can't scale dynamically
**Current**: Static configuration only
**Required**: Consul/etcd or DNS-based discovery
**Effort**: 1 hour
**Milestone**: 3

---

### 🟢 Low (Nice to Have)

#### Gap 10: No Embedding Quantization
**Impact**: Higher memory usage than necessary
**Current**: Full FP32 embeddings
**Required**: FP16 option for 50% memory reduction
**Effort**: 1 hour
**Milestone**: 6

#### Gap 11: No ONNX Runtime
**Impact**: Slower CPU inference for HuggingFace
**Current**: PyTorch inference
**Required**: ONNX export for 2x speedup
**Effort**: 1 hour
**Milestone**: 6

#### Gap 12: No Snapshot Recovery
**Impact**: Slow ledger replay on restart
**Current**: Must replay all events
**Required**: Periodic state snapshots
**Effort**: 1 hour
**Milestone**: 4

---

## Updated Priority Matrix

| Gap | Effort | Impact | Milestone | Priority |
|-----|--------|--------|-----------|----------|
| Client SDK | 2 hr | 🔴 Critical | M3 | **Do Next** |
| Event Bus Bridge | 1 hr | 🔴 Critical | M3 | **Do Next** |
| Service Discovery | 1 hr | 🟡 Medium | M3 | **Do Next** |
| Hash-Chained Ledger | 1 hr | 🟠 High | M4 | Phase 2 |
| Token Cleanup | 30 min | 🟠 High | M4 | Phase 2 |
| Ledger Rotation | 30 min | 🟡 Medium | M4 | Phase 2 |
| Model Tokenizers | 1 hr | 🟡 Medium | M5 | Phase 3 |
| Context Summarization | 2 hr | 🟡 Medium | M5 | Phase 3 |
| FAISS Backend | 2 hr | 🟠 High | M6 | Phase 3 |
| Embedding Quantization | 1 hr | 🟢 Low | M6 | Phase 4 |
| ONNX Runtime | 1 hr | 🟢 Low | M6 | Phase 4 |
| Snapshot Recovery | 1 hr | 🟢 Low | M4 | Phase 4 |

---

## Recommended Milestone Order

```
Milestone 3: Aperion Integration (NEXT - 4 hours)
├── Python Client SDK (2 hr) - enables clean integration
├── Event Bus Bridge (1 hr) - real-time state sync
└── Service Discovery (1 hr) - containerization support

Milestone 4: Ledger Hardening (3 hours)
├── Hash chaining (1 hr) - tamper-proof audit
├── Token cleanup (30 min) - memory management
├── Log rotation (30 min) - disk management
└── Snapshot recovery (1 hr) - fast restart

Milestone 5: Context Intelligence (4 hours)
├── Model tokenizers (1 hr) - accurate counts
├── Sliding window (2 hr) - long context
└── Priority decay (1 hr) - recency weighting

Milestone 6: Vector Enhancements (4 hours)
├── FAISS backend (2 hr) - alternative store
├── FP16 quantization (1 hr) - memory reduction
└── ONNX runtime (1 hr) - inference speedup
```

---

## Dependency Graph

```
     ┌──────────────────────────────────────────────────────────────┐
     │              Milestones 1-2 COMPLETE ✅                      │
     │  (Production hardening, observability, resilience)          │
     └──────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────┐
              │     Milestone 3: Integration        │  ◄── DO NEXT
              │  (Client SDK, Event Bridge, Disc.)  │
              └─────────────────┬───────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  M4: Ledger     │  │  M5: Context    │  │  M6: Vector     │
│  (Tamper-proof, │  │  (Tokenizers,   │  │  (FAISS, FP16,  │
│   rotation)     │  │   summarize)    │  │   ONNX)         │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Production Ready    │
                    │   Full Integration    │
                    └───────────────────────┘
```

M3 is blocking: Without the client SDK, agents can't cleanly use Cortex.
M4, M5, M6 can run in parallel after M3.
