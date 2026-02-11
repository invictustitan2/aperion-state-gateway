# Reference Documentation Index

> aperion-state-gateway / ref
> Generated: 2026-02-08

---

## Overview

This folder contains reference documentation, best practices, and research findings
for improving the Cortex service and integrating it with aperion-legendary-ai-main.

---

## Documents

### 📊 GAP_ANALYSIS.md
**Priority Matrix for Improvements**

Comprehensive analysis of gaps between current implementation and production requirements.
Includes priority levels (Critical/High/Medium/Low) and recommended immediate actions.

Key gaps identified:
- SQLite WAL mode not enabled (🔴 Critical)
- No embedding model caching (🟠 High)
- No circuit breakers (🟠 High)
- No comprehensive health checks (🟠 High)
- No tamper-proofing on audit ledger (🟠 High)

---

### 🚀 FASTAPI_BEST_PRACTICES.md
**Production FastAPI Patterns (2025-2026)**

- Layered dependency injection
- Resource cleanup with `yield`
- Performance optimization (async, caching)
- Error handling patterns
- Testing with dependency overrides
- Project structure recommendations

---

### 🔍 CHROMADB_PRODUCTION.md
**Vector Store Deployment Guide**

- Memory sizing formulas
- Known issues and mitigations (memory leaks, disk bloat)
- Performance optimization (batching, LRU cache)
- Embedding model optimization
- High availability patterns
- Docker configuration

---

### 📝 CONTEXT_ENGINEERING.md
**LLM Token Management**

- Token counting with tiktoken
- Model-encoding mapping
- Context window limits (2025 models)
- Optimization strategies:
  - Priority-based packing
  - Sliding window
  - Summarization
  - RAG-based retrieval
- Anti-patterns to avoid

---

### 📜 EVENT_SOURCING.md
**Immutable Audit Ledger Best Practices**

- Core principles (immutability, event design)
- Event structure requirements
- Storage patterns (JSONL, SQLite, PostgreSQL)
- Tamper-proofing (hash chaining, signatures)
- Snapshotting for performance
- Retention and rotation policies

---

### 💾 SQLITE_PRODUCTION.md
**Database Configuration Guide**

- WAL mode setup and configuration
- Connection pooling patterns
- Transaction best practices
- Checkpointing and monitoring
- Indexing strategies
- Backup procedures
- When to migrate to PostgreSQL

---

### 🔗 INTEGRATION_GUIDE.md
**Cortex ↔ Aperion Integration**

- Architecture overview
- Current connection points
- Client SDK implementation
- Migration steps (phased approach)
- Configuration (env vars, Docker)
- Testing integration
- Rollback plan

---

### 🛡️ RESILIENCE_PATTERNS.md
**Fault Tolerance Patterns**

- Circuit breaker implementation
- Health check patterns (liveness/readiness)
- Rate limiting
- Graceful shutdown
- Retry with exponential backoff
- Fallback patterns (cache, default)

---

## Quick Reference

### Immediate Actions (Do First)

```bash
# 1. Enable SQLite WAL mode in database.py
# Add after connection creation:
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
conn.execute("PRAGMA busy_timeout=5000;")

# 2. Add embedding model caching in vector_store.py
# Add module-level cache:
_MODEL_CACHE: dict[str, Any] = {}

# 3. Implement comprehensive health checks
# See RESILIENCE_PATTERNS.md section 2

# 4. Add graceful shutdown
# See RESILIENCE_PATTERNS.md section 4
```

### Key Metrics to Monitor

| Metric | Target | Alert |
|--------|--------|-------|
| Request latency p99 | < 500ms | > 1s |
| Error rate | < 0.1% | > 1% |
| Memory usage | < 80% | > 85% |
| SQLite WAL size | < 100MB | > 500MB |
| ChromaDB queries/sec | > 100 | < 50 |

### External Resources

- FastAPI: https://fastapi.tiangolo.com/
- ChromaDB: https://docs.trychroma.com/
- tiktoken: https://github.com/openai/tiktoken
- pybreaker: https://github.com/danielfm/pybreaker
- SQLite WAL: https://sqlite.org/wal.html

---

## Usage

1. **Before implementing a feature**: Check GAP_ANALYSIS.md for known issues
2. **For code patterns**: Reference the specific best practices doc
3. **For integration work**: Follow INTEGRATION_GUIDE.md
4. **For production deployment**: Review RESILIENCE_PATTERNS.md

---

## Maintenance

Update these docs when:
- New best practices emerge
- Bugs are discovered
- Dependencies are upgraded
- Integration patterns change

Last updated: 2026-02-08
