# ChromaDB Production Guide

> Reference documentation for production ChromaDB deployment
> Sources: docs.trychroma.com, cookbook.chromadb.dev, github.com/chroma-core/chroma

---

## 1. Memory Sizing

### RAM Requirements

ChromaDB keeps HNSW indexes entirely in RAM. Sizing formula:

```
Max Collection Size (millions) ≈ RAM (GB) × 0.245
```

For 1024-dimensional embeddings:
- 8GB RAM → ~2M documents
- 16GB RAM → ~4M documents
- 32GB RAM → ~8M documents

**Always reserve 20-30% headroom** for OS and ChromaDB overhead.

### Monitoring

```python
import psutil

def check_memory_health():
    mem = psutil.virtual_memory()
    if mem.percent > 85:
        logger.warning(f"High memory usage: {mem.percent}%")
        return False
    return True
```

---

## 2. Known Issues (2025)

### 2.1 Memory Leaks

**Issue**: Memory steadily increases after collection create/delete cycles.

**Mitigation**:
- Minimize collection creation/deletion in production
- Upgrade to ChromaDB >= 1.0.6
- Schedule periodic service restarts
- Monitor with `process.memory_info().rss`

**Reference**: https://github.com/chroma-core/chroma/issues/4024

### 2.2 Disk Bloat After Reindexing

**Issue**: Disk usage grows after many insert/delete cycles.

**Mitigation**:
- Periodic vacuum/compaction
- Monitor disk usage
- Archive and recreate collections periodically

**Reference**: https://github.com/chroma-core/chroma/issues/4737

---

## 3. Performance Optimization

### 3.1 Batch Operations

Always batch inserts for efficiency:

```python
# ✅ Good: Batch insert
collection.add(
    documents=documents[:1000],
    ids=[f"doc_{i}" for i in range(1000)],
    metadatas=metadatas[:1000],
)

# ❌ Bad: Single inserts
for doc in documents:
    collection.add(documents=[doc], ids=[doc.id])
```

### 3.2 LRU Cache for Collections

Enable collection unloading for memory management:

```python
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_data",
    settings=Settings(
        # Enable LRU cache to unload unused collections
        chroma_memory_limit_bytes=1024 * 1024 * 1024,  # 1GB
    )
)
```

### 3.3 Checkpointing

For heavy write loads, trigger manual checkpoints:

```python
# ChromaDB handles this internally, but for SQLite backend:
# The underlying SQLite uses WAL mode with auto-checkpoint
```

---

## 4. Embedding Optimization

### 4.1 Choose Efficient Models

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| text-embedding-3-small | 1536 | API | Best |

### 4.2 Quantization

Reduce memory with FP16 embeddings:

```python
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")
model = model.half()  # FP16

# For GPU: model.to("cuda").half()
```

### 4.3 ONNX Backend

Faster CPU inference:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    backend="onnx",
)
```

---

## 5. High Availability Patterns

### 5.1 Single-Node (Current)

```
┌─────────────────┐
│   Cortex API    │
├─────────────────┤
│   ChromaDB      │
│   (Embedded)    │
├─────────────────┤
│   Local Disk    │
└─────────────────┘
```

**Pros**: Simple, fast, no network overhead
**Cons**: Single point of failure, vertical scaling only

### 5.2 Client-Server Mode

```
┌─────────────┐     ┌─────────────┐
│ Cortex API  │────▶│ Chroma      │
│ (Replica 1) │     │ Server      │
└─────────────┘     │             │
┌─────────────┐     │ (Dedicated  │
│ Cortex API  │────▶│  Host)      │
│ (Replica 2) │     └─────────────┘
└─────────────┘
```

**Migration**:
```python
import chromadb

# Instead of:
# client = chromadb.PersistentClient(path="./data")

# Use:
client = chromadb.HttpClient(
    host="chroma-server",
    port=8000,
)
```

### 5.3 Future: Distributed (Milvus/Pinecone)

For massive scale, consider purpose-built distributed vector DBs.

---

## 6. Deployment Checklist

- [ ] Set memory limits in container (leave 20% headroom)
- [ ] Mount persistent volume for `/data`
- [ ] Monitor memory usage (alert at 85%)
- [ ] Monitor disk usage (alert at 80%)
- [ ] Schedule periodic restarts (weekly for memory leak mitigation)
- [ ] Use batch operations for all inserts
- [ ] Limit collection count (< 100 recommended)
- [ ] Test recovery from crashes
- [ ] Backup data directory regularly

---

## 7. Docker Configuration

```dockerfile
FROM python:3.12-slim

# Install dependencies
RUN pip install chromadb sentence-transformers

# Set memory limits via orchestrator, not here
ENV CHROMA_SERVER_HOST=0.0.0.0
ENV CHROMA_SERVER_PORT=8000

# Persist data
VOLUME /data

CMD ["chroma", "run", "--path", "/data"]
```

```yaml
# docker-compose.yml
services:
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

---

## References

- https://docs.trychroma.com/production/administration/performance
- https://cookbook.chromadb.dev/strategies/memory-management/
- https://cookbook.chromadb.dev/running/performance-tips/
- https://github.com/chroma-core/chroma/issues/4024
- https://github.com/chroma-core/chroma/issues/4737
