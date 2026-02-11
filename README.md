# Aperion State Gateway — "The Cortex"

> Unified memory and state management service for the Aperion AI platform.

The Cortex separates **Reasoning** (transient LLM processing) from **Memory** (persistent storage), providing a centralized service for context packing, vector search, and the event ledger.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        THE CORTEX                                │
│                     (Port 4949)                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   HOT       │  │   WARM      │  │         COLD            │  │
│  │   STATE     │  │   CONTEXT   │  │        STORAGE          │  │
│  │             │  │             │  │                         │  │
│  │ - Sessions  │  │ - Token     │  │ - Vector DB (Chroma)    │  │
│  │ - Fast KV   │  │   Packing   │  │ - SQLite Archives       │  │
│  │ - TTL Cache │  │ - LLM-ready │  │ - Training Ledger       │  │
│  │             │  │   Output    │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │     API Endpoints (REST)      │
            │                               │
            │  GET  /state/snapshot         │
            │  POST /vector/search          │
            │  GET  /db/schema/{name}       │
            │  POST /audit/record           │
            │  GET  /events/stream (SSE)    │
            └───────────────────────────────┘
```

## Memory Tiering

### Hot State
Active session data with low-latency access. In-memory with optional Redis backend.

### Warm Context
The "Context Window" manager. Accepts raw messages/files and returns a token-optimized `ContextPack` ready for the LLM.

### Cold Storage
Long-term persistence via Vector DB (ChromaDB/FAISS) and SQLite.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the service
aperion-cortex --port 4949

# Or with uvicorn directly
uvicorn aperion_cortex.service.app:app --host 0.0.0.0 --port 4949
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/state/snapshot` | GET | Returns signed ContextPack |
| `/vector/search` | POST | Semantic search |
| `/db/list` | GET | List databases |
| `/db/schema/{name}` | GET | Database introspection |
| `/audit/record` | POST | Log agent execution |
| `/events/stream` | GET | SSE event stream |
| `/healthz` | GET | Health check |

## Docker

```bash
docker build -t aperion-cortex .
docker run -p 4949:4949 aperion-cortex
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORTEX_PORT` | `4949` | Service port |
| `CORTEX_SIGNING_KEY` | (required) | Key for signing ContextPacks |
| `CORTEX_DB_PATH` | `data/cortex.db` | SQLite database path |
| `CORTEX_VECTOR_BACKEND` | `chroma` | Vector store: `chroma` or `faiss` |
| `CORTEX_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CORTEX_LOG_LEVEL` | `INFO` | Logging level |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=aperion_cortex --cov-report=html

# Lint
ruff check src tests

# Type check
mypy src
```

## License

MIT
