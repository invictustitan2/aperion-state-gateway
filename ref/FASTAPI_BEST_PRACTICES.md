# FastAPI Best Practices (2025-2026)

> Reference documentation for production-grade FastAPI services
> Sources: orchestrator.dev, fastlaunchapi.dev, hrekov.com, thelinuxcode.com

---

## 1. Dependency Injection Patterns

### 1.1 Use Layered Dependencies

Chain dependencies for authentication → authorization → business logic:

```python
def verify_jwt(token: str = Depends(oauth2_scheme)):
    """First layer: verify token is valid."""
    payload = decode_jwt(token)
    return payload["user_id"]

def get_user(user_id: int = Depends(verify_jwt)):
    """Second layer: load user from database."""
    return db.get_user(user_id)

def require_admin(user: User = Depends(get_user)):
    """Third layer: check permissions."""
    if not user.is_admin:
        raise HTTPException(403)
    return user

@app.get("/admin/dashboard")
def admin_dashboard(user: User = Depends(require_admin)):
    return {"message": f"Welcome, {user.name}"}
```

### 1.2 Use `yield` for Resource Cleanup

Generator-based dependencies ensure proper cleanup:

```python
def get_db():
    """Database session with automatic cleanup."""
    db = Session()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
```

### 1.3 Avoid Per-Request Heavy Object Creation

❌ **Bad**: Creates new HTTP client per request
```python
async def call_external():
    async with httpx.AsyncClient() as client:
        return await client.get("...")
```

✅ **Good**: Reuse client from app state
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()

async def get_http_client(request: Request):
    return request.app.state.http_client
```

---

## 2. Performance Optimization

### 2.1 Use Async for I/O-Bound Operations

```python
# ✅ Async for database, HTTP, file I/O
@app.get("/data")
async def get_data():
    async with aiohttp.ClientSession() as session:
        data = await session.get("...")
    return data

# ❌ Don't block the event loop
@app.get("/data")
def get_data():
    requests.get("...")  # Blocks!
```

### 2.2 Thread Pool for CPU-Bound Work

```python
from concurrent.futures import ThreadPoolExecutor
from functools import partial

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/process")
async def process_data(data: bytes):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        partial(cpu_intensive_function, data)
    )
    return result
```

### 2.3 Response Caching

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="cortex:")

@app.get("/expensive")
@cache(expire=300)  # 5 minutes
async def expensive_operation():
    return compute_expensive_result()
```

---

## 3. Error Handling

### 3.1 Custom Exception Handlers

```python
class BusinessError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message

@app.exception_handler(BusinessError)
async def business_error_handler(request: Request, exc: BusinessError):
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.code,
            "message": exc.message,
            "correlation_id": request.state.correlation_id,
        },
    )
```

### 3.2 Structured Error Responses

```python
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict[str, Any] | None = None
    correlation_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

---

## 4. Testing

### 4.1 Dependency Overrides

```python
def override_get_db():
    return MockDatabase()

def override_external_service():
    return MockExternalService()

@pytest.fixture
def client():
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[external_service] = override_external_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

### 4.2 Async Test Fixtures

```python
import pytest_asyncio

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_async_endpoint(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
```

---

## 5. Project Structure

```
src/
├── cortex/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── service/             # API layer
│   │   ├── app.py           # FastAPI app factory
│   │   ├── router.py        # Endpoints
│   │   ├── models.py        # Pydantic schemas
│   │   ├── auth.py          # Authentication
│   │   └── dependencies.py  # Shared dependencies
│   ├── memory/              # Domain: memory management
│   │   ├── context.py
│   │   ├── vector_store.py
│   │   └── hot_state.py
│   ├── persistence/         # Data layer
│   │   ├── database.py
│   │   └── ledger.py
│   └── core/                # Shared utilities
│       ├── config.py
│       ├── logging.py
│       └── exceptions.py
```

---

## 6. Configuration

### 6.1 Pydantic Settings

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CORTEX_",
        case_sensitive=False,
    )
    
    debug: bool = False
    database_url: str = "sqlite:///data/cortex.db"
    signing_key: str = ""
    port: int = 4949
    
    @property
    def is_production(self) -> bool:
        return not self.debug

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## 7. Security Checklist

- [ ] Use HTTPS in production (TLS termination at load balancer)
- [ ] Set secure CORS origins (not `*`)
- [ ] Implement rate limiting
- [ ] Validate all input with Pydantic
- [ ] Use parameterized queries (no SQL injection)
- [ ] Sanitize error messages (no stack traces in production)
- [ ] Implement request size limits
- [ ] Set security headers (HSTS, CSP, etc.)

---

## References

- https://orchestrator.dev/blog/2025-1-30-fastapi-production-patterns/
- https://fastlaunchapi.dev/blog/fastapi-best-practices-production-2026/
- https://www.hrekov.com/blog/advanced-fastapi-dependency-injection
- https://thelinuxcode.com/dependency-injection-in-fastapi-2026-playbook-for-modular-testable-apis/
