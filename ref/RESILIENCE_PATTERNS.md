# Service Resilience Patterns

> Reference documentation for circuit breakers, health checks, and fault tolerance
> Sources: baeldung.com, softwarepatternslexicon.com, aritro.in

---

## 1. Circuit Breaker Pattern

### 1.1 States

```
     Success
        │
        ▼
┌───────────────┐      Failures > Threshold      ┌──────────────┐
│    CLOSED     │ ──────────────────────────────▶│    OPEN      │
│ (Normal Flow) │                                │ (Fail Fast)  │
└───────────────┘                                └──────────────┘
        ▲                                               │
        │                                               │
        │                                        Timeout
        │                                               │
        │             Success                           ▼
        └────────────────────────────────────── ┌──────────────┐
                                                │  HALF-OPEN   │
                                                │ (Test Call)  │
                                                └──────────────┘
                                                        │
                                        Failure         │
                                        ────────────────┘
```

### 1.2 Implementation with pybreaker

```python
from pybreaker import CircuitBreaker, CircuitBreakerError

# Configure breaker
embedding_breaker = CircuitBreaker(
    fail_max=5,           # Open after 5 failures
    reset_timeout=30,     # Try again after 30 seconds
    exclude=[ValueError], # Don't count these as failures
)

@embedding_breaker
def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings with circuit breaker protection."""
    return embedding_model.embed(texts)

# Usage with fallback
def safe_get_embeddings(texts: list[str]) -> list[list[float]]:
    try:
        return get_embeddings(texts)
    except CircuitBreakerError:
        # Fallback: return cached or default embeddings
        logger.warning("Embedding service circuit open, using fallback")
        return get_cached_embeddings(texts)
```

### 1.3 Custom Circuit Breaker

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

class CircuitBreaker:
    """Custom circuit breaker implementation."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: datetime | None = None
        self._half_open_calls = 0
        self._lock = Lock()
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time:
                    elapsed = datetime.now() - self._last_failure_time
                    if elapsed.total_seconds() >= self.config.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
            return self._state
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        state = self.state
        
        if state == CircuitState.OPEN:
            raise CircuitBreakerError(f"Circuit {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.config.half_open_max_calls:
                    # Recovery confirmed
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
    
    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed during test, go back to open
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN

class CircuitBreakerError(Exception):
    pass
```

---

## 2. Health Checks

### 2.1 Liveness vs Readiness

| Probe | Purpose | Failure Action |
|-------|---------|----------------|
| Liveness | Is process alive? | Restart container |
| Readiness | Can handle traffic? | Remove from load balancer |
| Startup | Has finished init? | Block other probes |

### 2.2 Comprehensive Health Check

```python
from fastapi import FastAPI
from datetime import datetime, timezone
from typing import Callable

app = FastAPI()

@dataclass
class HealthCheck:
    name: str
    check: Callable[[], bool]
    critical: bool = True  # If False, degraded instead of unhealthy

health_checks: list[HealthCheck] = []

def register_health_check(
    name: str,
    check: Callable[[], bool],
    critical: bool = True,
):
    """Register a health check."""
    health_checks.append(HealthCheck(name, check, critical))

# Register checks
register_health_check("database", lambda: check_database_connection())
register_health_check("vector_store", lambda: check_vector_store())
register_health_check("disk_space", lambda: check_disk_space(), critical=False)

@app.get("/healthz")
async def health():
    """Comprehensive health check."""
    results = {}
    all_critical_healthy = True
    all_healthy = True
    
    for check in health_checks:
        try:
            is_healthy = check.check()
            results[check.name] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "critical": check.critical,
            }
            if not is_healthy:
                all_healthy = False
                if check.critical:
                    all_critical_healthy = False
        except Exception as e:
            results[check.name] = {
                "status": "unhealthy",
                "error": str(e),
                "critical": check.critical,
            }
            all_healthy = False
            if check.critical:
                all_critical_healthy = False
    
    if all_critical_healthy:
        status = "ok" if all_healthy else "degraded"
        status_code = 200
    else:
        status = "unhealthy"
        status_code = 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": results,
        },
    )

@app.get("/ready")
async def ready():
    """Readiness probe - are all dependencies available?"""
    # Check critical dependencies only
    for check in health_checks:
        if check.critical:
            if not check.check():
                return JSONResponse(
                    status_code=503,
                    content={"ready": False, "reason": f"{check.name} not ready"},
                )
    return {"ready": True}

@app.get("/live")
async def live():
    """Liveness probe - is the process alive?"""
    # Minimal check - if we can respond, we're alive
    return {"alive": True}
```

### 2.3 Kubernetes Probes

```yaml
# kubernetes deployment
spec:
  containers:
    - name: cortex
      livenessProbe:
        httpGet:
          path: /live
          port: 4949
        initialDelaySeconds: 5
        periodSeconds: 10
        failureThreshold: 3
      
      readinessProbe:
        httpGet:
          path: /ready
          port: 4949
        initialDelaySeconds: 10
        periodSeconds: 5
        failureThreshold: 3
      
      startupProbe:
        httpGet:
          path: /healthz
          port: 4949
        initialDelaySeconds: 0
        periodSeconds: 5
        failureThreshold: 30  # 2.5 min startup time
```

---

## 3. Rate Limiting

### 3.1 With slowapi

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/vector/search")
@limiter.limit("100/minute")
async def vector_search(request: Request, body: VectorSearchRequest):
    return await do_search(body)
```

### 3.2 Token-Based Rate Limiting

```python
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock

class TokenRateLimiter:
    """Rate limiter per API token."""
    
    def __init__(self, rate: int, period_seconds: int):
        self.rate = rate
        self.period = timedelta(seconds=period_seconds)
        self._requests: dict[str, list[datetime]] = defaultdict(list)
        self._lock = Lock()
    
    def is_allowed(self, token: str) -> bool:
        """Check if request is allowed."""
        now = datetime.now()
        cutoff = now - self.period
        
        with self._lock:
            # Clean old requests
            self._requests[token] = [
                t for t in self._requests[token] if t > cutoff
            ]
            
            if len(self._requests[token]) >= self.rate:
                return False
            
            self._requests[token].append(now)
            return True

# Usage
rate_limiter = TokenRateLimiter(rate=100, period_seconds=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    
    if not rate_limiter.is_allowed(token):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers={"Retry-After": "60"},
        )
    
    return await call_next(request)
```

---

## 4. Graceful Shutdown

```python
import asyncio
import signal
from contextlib import asynccontextmanager

shutdown_event = asyncio.Event()
in_flight_requests = 0
request_lock = asyncio.Lock()

@asynccontextmanager
async def track_request():
    """Track in-flight requests for graceful shutdown."""
    global in_flight_requests
    async with request_lock:
        in_flight_requests += 1
    try:
        yield
    finally:
        async with request_lock:
            in_flight_requests -= 1

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with graceful shutdown."""
    
    def handle_shutdown(sig, frame):
        logger.info(f"Received {sig}, initiating shutdown...")
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    yield
    
    # Graceful shutdown
    logger.info("Starting graceful shutdown...")
    
    # Wait for in-flight requests (max 30 seconds)
    for _ in range(60):
        if in_flight_requests == 0:
            break
        logger.info(f"Waiting for {in_flight_requests} requests...")
        await asyncio.sleep(0.5)
    
    # Cleanup resources
    logger.info("Cleaning up resources...")
    await cleanup_database()
    await cleanup_vector_store()
    
    logger.info("Shutdown complete")
```

---

## 5. Retry with Backoff

```python
import asyncio
from functools import wraps
from typing import Type

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator for retry with exponential backoff."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3, exceptions=(httpx.TimeoutException,))
async def call_external_service():
    async with httpx.AsyncClient(timeout=5.0) as client:
        return await client.get("https://api.example.com/data")
```

---

## 6. Fallback Patterns

### 6.1 Cache Fallback

```python
class CacheFallback:
    """Fallback to cached data when primary fails."""
    
    def __init__(self, cache_ttl: int = 3600):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._ttl = timedelta(seconds=cache_ttl)
    
    async def get_with_fallback(
        self,
        key: str,
        primary: Callable[[], Awaitable[Any]],
    ) -> Any:
        try:
            result = await primary()
            # Cache successful result
            self._cache[key] = (result, datetime.now())
            return result
        except Exception as e:
            # Try cache fallback
            if key in self._cache:
                value, cached_at = self._cache[key]
                age = datetime.now() - cached_at
                logger.warning(
                    f"Using cached value (age: {age.seconds}s) "
                    f"due to error: {e}"
                )
                return value
            raise
```

### 6.2 Default Fallback

```python
async def get_embeddings_with_fallback(texts: list[str]) -> list[list[float]]:
    """Get embeddings with graceful fallback."""
    try:
        # Try primary embedding service
        return await embedding_service.embed(texts)
    except CircuitBreakerError:
        # Try backup service
        try:
            return await backup_embedding_service.embed(texts)
        except Exception:
            # Last resort: return zero vectors
            logger.error("All embedding services failed, returning zeros")
            return [[0.0] * 384 for _ in texts]
```

---

## 7. Cortex Implementation Gaps

### Current State
- Basic `/healthz` and `/ready` endpoints
- No circuit breakers
- No rate limiting
- No graceful shutdown
- No retry logic

### Recommended Additions

```python
# src/cortex/service/resilience.py

from pybreaker import CircuitBreaker

# Circuit breakers for external dependencies
embedding_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=30,
    name="embedding_service",
)

chroma_breaker = CircuitBreaker(
    fail_max=3,
    reset_timeout=60,
    name="chroma_db",
)

# Rate limiter
from slowapi import Limiter
limiter = Limiter(key_func=lambda request: request.state.token)

# Health check registry
health_checks = [
    HealthCheck("sqlite", check_sqlite),
    HealthCheck("chromadb", check_chroma),
    HealthCheck("disk", check_disk, critical=False),
]
```

---

## References

- https://www.baeldung.com/cs/microservices-circuit-breaker-pattern
- https://softwarepatternslexicon.com/microservices/8/3/
- https://www.aritro.in/post/fastapi-resiliency-circuit-breakers-rate-limiting-and-external-api-management/
- https://dev.to/lisan_al_gaib/building-a-health-check-microservice-with-fastapi-26jo
