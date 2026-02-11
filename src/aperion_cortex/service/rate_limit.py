"""Rate limiting for the Cortex service.

Provides token-bucket rate limiting to protect against resource exhaustion.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    # Requests per window
    requests_per_minute: int = 60
    requests_per_second: int = 10
    
    # Burst allowance (token bucket capacity)
    burst_size: int = 20
    
    # Paths to exclude from rate limiting
    excluded_paths: tuple[str, ...] = ("/healthz", "/ready", "/metrics")


@dataclass 
class TokenBucket:
    """Token bucket for rate limiting."""
    
    capacity: float
    tokens: float = field(init=False)
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time, init=False)
    
    def __post_init__(self):
        self.tokens = self.capacity
    
    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    @property
    def time_until_available(self) -> float:
        """Seconds until at least 1 token is available."""
        if self.tokens >= 1:
            return 0.0
        tokens_needed = 1 - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """Rate limiter using token bucket algorithm.
    
    Supports per-client rate limiting based on token or IP.
    """
    
    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._buckets: dict[str, TokenBucket] = {}
        self._lock_time: dict[str, float] = defaultdict(float)
    
    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create bucket for a key."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=float(self.config.burst_size),
                refill_rate=self.config.requests_per_second,
            )
        return self._buckets[key]
    
    def check(self, key: str) -> tuple[bool, float]:
        """Check if request is allowed.
        
        Args:
            key: Rate limit key (e.g., token or IP)
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        bucket = self._get_bucket(key)
        
        if bucket.consume():
            return True, 0.0
        
        retry_after = bucket.time_until_available
        return False, retry_after
    
    def get_headers(self, key: str) -> dict[str, str]:
        """Get rate limit headers for response."""
        bucket = self._get_bucket(key)
        return {
            "X-RateLimit-Limit": str(self.config.requests_per_minute),
            "X-RateLimit-Remaining": str(int(bucket.tokens)),
            "X-RateLimit-Reset": str(int(time.time() + 60)),
        }
    
    def cleanup_old_buckets(self, max_age_seconds: float = 3600) -> int:
        """Remove buckets that haven't been used recently."""
        now = time.time()
        to_remove = []
        
        for key, bucket in self._buckets.items():
            if now - bucket.last_refill > max_age_seconds:
                to_remove.append(key)
        
        for key in to_remove:
            del self._buckets[key]
        
        return len(to_remove)


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter(config: RateLimitConfig | None = None) -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(config)
    return _rate_limiter


def _get_client_key(request: Request) -> str:
    """Extract rate limit key from request.
    
    Priority:
    1. Authorization token (if present)
    2. X-Forwarded-For header (for proxied requests)
    3. Client IP address
    """
    # Try to use auth token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        # Use hash of token to avoid storing tokens
        import hashlib
        token_hash = hashlib.sha256(auth_header.encode()).hexdigest()[:16]
        return f"token:{token_hash}"
    
    # Try X-Forwarded-For for proxied requests
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take first IP (original client)
        client_ip = forwarded_for.split(",")[0].strip()
        return f"ip:{client_ip}"
    
    # Fall back to direct client IP
    if request.client:
        return f"ip:{request.client.host}"
    
    return "anonymous"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits."""
    
    def __init__(self, app, config: RateLimitConfig | None = None):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.limiter = get_rate_limiter(config)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip excluded paths
        if request.url.path in self.config.excluded_paths:
            return await call_next(request)
        
        # Get client key and check rate limit
        client_key = _get_client_key(request)
        allowed, retry_after = self.limiter.check(client_key)
        
        if not allowed:
            # Return 429 Too Many Requests
            return Response(
                content=f'{{"error": "Rate limit exceeded", "retry_after": {retry_after:.1f}}}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    **self.limiter.get_headers(client_key),
                },
            )
        
        # Process request and add rate limit headers
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        for key, value in self.limiter.get_headers(client_key).items():
            response.headers[key] = value
        
        return response


__all__ = [
    "RateLimitConfig",
    "RateLimiter",
    "RateLimitMiddleware",
    "get_rate_limiter",
]
