"""Request middleware for the Cortex service.

Provides:
- Correlation ID propagation for distributed tracing
- Request logging with context
"""

from __future__ import annotations

import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Header names for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to propagate or generate correlation IDs.
    
    Correlation IDs enable distributed tracing across services.
    - If incoming request has X-Correlation-ID, use it
    - Otherwise, generate a new UUID
    - Add to response headers for client visibility
    - Bind to structured logging context
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract or generate correlation ID
        correlation_id = request.headers.get(CORRELATION_ID_HEADER)
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Generate unique request ID for this specific request
        request_id = str(uuid.uuid4())[:8]
        
        # Store in request state for access by handlers
        request.state.correlation_id = correlation_id
        request.state.request_id = request_id
        
        # Bind to structured logging context
        try:
            from aperion_cortex.service.logging import bind_context, clear_context
            bind_context(
                correlation_id=correlation_id,
                request_id=request_id,
                path=request.url.path,
                method=request.method,
            )
        except ImportError:
            # structlog not configured, skip context binding
            pass
        
        try:
            response = await call_next(request)
            
            # Add correlation ID to response headers
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            response.headers[REQUEST_ID_HEADER] = request_id
            
            return response
        finally:
            # Clear logging context after request
            try:
                from aperion_cortex.service.logging import clear_context
                clear_context()
            except ImportError:
                pass


def get_correlation_id(request: Request) -> str | None:
    """Get the correlation ID from a request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Correlation ID if available, None otherwise
    """
    return getattr(request.state, "correlation_id", None)


def get_request_id(request: Request) -> str | None:
    """Get the request ID from a request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request ID if available, None otherwise
    """
    return getattr(request.state, "request_id", None)


__all__ = [
    "CorrelationIdMiddleware",
    "CORRELATION_ID_HEADER",
    "REQUEST_ID_HEADER",
    "get_correlation_id",
    "get_request_id",
]
