"""Prometheus metrics for the Cortex service.

Exports key metrics for monitoring:
- Request latency histograms
- Request counts by endpoint and status
- Vector search performance
- Context optimization stats
- Active connections
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class MetricsCollector:
    """Collects and exposes Prometheus-style metrics.
    
    This is a lightweight metrics implementation that doesn't require
    the prometheus_client library. Metrics are exposed via /metrics endpoint.
    """
    
    # Counters
    request_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    request_errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Histograms (store raw values for percentile calculation)
    request_latency: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    vector_search_latency: list[float] = field(default_factory=list)
    embedding_latency: list[float] = field(default_factory=list)
    
    # Gauges
    active_requests: int = 0
    vector_store_docs: int = 0
    hot_state_entries: int = 0
    
    # Max samples to keep for histograms
    max_samples: int = 1000

    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record a completed request."""
        key = f"{method}:{path}"
        self.request_count[key] += 1
        
        if status_code >= 400:
            error_key = f"{key}:{status_code}"
            self.request_errors[error_key] += 1
        
        # Trim histogram if too large
        if len(self.request_latency[key]) >= self.max_samples:
            self.request_latency[key] = self.request_latency[key][-self.max_samples // 2:]
        self.request_latency[key].append(duration_seconds)

    def record_vector_search(self, duration_seconds: float) -> None:
        """Record a vector search operation."""
        if len(self.vector_search_latency) >= self.max_samples:
            self.vector_search_latency = self.vector_search_latency[-self.max_samples // 2:]
        self.vector_search_latency.append(duration_seconds)

    def record_embedding(self, duration_seconds: float) -> None:
        """Record an embedding operation."""
        if len(self.embedding_latency) >= self.max_samples:
            self.embedding_latency = self.embedding_latency[-self.max_samples // 2:]
        self.embedding_latency.append(duration_seconds)

    def _percentile(self, values: list[float], p: float) -> float:
        """Calculate percentile from list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        # Request counts
        lines.append("# HELP cortex_requests_total Total number of HTTP requests")
        lines.append("# TYPE cortex_requests_total counter")
        for key, count in self.request_count.items():
            method, path = key.split(":", 1)
            lines.append(f'cortex_requests_total{{method="{method}",path="{path}"}} {count}')
        
        # Request errors
        lines.append("# HELP cortex_request_errors_total Total number of HTTP errors")
        lines.append("# TYPE cortex_request_errors_total counter")
        for key, count in self.request_errors.items():
            parts = key.rsplit(":", 2)
            method, path, status = parts[0], parts[1], parts[2]
            lines.append(f'cortex_request_errors_total{{method="{method}",path="{path}",status="{status}"}} {count}')
        
        # Request latency percentiles
        lines.append("# HELP cortex_request_duration_seconds Request latency percentiles")
        lines.append("# TYPE cortex_request_duration_seconds summary")
        for key, values in self.request_latency.items():
            method, path = key.split(":", 1)
            for quantile in [0.5, 0.9, 0.99]:
                p_value = self._percentile(values, quantile)
                lines.append(f'cortex_request_duration_seconds{{method="{method}",path="{path}",quantile="{quantile}"}} {p_value:.6f}')
        
        # Vector search latency
        if self.vector_search_latency:
            lines.append("# HELP cortex_vector_search_seconds Vector search latency")
            lines.append("# TYPE cortex_vector_search_seconds summary")
            for quantile in [0.5, 0.9, 0.99]:
                p_value = self._percentile(self.vector_search_latency, quantile)
                lines.append(f'cortex_vector_search_seconds{{quantile="{quantile}"}} {p_value:.6f}')
        
        # Embedding latency
        if self.embedding_latency:
            lines.append("# HELP cortex_embedding_seconds Embedding generation latency")
            lines.append("# TYPE cortex_embedding_seconds summary")
            for quantile in [0.5, 0.9, 0.99]:
                p_value = self._percentile(self.embedding_latency, quantile)
                lines.append(f'cortex_embedding_seconds{{quantile="{quantile}"}} {p_value:.6f}')
        
        # Gauges
        lines.append("# HELP cortex_active_requests Current number of in-flight requests")
        lines.append("# TYPE cortex_active_requests gauge")
        lines.append(f"cortex_active_requests {self.active_requests}")
        
        lines.append("# HELP cortex_vector_store_documents Number of documents in vector store")
        lines.append("# TYPE cortex_vector_store_documents gauge")
        lines.append(f"cortex_vector_store_documents {self.vector_store_docs}")
        
        lines.append("# HELP cortex_hot_state_entries Number of entries in hot state cache")
        lines.append("# TYPE cortex_hot_state_entries gauge")
        lines.append(f"cortex_hot_state_entries {self.hot_state_entries}")
        
        return "\n".join(lines) + "\n"


# Global metrics collector instance
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        metrics = get_metrics()
        metrics.active_requests += 1
        
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            duration = time.perf_counter() - start_time
            
            # Skip metrics endpoint itself
            if request.url.path != "/metrics":
                metrics.record_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_seconds=duration,
                )
            
            return response
        finally:
            metrics.active_requests -= 1


def add_metrics_endpoint(app: FastAPI) -> None:
    """Add /metrics endpoint to FastAPI app."""
    
    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:
        """Prometheus metrics endpoint."""
        metrics = get_metrics()
        return Response(
            content=metrics.export_prometheus(),
            media_type="text/plain; charset=utf-8",
        )


__all__ = [
    "MetricsCollector",
    "MetricsMiddleware", 
    "get_metrics",
    "add_metrics_endpoint",
]
