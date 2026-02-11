"""FastAPI application factory for the Cortex service."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import ScopedTokenManager, set_token_manager
from .config import CortexConfig
from .core import CortexService
from .executor import get_executor, shutdown_executor
from .router import build_router

logger = logging.getLogger(__name__)

# Graceful shutdown state
_shutdown_event: asyncio.Event | None = None
_shutdown_timeout = 30  # seconds to wait for in-flight requests


def _handle_shutdown_signal(signum: int, frame) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, initiating graceful shutdown...")
    if _shutdown_event:
        _shutdown_event.set()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for startup/shutdown."""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _handle_shutdown_signal(s, None))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, _handle_shutdown_signal)

    # Startup - initialize thread pool for blocking operations
    logger.info("Starting Cortex service...")
    get_executor()  # Pre-create executor
    
    yield
    
    # Shutdown - cleanup resources
    logger.info("Shutting down Cortex service...")
    
    # Close service resources
    service: CortexService | None = getattr(app.state, "cortex_service", None)
    if service and service._repository:
        service._repository.close()
    
    # Shutdown thread pool executor (waits for pending tasks)
    shutdown_executor(wait=True)
    
    logger.info("Cortex service shutdown complete")


def create_cortex_app(
    config: CortexConfig,
    **service_kwargs,
) -> FastAPI:
    """Create and configure the Cortex FastAPI application.

    Args:
        config: CortexConfig instance
        **service_kwargs: Additional kwargs passed to CortexService

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Aperion Cortex",
        description="Unified memory and state management service",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware (protects against resource exhaustion)
    from .rate_limit import RateLimitMiddleware, RateLimitConfig
    app.add_middleware(
        RateLimitMiddleware,
        config=RateLimitConfig(
            requests_per_minute=120,
            requests_per_second=20,
            burst_size=30,
        ),
    )

    # Correlation ID middleware (adds X-Correlation-ID to all requests)
    from .middleware import CorrelationIdMiddleware
    app.add_middleware(CorrelationIdMiddleware)

    # Metrics middleware (collects request timing)
    from .metrics import MetricsMiddleware, add_metrics_endpoint
    app.add_middleware(MetricsMiddleware)
    add_metrics_endpoint(app)

    # Token manager
    token_manager = ScopedTokenManager(config)
    set_token_manager(token_manager)

    # Create service
    cortex_service = CortexService(config, **service_kwargs)

    # Build and include router
    router = build_router(cortex_service)
    app.include_router(router)

    # Store references in app state
    app.state.cortex_service = cortex_service
    app.state.token_manager = token_manager
    app.state.config = config
    app.state.ready = False  # Track readiness state

    # Health endpoint (no auth required) - checks actual dependencies
    @app.get("/healthz")
    def healthz() -> dict:
        """Health check endpoint with dependency verification."""
        checks = {}
        all_healthy = True

        # Check database connection
        try:
            if cortex_service._repository:
                # Simple query to verify connection
                cortex_service._repository._get_conn().execute("SELECT 1")
                checks["database"] = {"status": "healthy"}
            else:
                checks["database"] = {"status": "not_configured"}
        except Exception as e:
            checks["database"] = {"status": "unhealthy", "error": str(e)}
            all_healthy = False

        # Check vector store
        try:
            if cortex_service._vector_store:
                # Get count to verify it's working
                count = cortex_service._vector_store.count()
                checks["vector_store"] = {"status": "healthy", "doc_count": count}
            else:
                checks["vector_store"] = {"status": "not_configured"}
        except Exception as e:
            checks["vector_store"] = {"status": "unhealthy", "error": str(e)}
            all_healthy = False

        # Check audit ledger
        try:
            if cortex_service._ledger:
                checks["audit_ledger"] = {"status": "healthy"}
            else:
                checks["audit_ledger"] = {"status": "not_configured"}
        except Exception as e:
            checks["audit_ledger"] = {"status": "unhealthy", "error": str(e)}
            all_healthy = False

        return {
            "status": "ok" if all_healthy else "degraded",
            "service": "cortex",
            "version": "0.1.0",
            "checks": checks,
        }

    # Ready endpoint (no auth required) - for K8s readiness probe
    @app.get("/ready")
    def ready() -> dict:
        """Readiness probe - returns true when service can accept traffic."""
        # Check if service is fully initialized
        is_ready = (
            cortex_service is not None
            and hasattr(app.state, "ready")
        )
        
        # Also check critical dependencies are accessible
        if is_ready:
            try:
                # Quick sanity check on database
                if cortex_service._repository:
                    cortex_service._repository._get_conn().execute("SELECT 1")
            except Exception:
                is_ready = False

        return {
            "ready": is_ready,
            "service": "cortex",
        }

    # Mark as ready after all initialization
    app.state.ready = True

    return app


# Convenience: create app with config from environment
def create_app_from_env() -> FastAPI:
    """Create app using environment variable configuration."""
    config = CortexConfig.from_env()
    return create_cortex_app(config)


__all__ = ["create_cortex_app", "create_app_from_env"]
