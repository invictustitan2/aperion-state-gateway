"""SSE Event Bridge: Relays Cortex events to Aperion EventBus.

This bridge subscribes to Cortex's SSE stream and emits events to the
Aperion EventBus, enabling real-time state synchronization between
the Cortex memory service and Aperion agents.

Example:
    >>> from aperion_cortex_client.integration import CortexEventBridge
    >>> from stack.aperion.foundation.event_bus import EventBus
    >>> 
    >>> bus = EventBus()
    >>> bridge = CortexEventBridge(
    ...     cortex_url="http://localhost:4949",
    ...     token="...",
    ...     event_bus=bus,
    ... )
    >>> await bridge.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol for EventBus compatibility
# ---------------------------------------------------------------------------


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol matching Aperion EventBus.emit() signature."""
    
    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str | None = None,
        correlation_id: str | None = None,
        wait_for_handlers: bool = False,
    ) -> str:
        """Emit an event and return event ID."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EventBridgeConfig:
    """Configuration for the Cortex event bridge."""
    
    cortex_url: str
    token: str | None = None
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    reconnect_backoff: float = 2.0
    timeout: float = 30.0
    source_name: str = "cortex"
    event_type_prefix: str = "cortex."


# ---------------------------------------------------------------------------
# Event Type Mapping
# ---------------------------------------------------------------------------


# Map Cortex event types to Aperion EventBus topics
EVENT_TYPE_MAP: dict[str, str] = {
    # State events
    "state.updated": "cortex.state.updated",
    "state.snapshot": "cortex.state.snapshot",
    "state.invalidated": "cortex.state.invalidated",
    
    # Vector store events
    "vector.indexed": "cortex.vector.indexed",
    "vector.deleted": "cortex.vector.deleted",
    "vector.searched": "cortex.vector.searched",
    
    # Audit events
    "audit.recorded": "cortex.audit.recorded",
    "audit.rotated": "cortex.audit.rotated",
    
    # Health events
    "health.degraded": "cortex.health.degraded",
    "health.recovered": "cortex.health.recovered",
    
    # Context events
    "context.packed": "cortex.context.packed",
    "context.overflow": "cortex.context.overflow",
}


def map_event_type(cortex_type: str, prefix: str = "cortex.") -> str:
    """Map Cortex event type to Aperion EventBus topic.
    
    Uses explicit mapping if available, otherwise prefixes with 'cortex.'.
    """
    if cortex_type in EVENT_TYPE_MAP:
        return EVENT_TYPE_MAP[cortex_type]
    return f"{prefix}{cortex_type}"


# ---------------------------------------------------------------------------
# Event Bridge
# ---------------------------------------------------------------------------


class CortexEventBridge:
    """Bridges Cortex SSE events to Aperion EventBus.
    
    Features:
    - Automatic reconnection with exponential backoff
    - Event type mapping (Cortex → Aperion)
    - Correlation ID propagation
    - Graceful shutdown
    
    Example:
        >>> bridge = CortexEventBridge(
        ...     cortex_url="http://localhost:4949",
        ...     token="...",
        ...     event_bus=my_event_bus,
        ... )
        >>> await bridge.start()
        >>> # ... later
        >>> await bridge.stop()
    """
    
    def __init__(
        self,
        cortex_url: str,
        event_bus: EventEmitter,
        *,
        token: str | None = None,
        config: EventBridgeConfig | None = None,
    ):
        self._config = config or EventBridgeConfig(
            cortex_url=cortex_url.rstrip("/"),
            token=token,
        )
        if not config:
            self._config.cortex_url = cortex_url.rstrip("/")
            self._config.token = token
            
        self._event_bus = event_bus
        self._running = False
        self._task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None
        self._reconnect_delay = self._config.reconnect_delay
        
        # Stats
        self._events_received = 0
        self._events_emitted = 0
        self._reconnect_count = 0
        self._last_event_at: datetime | None = None
        self._connected = False
    
    @property
    def is_running(self) -> bool:
        """Check if bridge is running."""
        return self._running
    
    @property
    def is_connected(self) -> bool:
        """Check if SSE stream is connected."""
        return self._connected
    
    @property
    def stats(self) -> dict[str, Any]:
        """Get bridge statistics."""
        return {
            "running": self._running,
            "connected": self._connected,
            "events_received": self._events_received,
            "events_emitted": self._events_emitted,
            "reconnect_count": self._reconnect_count,
            "last_event_at": self._last_event_at.isoformat() if self._last_event_at else None,
        }
    
    async def start(self) -> None:
        """Start the event bridge.
        
        Connects to Cortex SSE stream and begins relaying events.
        Automatically reconnects on disconnection.
        """
        if self._running:
            logger.warning("Event bridge already running")
            return
        
        self._running = True
        self._client = httpx.AsyncClient(timeout=None)  # No timeout for SSE
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Cortex event bridge started",
            extra={"cortex_url": self._config.cortex_url},
        )
    
    async def stop(self) -> None:
        """Stop the event bridge.
        
        Gracefully disconnects from SSE stream and cleans up resources.
        """
        if not self._running:
            return
        
        self._running = False
        self._connected = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        if self._client:
            await self._client.aclose()
            self._client = None
        
        logger.info(
            "Cortex event bridge stopped",
            extra={"stats": self.stats},
        )
    
    async def _run_loop(self) -> None:
        """Main event loop with reconnection logic."""
        while self._running:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                
                logger.warning(
                    "SSE connection failed, reconnecting...",
                    extra={
                        "error": str(e),
                        "delay": self._reconnect_delay,
                        "reconnect_count": self._reconnect_count,
                    },
                )
                
                # Exponential backoff
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * self._config.reconnect_backoff,
                    self._config.max_reconnect_delay,
                )
    
    async def _connect_and_stream(self) -> None:
        """Connect to SSE stream and process events."""
        headers = {}
        if self._config.token:
            headers["Authorization"] = f"Bearer {self._config.token}"
        
        url = f"{self._config.cortex_url}/events/stream"
        
        async with self._client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            self._connected = True
            self._reconnect_delay = self._config.reconnect_delay  # Reset backoff
            
            logger.info("Connected to Cortex SSE stream")
            
            async for line in response.aiter_lines():
                if not self._running:
                    break
                
                await self._process_sse_line(line)
    
    async def _process_sse_line(self, line: str) -> None:
        """Process a single SSE line."""
        if not line or line.startswith(":"):
            # Empty line or comment
            return
        
        if line.startswith("data:"):
            data = line[5:].strip()
            if data:
                await self._handle_event_data(data)
    
    async def _handle_event_data(self, data: str) -> None:
        """Parse and emit event to EventBus."""
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in SSE data", extra={"data": data[:100]})
            return
        
        self._events_received += 1
        self._last_event_at = datetime.now(timezone.utc)
        
        # Extract event fields
        cortex_type = event.get("event_type", "unknown")
        payload = event.get("payload", {})
        correlation_id = event.get("correlation_id")
        resource_ids = event.get("resource_ids", [])
        
        # Map to Aperion event type
        aperion_type = map_event_type(cortex_type, self._config.event_type_prefix)
        
        # Enrich payload
        enriched_payload = {
            **payload,
            "cortex_event_type": cortex_type,
            "resource_ids": resource_ids,
            "bridge_timestamp": self._last_event_at.isoformat(),
        }
        
        # Emit to EventBus
        try:
            self._event_bus.emit(
                event_type=aperion_type,
                payload=enriched_payload,
                source=self._config.source_name,
                correlation_id=correlation_id,
            )
            self._events_emitted += 1
            
            logger.debug(
                "Event relayed",
                extra={
                    "cortex_type": cortex_type,
                    "aperion_type": aperion_type,
                    "correlation_id": correlation_id,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to emit event to EventBus",
                extra={"error": str(e), "event_type": aperion_type},
            )


# ---------------------------------------------------------------------------
# Async Context Manager Support
# ---------------------------------------------------------------------------


class CortexEventBridgeContext:
    """Async context manager for CortexEventBridge.
    
    Example:
        >>> async with CortexEventBridgeContext(
        ...     cortex_url="http://localhost:4949",
        ...     event_bus=my_bus,
        ... ) as bridge:
        ...     # Bridge is running
        ...     await asyncio.sleep(60)
        >>> # Bridge is stopped
    """
    
    def __init__(
        self,
        cortex_url: str,
        event_bus: EventEmitter,
        **kwargs,
    ):
        self._bridge = CortexEventBridge(
            cortex_url=cortex_url,
            event_bus=event_bus,
            **kwargs,
        )
    
    async def __aenter__(self) -> CortexEventBridge:
        await self._bridge.start()
        return self._bridge
    
    async def __aexit__(self, *args) -> None:
        await self._bridge.stop()
