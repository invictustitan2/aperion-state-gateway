"""Cortex Client SDK for Aperion State Gateway.

Provides async and sync interfaces for interacting with the Cortex
memory and state management service.

Example:
    >>> from aperion_cortex_client import CortexClient
    >>> client = CortexClient("http://localhost:4949", token="...")
    >>> pack = await client.get_snapshot()
    >>> hits = await client.vector_search("query text")

Integration:
    >>> from aperion_cortex_client.integration import CortexEventBridge
    >>> bridge = CortexEventBridge(cortex_url, event_bus=my_bus)
    >>> await bridge.start()
"""

from .client import (
    CortexClient,
    CortexClientSync,
    CortexConfig,
    CortexError,
    ContextPack,
    Resource,
    VectorHit,
)

__all__ = [
    "CortexClient",
    "CortexClientSync",
    "CortexConfig",
    "CortexError",
    "ContextPack",
    "Resource",
    "VectorHit",
]
__version__ = "0.1.0"
