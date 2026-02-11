"""Tests for Cortex Event Bridge."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from aperion_cortex_client.integration.event_bridge import (
    CortexEventBridge,
    EventBridgeConfig,
    map_event_type,
    EVENT_TYPE_MAP,
)


# ---------------------------------------------------------------------------
# Mock EventBus
# ---------------------------------------------------------------------------


class MockEventBus:
    """Mock EventBus for testing."""
    
    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.emit_count = 0
    
    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str | None = None,
        correlation_id: str | None = None,
        wait_for_handlers: bool = False,
    ) -> str:
        """Record emitted event."""
        self.emit_count += 1
        event_id = f"evt-{self.emit_count}"
        self.events.append({
            "id": event_id,
            "event_type": event_type,
            "payload": payload,
            "source": source,
            "correlation_id": correlation_id,
        })
        return event_id


# ---------------------------------------------------------------------------
# Event Type Mapping Tests
# ---------------------------------------------------------------------------


class TestEventTypeMapping:
    """Tests for event type mapping."""

    def test_known_event_types_mapped(self):
        """Known event types use explicit mapping."""
        assert map_event_type("state.updated") == "cortex.state.updated"
        assert map_event_type("vector.indexed") == "cortex.vector.indexed"
        assert map_event_type("audit.recorded") == "cortex.audit.recorded"

    def test_unknown_event_types_prefixed(self):
        """Unknown event types get prefix added."""
        assert map_event_type("custom.event") == "cortex.custom.event"
        assert map_event_type("foo") == "cortex.foo"

    def test_custom_prefix(self):
        """Custom prefix is applied to unknown types."""
        assert map_event_type("custom.event", prefix="memory.") == "memory.custom.event"

    def test_all_mapped_types_have_prefix(self):
        """All mapped event types have cortex. prefix."""
        for aperion_type in EVENT_TYPE_MAP.values():
            assert aperion_type.startswith("cortex.")


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestEventBridgeConfig:
    """Tests for EventBridgeConfig."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = EventBridgeConfig(cortex_url="http://localhost:4949")
        assert config.reconnect_delay == 1.0
        assert config.max_reconnect_delay == 60.0
        assert config.reconnect_backoff == 2.0
        assert config.source_name == "cortex"
        assert config.event_type_prefix == "cortex."
        assert config.token is None

    def test_custom_values(self):
        """Config accepts custom values."""
        config = EventBridgeConfig(
            cortex_url="http://localhost:4949",
            token="secret",
            reconnect_delay=2.0,
            source_name="memory",
        )
        assert config.token == "secret"
        assert config.reconnect_delay == 2.0
        assert config.source_name == "memory"


# ---------------------------------------------------------------------------
# Event Bridge Tests
# ---------------------------------------------------------------------------


class TestCortexEventBridge:
    """Tests for CortexEventBridge."""

    def test_initial_state(self):
        """Bridge starts in stopped state."""
        bus = MockEventBus()
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=bus,
        )
        assert not bridge.is_running
        assert not bridge.is_connected
        assert bridge.stats["events_received"] == 0
        assert bridge.stats["events_emitted"] == 0

    def test_stats_structure(self):
        """Stats returns expected structure."""
        bus = MockEventBus()
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=bus,
        )
        stats = bridge.stats
        assert "running" in stats
        assert "connected" in stats
        assert "events_received" in stats
        assert "events_emitted" in stats
        assert "reconnect_count" in stats
        assert "last_event_at" in stats

    def test_config_passed_through(self):
        """Custom config is used."""
        bus = MockEventBus()
        config = EventBridgeConfig(
            cortex_url="http://custom:9999",
            token="test-token",
            source_name="test-source",
        )
        bridge = CortexEventBridge(
            cortex_url="http://ignored",  # Should be overridden by config
            event_bus=bus,
            config=config,
        )
        assert bridge._config.cortex_url == "http://custom:9999"
        assert bridge._config.token == "test-token"

    @pytest.mark.asyncio
    async def test_handle_event_data_emits_to_bus(self):
        """Event data is parsed and emitted to EventBus."""
        bus = MockEventBus()
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=bus,
        )
        
        # Simulate receiving event data
        event_data = json.dumps({
            "event_type": "state.updated",
            "payload": {"key": "value"},
            "correlation_id": "corr-123",
            "resource_ids": ["r1", "r2"],
        })
        
        await bridge._handle_event_data(event_data)
        
        assert bridge._events_received == 1
        assert bridge._events_emitted == 1
        assert len(bus.events) == 1
        
        emitted = bus.events[0]
        assert emitted["event_type"] == "cortex.state.updated"
        assert emitted["source"] == "cortex"
        assert emitted["correlation_id"] == "corr-123"
        assert emitted["payload"]["key"] == "value"
        assert emitted["payload"]["cortex_event_type"] == "state.updated"
        assert emitted["payload"]["resource_ids"] == ["r1", "r2"]

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self):
        """Invalid JSON is logged but doesn't crash."""
        bus = MockEventBus()
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=bus,
        )
        
        await bridge._handle_event_data("not valid json")
        
        assert bridge._events_received == 0
        assert len(bus.events) == 0

    @pytest.mark.asyncio
    async def test_process_sse_line_data(self):
        """SSE data: lines are processed."""
        bus = MockEventBus()
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=bus,
        )
        
        event_data = json.dumps({
            "event_type": "vector.indexed",
            "payload": {"doc_id": "doc-1"},
        })
        
        await bridge._process_sse_line(f"data:{event_data}")
        
        assert bridge._events_received == 1
        assert bus.events[0]["event_type"] == "cortex.vector.indexed"

    @pytest.mark.asyncio
    async def test_process_sse_line_ignores_comments(self):
        """SSE comment lines are ignored."""
        bus = MockEventBus()
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=bus,
        )
        
        await bridge._process_sse_line(": this is a comment")
        await bridge._process_sse_line("")
        
        assert bridge._events_received == 0

    @pytest.mark.asyncio
    async def test_last_event_timestamp_updated(self):
        """Last event timestamp is updated on each event."""
        bus = MockEventBus()
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=bus,
        )
        
        assert bridge._last_event_at is None
        
        await bridge._handle_event_data('{"event_type": "test", "payload": {}}')
        
        assert bridge._last_event_at is not None
        assert isinstance(bridge._last_event_at, datetime)

    @pytest.mark.asyncio
    async def test_bus_emit_failure_counted(self):
        """EventBus emit failures don't crash bridge."""
        class FailingBus:
            def emit(self, *args, **kwargs):
                raise RuntimeError("Bus broken")
        
        bridge = CortexEventBridge(
            cortex_url="http://localhost:4949",
            event_bus=FailingBus(),
        )
        
        # Should not raise
        await bridge._handle_event_data('{"event_type": "test", "payload": {}}')
        
        assert bridge._events_received == 1
        assert bridge._events_emitted == 0  # Failed to emit
