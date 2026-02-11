"""Unit tests for Context Engine token packing algorithms."""

import pytest

from aperion_cortex.memory.context import (
    ContextEngine,
    ContextItem,
    count_tokens,
    optimize_context,
)


class TestCountTokens:
    """Tests for token counting functionality."""

    def test_count_tokens_empty_string(self):
        """Empty string should return 0 tokens."""
        assert count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        """Simple text should return reasonable token count."""
        tokens = count_tokens("Hello, world!")
        # Should be around 4 tokens for this phrase
        assert 2 <= tokens <= 6

    def test_count_tokens_longer_text(self):
        """Longer text should have proportionally more tokens."""
        short = count_tokens("Hello")
        long = count_tokens("Hello " * 100)
        assert long > short * 50

    def test_count_tokens_special_characters(self):
        """Special characters should be handled."""
        tokens = count_tokens("Hello 👋 World 🌍!")
        assert tokens > 0


class TestOptimizeContext:
    """Tests for context optimization algorithm."""

    def test_optimize_empty_list(self):
        """Empty list should return empty result."""
        result = optimize_context([], max_tokens=1000)
        assert result.items == []
        assert result.total_tokens == 0
        assert result.items_included == 0
        assert result.items_dropped == 0

    def test_optimize_single_item_fits(self):
        """Single item that fits should be included."""
        items = [ContextItem(content="Hello world", priority=1)]
        result = optimize_context(items, max_tokens=1000)

        assert len(result.items) == 1
        assert result.items_included == 1
        assert result.items_dropped == 0

    def test_optimize_priority_ordering(self):
        """Higher priority items should be included first."""
        items = [
            ContextItem(content="Low priority", priority=1),
            ContextItem(content="High priority", priority=10),
            ContextItem(content="Medium priority", priority=5),
        ]
        result = optimize_context(items, max_tokens=1000)

        # All should fit
        assert result.items_included == 3

        # First item should be highest priority
        priorities = [item.priority for item in result.items]
        assert priorities[0] == 10

    def test_optimize_respects_token_budget(self):
        """Should not exceed token budget."""
        # Create items with known token counts
        items = [
            ContextItem(content="word " * 100, priority=10),  # ~100 tokens
            ContextItem(content="word " * 100, priority=5),   # ~100 tokens
            ContextItem(content="word " * 100, priority=1),   # ~100 tokens
        ]
        result = optimize_context(items, max_tokens=150)

        # Only highest priority should fit
        assert result.items_included == 1
        assert result.items_dropped == 2
        assert result.items[0].priority == 10

    def test_optimize_with_reserve_tokens(self):
        """Reserve tokens should reduce available budget."""
        items = [ContextItem(content="word " * 50, priority=1)]
        
        # With no reserve, should fit
        result1 = optimize_context(items, max_tokens=100, reserve_tokens=0)
        
        # With large reserve, might not fit
        result2 = optimize_context(items, max_tokens=100, reserve_tokens=90)

        assert result1.items_included >= result2.items_included


class TestContextEngine:
    """Tests for ContextEngine class."""

    def test_engine_creation(self):
        """Engine should be created with defaults."""
        engine = ContextEngine()
        assert engine.max_tokens == 4096
        assert engine.item_count == 0

    def test_engine_add_content(self):
        """Adding content should increase item count."""
        engine = ContextEngine()
        engine.add("Hello world", priority=1)
        engine.add("Another item", priority=2)

        assert engine.item_count == 2

    def test_engine_chaining(self):
        """Add methods should return self for chaining."""
        engine = ContextEngine()
        result = engine.add("First").add("Second").add("Third")

        assert result is engine
        assert engine.item_count == 3

    def test_engine_pack(self):
        """Pack should return PackingResult."""
        engine = ContextEngine(max_tokens=1000)
        engine.add("Hello", priority=10)
        engine.add("World", priority=5)

        result = engine.pack()

        assert result.items_included == 2
        assert result.total_tokens > 0

    def test_engine_get_packed_text(self):
        """get_packed_text should return concatenated content."""
        engine = ContextEngine(max_tokens=1000)
        engine.add("Hello", priority=10)
        engine.add("World", priority=5)

        text = engine.get_packed_text(separator=" ")

        assert "Hello" in text
        assert "World" in text

    def test_engine_clear(self):
        """Clear should remove all items."""
        engine = ContextEngine()
        engine.add("Item 1").add("Item 2")
        engine.clear()

        assert engine.item_count == 0

    def test_engine_estimate_tokens(self):
        """estimate_tokens should return total for all items."""
        engine = ContextEngine()
        engine.add("Hello world")
        engine.add("Another sentence here")

        estimate = engine.estimate_tokens()

        assert estimate > 0

    def test_engine_category_tracking(self):
        """Items should track their categories."""
        engine = ContextEngine(max_tokens=1000)
        engine.add("User message", priority=10, category="user")
        engine.add("System context", priority=5, category="system")

        result = engine.pack()

        categories = [item.category for item in result.items]
        assert "user" in categories
        assert "system" in categories


class TestContextItemModel:
    """Tests for ContextItem dataclass."""

    def test_item_creation_defaults(self):
        """Item should have sensible defaults."""
        item = ContextItem(content="Test content")

        assert item.content == "Test content"
        assert item.priority == 0
        assert item.category == "general"
        assert item.metadata == {}

    def test_item_with_metadata(self):
        """Item should store metadata."""
        item = ContextItem(
            content="Test",
            priority=5,
            metadata={"source": "user", "id": 123},
        )

        assert item.metadata["source"] == "user"
        assert item.metadata["id"] == 123
