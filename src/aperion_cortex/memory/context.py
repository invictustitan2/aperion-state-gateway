"""Context Engine - Token packing and optimization for LLM consumption.

The Context Engine is the core of the "Warm Context" tier. It accepts
raw messages, files, and other content, then returns a token-optimized
ContextPack ready for the LLM's context window.

Key Features:
- Token counting using tiktoken
- Priority-based item selection
- Automatic trimming to fit token budget
- Support for multiple encoding models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Lazy import tiktoken for environments without it
if TYPE_CHECKING:
    import tiktoken


@dataclass
class ContextItem:
    """An item to include in context packing."""

    content: str
    priority: int = 0  # Higher = more important
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)
    _token_count: int | None = field(default=None, repr=False)


@dataclass
class PackingResult:
    """Result of context packing operation."""

    items: list[ContextItem]
    total_tokens: int
    items_included: int
    items_dropped: int
    budget_used_percent: float


def _get_encoder(model: str = "cl100k_base") -> "tiktoken.Encoding":
    """Get tiktoken encoder, with fallback for missing models."""
    import tiktoken

    try:
        return tiktoken.get_encoding(model)
    except KeyError:
        # Fall back to cl100k_base (GPT-4/ChatGPT encoding)
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: The text to count tokens for
        model: Tiktoken encoding model (default: cl100k_base for GPT-4)

    Returns:
        Number of tokens in the text
    """
    try:
        encoder = _get_encoder(model)
        return len(encoder.encode(text))
    except ImportError:
        # Fallback: rough estimate if tiktoken not available
        # Average English word is ~1.3 tokens
        logger.warning("tiktoken not available, using rough token estimate")
        return int(len(text.split()) * 1.3)


def optimize_context(
    items: list[ContextItem],
    max_tokens: int,
    reserve_tokens: int = 0,
    encoding_model: str = "cl100k_base",
) -> PackingResult:
    """Optimize a list of items to fit within token budget.

    Algorithm:
    1. Sort items by priority (descending)
    2. Count tokens for each item
    3. Greedily add items until budget exhausted
    4. Return included items preserving priority order

    Args:
        items: List of ContextItems to pack
        max_tokens: Maximum token budget
        reserve_tokens: Tokens to reserve (e.g., for system prompt)
        encoding_model: Tiktoken encoding model name

    Returns:
        PackingResult with selected items and statistics
    """
    if not items:
        return PackingResult(
            items=[],
            total_tokens=0,
            items_included=0,
            items_dropped=0,
            budget_used_percent=0.0,
        )

    available_tokens = max_tokens - reserve_tokens
    if available_tokens <= 0:
        return PackingResult(
            items=[],
            total_tokens=0,
            items_included=0,
            items_dropped=len(items),
            budget_used_percent=0.0,
        )

    # Count tokens for each item (cache for efficiency)
    for item in items:
        if item._token_count is None:
            item._token_count = count_tokens(item.content, encoding_model)

    # Sort by priority (highest first)
    sorted_items = sorted(items, key=lambda x: x.priority, reverse=True)

    # Greedy packing
    included: list[ContextItem] = []
    total_tokens = 0

    for item in sorted_items:
        item_tokens = item._token_count or 0
        if total_tokens + item_tokens <= available_tokens:
            included.append(item)
            total_tokens += item_tokens

    items_dropped = len(items) - len(included)
    budget_used = (total_tokens / available_tokens * 100) if available_tokens > 0 else 0

    logger.debug(
        f"Context packed: {len(included)}/{len(items)} items, "
        f"{total_tokens}/{available_tokens} tokens ({budget_used:.1f}%)"
    )

    return PackingResult(
        items=included,
        total_tokens=total_tokens,
        items_included=len(included),
        items_dropped=items_dropped,
        budget_used_percent=budget_used,
    )


class ContextEngine:
    """Context Engine for building LLM-ready context packs.

    The Context Engine manages the "Warm Context" tier, accepting raw
    content and producing token-optimized output for LLM consumption.

    Example:
        engine = ContextEngine(max_tokens=4096)
        engine.add("User message", priority=100, category="user")
        engine.add("System context", priority=50, category="system")
        result = engine.pack()
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_tokens: int = 0,
        encoding_model: str = "cl100k_base",
    ):
        """Initialize the Context Engine.

        Args:
            max_tokens: Maximum token budget for output
            reserve_tokens: Tokens to reserve (e.g., for response)
            encoding_model: Tiktoken encoding model name
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.encoding_model = encoding_model
        self._items: list[ContextItem] = []

    def add(
        self,
        content: str,
        priority: int = 0,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> "ContextEngine":
        """Add content to the context.

        Args:
            content: The text content to add
            priority: Priority (higher = more likely to be included)
            category: Category for grouping/filtering
            metadata: Additional metadata

        Returns:
            Self for chaining
        """
        self._items.append(
            ContextItem(
                content=content,
                priority=priority,
                category=category,
                metadata=metadata or {},
            )
        )
        return self

    def add_item(self, item: ContextItem) -> "ContextEngine":
        """Add a pre-built ContextItem."""
        self._items.append(item)
        return self

    def add_items(self, items: list[ContextItem]) -> "ContextEngine":
        """Add multiple items at once."""
        self._items.extend(items)
        return self

    def clear(self) -> "ContextEngine":
        """Clear all items."""
        self._items.clear()
        return self

    def pack(self) -> PackingResult:
        """Pack current items into optimized context.

        Returns:
            PackingResult with selected items
        """
        return optimize_context(
            items=self._items,
            max_tokens=self.max_tokens,
            reserve_tokens=self.reserve_tokens,
            encoding_model=self.encoding_model,
        )

    def get_packed_text(self, separator: str = "\n\n") -> str:
        """Get packed items as a single text string.

        Args:
            separator: Separator between items

        Returns:
            Concatenated text of packed items
        """
        result = self.pack()
        return separator.join(item.content for item in result.items)

    @property
    def item_count(self) -> int:
        """Number of items currently in the engine."""
        return len(self._items)

    def estimate_tokens(self) -> int:
        """Estimate total tokens for all current items."""
        total = 0
        for item in self._items:
            if item._token_count is None:
                item._token_count = count_tokens(item.content, self.encoding_model)
            total += item._token_count or 0
        return total


def create_context_pack_from_items(
    items: list[dict[str, Any]],
    max_tokens: int = 4096,
    encoding_model: str = "cl100k_base",
) -> PackingResult:
    """Convenience function to create a context pack from dictionaries.

    Args:
        items: List of dicts with 'content', optional 'priority', 'category'
        max_tokens: Maximum token budget
        encoding_model: Tiktoken encoding model

    Returns:
        PackingResult with optimized items
    """
    context_items = [
        ContextItem(
            content=item.get("content", ""),
            priority=item.get("priority", 0),
            category=item.get("category", "general"),
            metadata=item.get("metadata", {}),
        )
        for item in items
    ]
    return optimize_context(context_items, max_tokens, encoding_model=encoding_model)
