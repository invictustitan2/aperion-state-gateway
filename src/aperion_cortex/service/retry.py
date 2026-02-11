"""Retry utilities with exponential backoff.

Provides decorators and utilities for retrying failed operations
with configurable backoff strategies.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    
    # Exceptions to retry on (default: all exceptions)
    retry_on: tuple[type[Exception], ...] = (Exception,)
    
    # Exceptions to NOT retry on
    no_retry_on: tuple[type[Exception], ...] = (ValueError, TypeError, KeyError)


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """Calculate backoff delay for a given attempt.
    
    Uses exponential backoff with optional jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential growth
        jitter: Whether to add random jitter
        
    Returns:
        Delay in seconds
    """
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)
    
    if jitter:
        # Add up to 25% jitter
        jitter_range = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)


def retry_sync(
    func: Callable[..., T] | None = None,
    *,
    config: RetryConfig | None = None,
    max_attempts: int | None = None,
    base_delay: float | None = None,
) -> Callable[..., T]:
    """Decorator for synchronous retry with exponential backoff.
    
    Example:
        @retry_sync(max_attempts=3, base_delay=1.0)
        def call_api():
            return requests.get("https://api.example.com")
    """
    cfg = config or RetryConfig()
    if max_attempts is not None:
        cfg.max_attempts = max_attempts
    if base_delay is not None:
        cfg.base_delay = base_delay
    
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            
            for attempt in range(cfg.max_attempts):
                try:
                    return fn(*args, **kwargs)
                except cfg.no_retry_on:
                    # Don't retry these
                    raise
                except cfg.retry_on as e:
                    last_exception = e
                    
                    if attempt < cfg.max_attempts - 1:
                        delay = calculate_backoff(
                            attempt,
                            cfg.base_delay,
                            cfg.max_delay,
                            cfg.exponential_base,
                            cfg.jitter,
                        )
                        logger.warning(
                            f"Retry {attempt + 1}/{cfg.max_attempts} for {fn.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
            
            # All attempts exhausted
            logger.error(f"All {cfg.max_attempts} attempts failed for {fn.__name__}")
            raise last_exception  # type: ignore
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


def retry_async(
    func: Callable[..., T] | None = None,
    *,
    config: RetryConfig | None = None,
    max_attempts: int | None = None,
    base_delay: float | None = None,
) -> Callable[..., T]:
    """Decorator for async retry with exponential backoff.
    
    Example:
        @retry_async(max_attempts=3, base_delay=1.0)
        async def call_api():
            async with httpx.AsyncClient() as client:
                return await client.get("https://api.example.com")
    """
    cfg = config or RetryConfig()
    if max_attempts is not None:
        cfg.max_attempts = max_attempts
    if base_delay is not None:
        cfg.base_delay = base_delay
    
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            
            for attempt in range(cfg.max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except cfg.no_retry_on:
                    raise
                except cfg.retry_on as e:
                    last_exception = e
                    
                    if attempt < cfg.max_attempts - 1:
                        delay = calculate_backoff(
                            attempt,
                            cfg.base_delay,
                            cfg.max_delay,
                            cfg.exponential_base,
                            cfg.jitter,
                        )
                        logger.warning(
                            f"Retry {attempt + 1}/{cfg.max_attempts} for {fn.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(delay)
            
            logger.error(f"All {cfg.max_attempts} attempts failed for {fn.__name__}")
            raise last_exception  # type: ignore
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


class RetryContext:
    """Context manager for retry operations.
    
    Example:
        async with RetryContext(max_attempts=3) as ctx:
            for attempt in ctx:
                try:
                    result = await risky_operation()
                    break
                except TransientError:
                    await ctx.backoff()
    """
    
    def __init__(self, config: RetryConfig | None = None, **kwargs):
        self.config = config or RetryConfig(**kwargs)
        self.attempt = 0
        self.last_exception: Exception | None = None
    
    def __iter__(self):
        for self.attempt in range(self.config.max_attempts):
            yield self.attempt
    
    def backoff_delay(self) -> float:
        """Get current backoff delay."""
        return calculate_backoff(
            self.attempt,
            self.config.base_delay,
            self.config.max_delay,
            self.config.exponential_base,
            self.config.jitter,
        )
    
    def backoff_sync(self) -> None:
        """Sleep for backoff period (sync)."""
        delay = self.backoff_delay()
        logger.debug(f"Backing off for {delay:.2f}s (attempt {self.attempt + 1})")
        time.sleep(delay)
    
    async def backoff(self) -> None:
        """Sleep for backoff period (async)."""
        delay = self.backoff_delay()
        logger.debug(f"Backing off for {delay:.2f}s (attempt {self.attempt + 1})")
        await asyncio.sleep(delay)


__all__ = [
    "RetryConfig",
    "RetryContext",
    "calculate_backoff",
    "retry_async",
    "retry_sync",
]
