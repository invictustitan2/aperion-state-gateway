"""Circuit breaker pattern for fault tolerance.

Implements the circuit breaker pattern to prevent cascading failures
when external services (like embedding APIs) are failing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    failure_threshold: int = 5     # Failures before opening
    success_threshold: int = 2     # Successes to close from half-open
    timeout_seconds: float = 30.0  # Time before trying again
    
    # Optional: exceptions that should trip the breaker
    trip_on: tuple[type[Exception], ...] = (Exception,)
    
    # Exceptions that should NOT trip the breaker (e.g., validation errors)
    ignore: tuple[type[Exception], ...] = (ValueError, TypeError)


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, name: str, time_remaining: float):
        self.name = name
        self.time_remaining = time_remaining
        super().__init__(
            f"Circuit breaker '{name}' is open. "
            f"Retry in {time_remaining:.1f} seconds."
        )


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation.
    
    Example:
        breaker = CircuitBreaker("openai-embeddings")
        
        try:
            result = breaker.call(lambda: openai_client.embed(text))
        except CircuitBreakerOpen:
            # Use fallback or return cached result
            result = fallback_embed(text)
    """
    
    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    
    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.config.timeout_seconds:
                logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state
    
    @property
    def time_until_retry(self) -> float:
        """Seconds until circuit breaker allows retry."""
        if self._state != CircuitState.OPEN:
            return 0.0
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self.config.timeout_seconds - elapsed)
    
    def call(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Any exception from the function (after recording failure)
        """
        state = self.state
        
        if state == CircuitState.OPEN:
            raise CircuitBreakerOpen(self.name, self.time_until_retry)
        
        try:
            result = func()
            self._on_success()
            return result
        except self.config.ignore:
            # Don't trip on ignored exceptions, just re-raise
            raise
        except self.config.trip_on as e:
            self._on_failure(e)
            raise
    
    async def call_async(self, func: Callable[[], Any]) -> Any:
        """Execute async function with circuit breaker protection."""
        state = self.state
        
        if state == CircuitState.OPEN:
            raise CircuitBreakerOpen(self.name, self.time_until_retry)
        
        try:
            result = await func()
            self._on_success()
            return result
        except self.config.ignore:
            raise
        except self.config.trip_on as e:
            self._on_failure(e)
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                logger.info(f"Circuit breaker '{self.name}' closing after recovery")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0
    
    def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        logger.warning(
            f"Circuit breaker '{self.name}' recorded failure "
            f"({self._failure_count}/{self.config.failure_threshold}): {exception}"
        )
        
        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            logger.warning(f"Circuit breaker '{self.name}' reopening after half-open failure")
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.config.failure_threshold:
            logger.warning(f"Circuit breaker '{self.name}' opening after threshold reached")
            self._state = CircuitState.OPEN
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "time_until_retry": self.time_until_retry,
        }


# Global circuit breakers for shared use
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a named circuit breaker.
    
    Args:
        name: Unique name for the circuit breaker
        config: Optional configuration (only used on creation)
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            config=config or CircuitBreakerConfig(),
        )
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    return _circuit_breakers.copy()


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitState",
    "get_circuit_breaker",
    "get_all_circuit_breakers",
]
