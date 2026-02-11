"""Thread pool executor for CPU-bound and blocking operations.

This module provides infrastructure for running blocking operations
in the async FastAPI context without blocking the event loop.

Usage:
    from aperion_cortex.service.executor import run_in_executor, async_wrap

    # Option 1: Direct call
    result = await run_in_executor(blocking_function, arg1, arg2)

    # Option 2: Decorator
    @async_wrap
    def cpu_bound_function(data):
        return heavy_computation(data)

    result = await cpu_bound_function(data)
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from typing import Any, Callable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Module-level executor storage
_EXECUTOR_SLOT: dict[str, ThreadPoolExecutor | None] = {"executor": None}

# Default configuration
DEFAULT_MAX_WORKERS = 4
DEFAULT_THREAD_PREFIX = "cortex-blocking-"


def get_executor(
    max_workers: int = DEFAULT_MAX_WORKERS,
    thread_name_prefix: str = DEFAULT_THREAD_PREFIX,
) -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor.

    Args:
        max_workers: Maximum number of worker threads
        thread_name_prefix: Prefix for thread names

    Returns:
        ThreadPoolExecutor instance
    """
    if _EXECUTOR_SLOT["executor"] is None:
        logger.info(
            f"Creating thread pool executor with {max_workers} workers"
        )
        _EXECUTOR_SLOT["executor"] = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
    return _EXECUTOR_SLOT["executor"]


def shutdown_executor(wait: bool = True) -> None:
    """Shutdown the executor gracefully.

    Args:
        wait: If True, wait for all pending tasks to complete
    """
    executor = _EXECUTOR_SLOT.get("executor")
    if executor:
        logger.info("Shutting down thread pool executor...")
        executor.shutdown(wait=wait)
        _EXECUTOR_SLOT["executor"] = None
        logger.info("Thread pool executor shutdown complete")


async def run_in_executor(
    func: Callable[..., R],
    *args: Any,
    **kwargs: Any,
) -> R:
    """Run a blocking function in the thread pool executor.

    Use this for:
    - CPU-bound operations (embedding encoding, hashing)
    - Sync HTTP clients (legacy code)
    - Database operations without async driver
    - File I/O operations

    Args:
        func: The blocking function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function

    Raises:
        RuntimeError: If called outside of an async context

    Example:
        # Run a blocking function
        result = await run_in_executor(sync_http_call, url, timeout=30)

        # Run with keyword arguments
        result = await run_in_executor(
            process_data,
            data,
            format="json",
            validate=True,
        )
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()

    # Wrap with partial to handle kwargs
    if kwargs:
        func_with_args = partial(func, *args, **kwargs)
    else:
        func_with_args = partial(func, *args) if args else func

    return await loop.run_in_executor(executor, func_with_args)


def async_wrap(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to wrap a sync function for async execution.

    The wrapped function will run in the thread pool executor,
    allowing it to be called with await without blocking the event loop.

    Args:
        func: The synchronous function to wrap

    Returns:
        An async version of the function

    Example:
        @async_wrap
        def cpu_bound_function(data: bytes) -> dict:
            return heavy_computation(data)

        # Now callable as:
        result = await cpu_bound_function(large_data)
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await run_in_executor(func, *args, **kwargs)

    return wrapper


class AsyncExecutorMixin:
    """Mixin class providing async execution helpers.

    Inherit from this to add async execution capabilities to a class.

    Example:
        class MyService(AsyncExecutorMixin):
            def _heavy_sync_operation(self, data):
                # CPU-bound work
                return result

            async def process(self, data):
                return await self._run_async(
                    self._heavy_sync_operation, data
                )
    """

    async def _run_async(
        self,
        func: Callable[..., R],
        *args: Any,
        **kwargs: Any,
    ) -> R:
        """Run a method in the thread pool executor."""
        return await run_in_executor(func, *args, **kwargs)


__all__ = [
    "get_executor",
    "shutdown_executor",
    "run_in_executor",
    "async_wrap",
    "AsyncExecutorMixin",
]
