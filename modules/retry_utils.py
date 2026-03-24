"""
retry_utils.py — Retry Logic with Exponential Backoff
======================================================
Provides robust retry mechanisms for API calls.
"""

import asyncio
import logging
import random
from functools import wraps
from typing import Callable, TypeVar, Any

log = logging.getLogger("retry")

T = TypeVar('T')

# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetryExhausted(Exception):
    """All retry attempts failed."""
    def __init__(self, last_error: Exception, attempts: int):
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(f"Failed after {attempts} attempts: {last_error}")


async def retry_async(
    func: Callable,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
    **kwargs,
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to call
        max_attempts: Maximum number of attempts (default 3)
        base_delay: Initial delay in seconds (default 1.0)
        max_delay: Maximum delay cap in seconds (default 30.0)
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exceptions to retry on
    
    Returns:
        Result of successful function call
    
    Raises:
        RetryExhausted: If all attempts fail
    """
    last_error = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
            
        except retryable_exceptions as e:
            last_error = e
            
            if attempt == max_attempts:
                log.warning(f"Retry exhausted after {attempt} attempts: {e}")
                raise RetryExhausted(e, attempt)
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            
            # Add jitter (±25%)
            if jitter:
                delay = delay * (0.75 + random.random() * 0.5)
            
            log.info(f"Attempt {attempt} failed ({e}), retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)
        
        except Exception as e:
            # Non-retryable exception — fail immediately
            log.error(f"Non-retryable error: {e}")
            raise
    
    raise RetryExhausted(last_error, max_attempts)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
):
    """
    Decorator for adding retry logic to async functions.
    
    Usage:
        @with_retry(max_attempts=3)
        async def my_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                func, *args,
                max_attempts=max_attempts,
                base_delay=base_delay,
                retryable_exceptions=retryable_exceptions,
                **kwargs,
            )
        return wrapper
    return decorator


class RateLimitHandler:
    """
    Handles rate limit (429) responses with smart backoff.
    Tracks rate limit hits and adjusts delay accordingly.
    """
    
    def __init__(self, base_delay: float = 1.0):
        self.base_delay = base_delay
        self.consecutive_hits = 0
        self._lock = asyncio.Lock()
    
    async def handle_rate_limit(self, retry_after: float = None) -> float:
        """
        Handle a rate limit hit. Returns delay to wait.
        
        Args:
            retry_after: Server-suggested retry delay (from header)
        
        Returns:
            Seconds to wait before retrying
        """
        async with self._lock:
            self.consecutive_hits += 1
            
            if retry_after:
                delay = retry_after
            else:
                # Exponential backoff based on consecutive hits
                delay = self.base_delay * (2 ** min(self.consecutive_hits, 5))
            
            # Add jitter
            delay = delay * (0.75 + random.random() * 0.5)
            
            log.warning(f"Rate limited. Waiting {delay:.1f}s (hit #{self.consecutive_hits})")
            return delay
    
    def reset(self):
        """Call after a successful request."""
        self.consecutive_hits = 0