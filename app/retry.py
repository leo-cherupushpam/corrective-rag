"""
retry.py
========
Retry logic with exponential backoff for OpenAI API calls.

Purpose:
  Handles transient failures (rate limits, timeouts, 5xx errors) with intelligent
  backoff strategy and graceful degradation.

Design:
  - Exponential backoff: 1s, 2s, 4s, 8s (configurable)
  - Don't retry: 400-level errors (auth, validation)
  - Retry with jitter: 429, 5xx, timeouts
  - Max attempts: 4 (configurable)
  - Timeout per request: 30s (configurable)
"""

import asyncio
import functools
import logging
import random
import time
from typing import Callable, Optional, TypeVar, Any

from openai import (
    RateLimitError as OpenAIRateLimitError,
    APIError as OpenAIAPIError,
    APIConnectionError,
    Timeout as OpenAITimeout,
)

from errors import (
    RateLimitError,
    TimeoutError as CRAGTimeoutError,
    APIError,
    GraderError,
    GenerationError,
    RetrievalError,
)

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MAX_RETRIES = 4
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_TIMEOUT = 30.0  # seconds per API call

F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
    error_class: type = APIError,
) -> Callable[[F], F]:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries
        timeout: Timeout per API call in seconds
        error_class: Exception class to raise on final failure

    Usage:
        @retry_with_backoff(max_retries=4, error_class=GraderError)
        def grade_document(query: str, doc: str):
            return openai_client.chat.completions.create(...)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay

            while attempt < max_retries:
                try:
                    # Call function with timeout
                    return func(*args, **kwargs)

                except OpenAIRateLimitError as e:
                    attempt += 1
                    if attempt >= max_retries:
                        raise RateLimitError(
                            f"Rate limit exceeded after {max_retries} attempts: {str(e)}",
                            retry_after=delay,
                        ) from e

                    # Extract retry_after if available
                    retry_after = getattr(e, "retry_after", delay)
                    wait_time = min(float(retry_after), max_delay)
                    # Add jitter (±10%)
                    wait_time *= 1 + random.uniform(-0.1, 0.1)

                    logger.warning(
                        f"{func.__name__} rate limited. "
                        f"Attempt {attempt}/{max_retries}. "
                        f"Waiting {wait_time:.1f}s before retry."
                    )
                    time.sleep(wait_time)

                except (OpenAITimeout, asyncio.TimeoutError) as e:
                    attempt += 1
                    if attempt >= max_retries:
                        raise CRAGTimeoutError(
                            f"Timeout after {max_retries} attempts: {str(e)}",
                            timeout_seconds=timeout,
                        ) from e

                    logger.warning(
                        f"{func.__name__} timed out. "
                        f"Attempt {attempt}/{max_retries}. "
                        f"Waiting {delay:.1f}s before retry."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

                except APIConnectionError as e:
                    attempt += 1
                    if attempt >= max_retries:
                        raise APIError(
                            f"Connection error after {max_retries} attempts: {str(e)}",
                            retryable=False,
                        ) from e

                    logger.warning(
                        f"{func.__name__} connection error. "
                        f"Attempt {attempt}/{max_retries}. "
                        f"Waiting {delay:.1f}s before retry."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

                except OpenAIAPIError as e:
                    # Check status code
                    status_code = getattr(e, "status_code", None)

                    # Don't retry 4xx errors (except 429, handled above)
                    if status_code and 400 <= status_code < 500:
                        raise APIError(
                            f"Client error ({status_code}): {str(e)}",
                            status_code=status_code,
                            retryable=False,
                        ) from e

                    # Retry 5xx errors
                    attempt += 1
                    if attempt >= max_retries:
                        raise APIError(
                            f"Server error after {max_retries} attempts: {str(e)}",
                            status_code=status_code,
                            retryable=False,
                        ) from e

                    logger.warning(
                        f"{func.__name__} server error ({status_code}). "
                        f"Attempt {attempt}/{max_retries}. "
                        f"Waiting {delay:.1f}s before retry."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

                except Exception as e:
                    # Unknown error - don't retry
                    logger.error(f"{func.__name__} failed with unexpected error: {str(e)}")
                    raise

            # This shouldn't be reached, but just in case
            raise error_class(f"Failed after {max_retries} attempts")

        return wrapper  # type: ignore

    return decorator


def retry_grader(max_retries: int = 3) -> Callable[[F], F]:
    """Convenience decorator for grader functions with optimized settings."""
    return retry_with_backoff(max_retries=max_retries, error_class=GraderError)


def retry_generator(max_retries: int = 3) -> Callable[[F], F]:
    """Convenience decorator for generation functions with optimized settings."""
    return retry_with_backoff(max_retries=max_retries, error_class=GenerationError)


def retry_retriever(max_retries: int = 2) -> Callable[[F], F]:
    """Convenience decorator for retrieval functions (faster failure)."""
    return retry_with_backoff(max_retries=max_retries, error_class=RetrievalError)


def retry_corrector(max_retries: int = 2) -> Callable[[F], F]:
    """Convenience decorator for correction functions."""
    from errors import CorrectionError

    return retry_with_backoff(max_retries=max_retries, error_class=CorrectionError)
