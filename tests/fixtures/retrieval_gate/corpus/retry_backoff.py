"""Exponential backoff with jitter for retrying transient failures."""

import logging
import random
import time

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 30.0


class RetriesExhausted(Exception):
    """Raised when every attempt failed."""


def backoff_delay(attempt, base=DEFAULT_BASE_DELAY, cap=DEFAULT_MAX_DELAY):
    """Delay before a given attempt, exponentially increasing and capped."""
    return min(cap, base * (2**attempt))


def jittered_delay(attempt, base=DEFAULT_BASE_DELAY, cap=DEFAULT_MAX_DELAY, rng=random):
    """Backoff delay with full jitter applied.

    Full jitter spreads retries across the whole interval rather than clustering
    them at its end. Without it, every client that failed at the same moment
    retries at the same moment, and the outage repeats on a schedule.
    """
    return rng.uniform(0, backoff_delay(attempt, base, cap))


def retry(operation, retry_on=(OSError,), max_attempts=DEFAULT_MAX_ATTEMPTS, sleep=time.sleep):
    """Call an operation, retrying listed exception types with jittered backoff."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return operation()
        except retry_on as exc:
            last_error = exc
            if attempt + 1 >= max_attempts:
                break
            delay = jittered_delay(attempt)
            logger.warning("Attempt %d failed (%s), retrying in %.2fs", attempt + 1, exc, delay)
            sleep(delay)
    raise RetriesExhausted(f"all {max_attempts} attempts failed") from last_error
