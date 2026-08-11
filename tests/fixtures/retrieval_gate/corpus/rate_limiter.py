"""Token bucket rate limiting for per-caller request quotas."""

import threading
import time

DEFAULT_BURST = 20
DEFAULT_REFILL_PER_SECOND = 5.0


class RateLimitExceeded(Exception):
    """Raised when a caller has no tokens left in its bucket."""


class TokenBucket:
    """A token bucket that refills continuously rather than on a fixed tick.

    Continuous refill matters because a fixed tick lets a caller spend a full
    bucket at the end of one window and another full bucket at the start of the
    next, admitting twice the intended burst across the boundary.
    """

    def __init__(self, burst=DEFAULT_BURST, refill_per_second=DEFAULT_REFILL_PER_SECOND):
        self.burst = burst
        self.refill_per_second = refill_per_second
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        """Add the tokens accrued since the last refill, capped at the burst size."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.refill_per_second)
        self._last_refill = now

    def consume(self, tokens=1):
        """Take tokens from the bucket, raising when the quota is exhausted."""
        with self._lock:
            self._refill()
            if self._tokens < tokens:
                raise RateLimitExceeded(f"bucket holds {self._tokens:.2f} tokens, {tokens} requested")
            self._tokens -= tokens
            return self._tokens

    def retry_after(self, tokens=1):
        """Seconds until the bucket holds enough tokens for the given request."""
        with self._lock:
            self._refill()
            deficit = max(0.0, tokens - self._tokens)
            return deficit / self.refill_per_second


class PerCallerLimiter:
    """One token bucket per caller identity, created lazily on first request."""

    def __init__(self, burst=DEFAULT_BURST, refill_per_second=DEFAULT_REFILL_PER_SECOND):
        self.burst = burst
        self.refill_per_second = refill_per_second
        self._buckets = {}
        self._lock = threading.Lock()

    def check(self, caller_id, tokens=1):
        """Consume from the caller's bucket, creating it if this is their first request."""
        with self._lock:
            bucket = self._buckets.get(caller_id)
            if bucket is None:
                bucket = TokenBucket(self.burst, self.refill_per_second)
                self._buckets[caller_id] = bucket
        return bucket.consume(tokens)
