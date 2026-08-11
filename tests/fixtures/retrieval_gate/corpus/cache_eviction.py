"""Least-recently-used cache with a bounded entry count."""

import threading
from collections import OrderedDict

DEFAULT_CAPACITY = 512


class LRUCache:
    """A mapping that evicts the least recently used entry once it is full.

    Backed by an OrderedDict rather than a plain dict plus a list, because
    move_to_end and popitem are both constant time, while maintaining recency in
    a list makes every hit a linear scan.
    """

    def __init__(self, capacity=DEFAULT_CAPACITY):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key, default=None):
        """Return a cached value and mark it most recently used."""
        with self._lock:
            if key not in self._entries:
                self.misses += 1
                return default
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key]

    def put(self, key, value):
        """Insert a value, evicting the least recently used entry if full."""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = value
            if len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
                self.evictions += 1

    def invalidate(self, key):
        """Drop one entry, returning whether it was present."""
        with self._lock:
            return self._entries.pop(key, None) is not None

    def hit_ratio(self):
        """Fraction of lookups served from the cache, or zero before any lookup."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def clear(self):
        """Drop every entry and reset the counters."""
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
