"""Cache — in-process LRU only.

Key is SHA-256 of the NFKC-normalised query string.

Why NOT semantic caching for params: near-misses ("עד מליון" ≈ "עד 2 מליון")
share high cosine similarity but must NOT share a cached result — wrong numbers
would be returned.  Exact-normalised key is the safe and correct choice here.
"""

from __future__ import annotations

import hashlib
import threading
import unicodedata
from collections import OrderedDict
from typing import Any


def make_cache_key(clean_q: str) -> str:
    normalised = unicodedata.normalize("NFKC", clean_q.strip().lower())
    return "yad2:" + hashlib.sha256(normalised.encode()).hexdigest()


class Cache:
    """Thread-safe in-process LRU cache."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._capacity = capacity
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def health(self) -> dict[str, bool]:
        return {"ok": True}


def create_cache(settings: Any) -> Cache:
    return Cache(capacity=settings.cache_size)
