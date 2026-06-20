"""Unit tests for the in-process LRU cache (cache/cache.py)."""

from __future__ import annotations

from cache.cache import Cache, make_cache_key


# ── make_cache_key ───────────────────────────────────────────────────────────────

def test_cache_key_is_stable_and_normalised():
    # Same query (modulo surrounding whitespace + case) → same key.
    assert make_cache_key("דירה בתל אביב") == make_cache_key("  דירה בתל אביב  ")
    assert make_cache_key("iPhone") == make_cache_key("iphone")


def test_cache_key_differs_for_different_queries():
    assert make_cache_key("דירה 3 חדרים") != make_cache_key("דירה 4 חדרים")


# ── Cache ────────────────────────────────────────────────────────────────────────

def test_get_miss_returns_none():
    assert Cache().get("nope") is None


def test_set_then_get_roundtrip():
    c = Cache()
    c.set("k", {"category": "רכב"})
    assert c.get("k") == {"category": "רכב"}


def test_lru_evicts_oldest_over_capacity():
    c = Cache(capacity=2)
    c.set("a", {"v": 1})
    c.set("b", {"v": 2})
    c.set("c", {"v": 3})          # exceeds capacity → "a" (oldest) evicted
    assert c.get("a") is None
    assert c.get("b") == {"v": 2}
    assert c.get("c") == {"v": 3}


def test_get_refreshes_recency():
    c = Cache(capacity=2)
    c.set("a", {"v": 1})
    c.set("b", {"v": 2})
    c.get("a")                    # "a" is now most-recently-used
    c.set("c", {"v": 3})          # "b" is now oldest → evicted, not "a"
    assert c.get("a") == {"v": 1}
    assert c.get("b") is None


def test_health_ok():
    assert Cache().health() == {"ok": True}
