"""Node: cache_lookup

Check the cache before any extraction work. On a hit, populate the result fields
and set cache_hit=True so build.py routes straight to END — extract, validate, and
cache_store are all skipped. This is the ≤150ms fast path.
"""

from __future__ import annotations

from cache.cache import make_cache_key
from graph.context import NodeContext
from graph.state import GraphState


def run(state: GraphState, ctx: NodeContext) -> dict:
    key = make_cache_key(state.clean_q)
    cached = ctx.cache.get(key)

    if cached is None:
        ctx.metrics.cache_misses_total.inc()
        return {"cache_key": key, "cache_hit": False}

    ctx.metrics.cache_hits_total.inc()
    return {
        "cache_key": key,
        "cache_hit": True,
        "category":   cached.get("category", ""),
        "params":     cached.get("params", {}),
        "confidence": cached.get("confidence", 0.0),
        "notes":      cached.get("notes", []),
    }
