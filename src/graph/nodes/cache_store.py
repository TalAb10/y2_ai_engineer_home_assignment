"""Node: cache_store

Persist the validated result to the cache.
Runs only on cache misses (the conditional edge skips it on hits).
"""

from __future__ import annotations

from graph.context import NodeContext
from graph.state import GraphState


def run(state: GraphState, ctx: NodeContext) -> dict:
    if not state.cache_key:
        return {}
    ctx.cache.set(state.cache_key, {
        "category":   state.category,
        "params":     state.params,
        "confidence": state.confidence,
        "notes":      state.notes,
    })
    return {}
