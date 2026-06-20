"""Graph builder — the ONLY file that knows the pipeline shape.

Pipeline:

    START
     → sanitize         input hygiene + injection detection
     → security_check   LLM deepcheck (only when security_flags non-empty)
          ├─(injection confirmed)──────────────────────────────────→ END
          └─(legitimate / no LLM)
     → normalize        typo correction, unit normalisation, tokenise
     → cache_lookup ──(hit)──────────────────────────────────────→ END
     → extract          pattern scan → LLM gap-fill; category falls out of segments
     → validate         schema enforcement + confidence
     → cache_store      persist to LRU / Redis
     → END

Adding or reordering a node is a one-line edit here.
Node files never reference each other.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.context import NodeContext
from graph.nodes import cache_lookup, cache_store, extract, normalize, sanitize, security_check, validate
from graph.state import GraphState


def _bind(node_module, ctx: NodeContext):
    """Close ctx into the node so LangGraph sees fn(state) -> dict."""
    import inspect

    async def wrapper(state: GraphState) -> dict:
        if inspect.iscoroutinefunction(node_module.run):
            return await node_module.run(state, ctx)
        return node_module.run(state, ctx)

    return wrapper


def _route_after_sanitize(state: GraphState) -> str:
    return "security_check" if state.security_flags else "normalize"


def _route_after_security_check(state: GraphState) -> str:
    return END if state.injection_confirmed else "normalize"


def _route_after_cache(state: GraphState) -> str:
    return END if state.cache_hit else "extract"


def build_graph(ctx: NodeContext) -> StateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("sanitize",       _bind(sanitize,       ctx))
    builder.add_node("security_check", _bind(security_check, ctx))
    builder.add_node("normalize",      _bind(normalize,      ctx))
    builder.add_node("cache_lookup",   _bind(cache_lookup,   ctx))
    builder.add_node("extract",        _bind(extract,        ctx))
    builder.add_node("validate",       _bind(validate,       ctx))
    builder.add_node("cache_store",    _bind(cache_store,    ctx))

    builder.set_entry_point("sanitize")
    builder.add_conditional_edges("sanitize", _route_after_sanitize,
                                  {"security_check": "security_check", "normalize": "normalize"})
    builder.add_conditional_edges("security_check", _route_after_security_check,
                                  {END: END, "normalize": "normalize"})
    builder.add_edge("normalize",      "cache_lookup")
    builder.add_conditional_edges("cache_lookup", _route_after_cache, {END: END, "extract": "extract"})
    builder.add_edge("extract",        "validate")
    builder.add_edge("validate",       "cache_store")
    builder.add_edge("cache_store",    END)

    return builder.compile()
