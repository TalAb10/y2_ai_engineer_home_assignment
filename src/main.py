"""FastAPI application entry point.

Endpoints:
  POST /parse   — main parsing endpoint
  GET  /health  — liveness check
  GET  /metrics — Prometheus exposition
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from cache.cache import create_cache
from config import get_settings
from graph.build import build_graph
from graph.context import NodeContext
from graph.state import GraphState
from llm.client import create_llm_client
from observability import metrics as m
from observability.logging_config import log_parse_decision, setup_logging
from patterns.library import PatternLibrary
from patterns.normalizations import NormalizationDB
from taxonomy.loader import load as load_taxonomy
from taxonomy.semantic_index import DEFAULT_INDEX_PATH, DEFAULT_META_PATH, SemanticTaxonomyIndex

_graph = None
_ctx: NodeContext | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _ctx
    settings = get_settings()
    setup_logging(settings.log_level)

    sem_index = None
    if DEFAULT_INDEX_PATH.exists() and DEFAULT_META_PATH.exists():
        try:
            sem_index = SemanticTaxonomyIndex.load(DEFAULT_INDEX_PATH, DEFAULT_META_PATH)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Could not load semantic index: %s", exc)

    _ctx = NodeContext(
        taxonomy=load_taxonomy(settings.taxonomy_path),
        llm=create_llm_client(settings),
        cache=create_cache(settings),
        settings=settings,
        metrics=m,
        pattern_library=PatternLibrary(),
        normalization_db=NormalizationDB(),
        semantic_index=sem_index,
    )
    _graph = build_graph(_ctx)
    yield


app = FastAPI(title="Yad2 Hebrew Search Parser", version="1.0.0", lifespan=lifespan)


class ParseRequest(BaseModel):
    q: str = Field(..., description="Free-text Hebrew search query")
    debug: bool = Field(default=False, description="Include debug info in response")


class DebugInfo(BaseModel):
    """Internal extraction state — only included when request.debug is true."""
    coverage: float
    llm_used: bool
    segments: list[dict[str, Any]]          # each segment has text, type, source ("pattern"|"llm")
    normalization_applied: dict[str, str]   # typo corrections: {original_word: corrected_word}
    taxonomy_hints: dict[str, list[str]]    # FAISS hits that were sent to the LLM as context


class ParseResponse(BaseModel):
    category: str
    params: dict[str, Any]
    confidence: float
    notes: list[str] = Field(default_factory=list)
    debug: DebugInfo | None = None


@app.post("/parse")
async def parse(req: ParseRequest):
    max_chars = _ctx.settings.max_input_chars  # type: ignore[union-attr]
    if len(req.q) > max_chars:
        m.errors_total.labels(error_type="query_too_long").inc()
        return JSONResponse(
            status_code=400,
            content={"error": "query_too_long", "max_chars": max_chars, "got": len(req.q)},
        )

    start = time.perf_counter()
    raw = await _graph.ainvoke(GraphState(raw_q=req.q))  # type: ignore[union-attr]
    final = GraphState.model_validate(dict(raw))

    m.request_latency_seconds.observe(time.perf_counter() - start)

    if final.injection_confirmed:
        m.errors_total.labels(error_type="injection_blocked").inc()
        return JSONResponse(status_code=400, content={"error": "blocked_query"})

    m.requests_total.labels(category=final.category or "unknown").inc()
    log_parse_decision(
        query=req.q, category=final.category, confidence=final.confidence,
        cache_hit=final.cache_hit, llm_used=final.llm_used,
    )

    debug_info = None
    if req.debug:
        debug_info = DebugInfo(
            coverage=final.extraction_coverage,
            llm_used=final.llm_used,
            segments=final.segments,
            normalization_applied=final.normalization_applied,
            taxonomy_hints=final.taxonomy_hints,
        )

    return ParseResponse(category=final.category, params=final.params,
                         confidence=final.confidence, notes=final.notes,
                         debug=debug_info)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_available": _ctx.llm.is_available() if _ctx else False,
        "cache": _ctx.cache.health() if _ctx else {},
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
