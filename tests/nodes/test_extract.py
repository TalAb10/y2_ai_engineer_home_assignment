"""Tests for the extract node — specifically the LLM-segment validation gate.

These use a *fake* LLM (deterministic, offline) so the LLM's output is fixed and
the behaviour under test is the extract node's handling of it, not the model.
"""

from __future__ import annotations

import dataclasses

import pytest

from graph.context import NodeContext
from graph.nodes import extract
from graph.state import GraphState
from llm.client import LLMResult, TokenUsage
from patterns.library import PatternLibrary
from patterns.normalizations import NormalizationDB


class _FakeLLM:
    """LLM stub returning a fixed segmentation, so tests are deterministic."""

    def __init__(self, segments: list[dict], normalizations: list[dict] | None = None):
        self._segments = segments
        self._normalizations = normalizations or []

    def is_available(self) -> bool:
        return True

    async def embed(self, text: str):
        return None  # no semantic hints → exercises the raw-text path

    async def complete_structured(self, **_kwargs) -> LLMResult:
        return LLMResult(
            parsed={"segments": self._segments, "normalizations": self._normalizations},
            refusal=None,
            usage=TokenUsage(),
        )


def _ctx_with_llm(taxonomy, mock_cache, settings, llm) -> NodeContext:
    from observability import metrics as m
    return NodeContext(
        taxonomy=taxonomy, llm=llm, cache=mock_cache, settings=settings, metrics=m,
        pattern_library=PatternLibrary(), normalization_db=NormalizationDB(),
        semantic_index=None,
    )


@pytest.mark.asyncio
async def test_unresolvable_llm_label_does_not_drive_classification(taxonomy, mock_cache, settings):
    """Regression: "במבע" ("on sale") mislabelled transaction_mode must NOT make
    the query real-estate.

    "במבצע" is not a real-estate transaction mode (those are מכירה/השכרה/…), so its
    value-extractor yields nothing. Such a segment must be dropped — not added to
    the segment list where it would cast a נדל״ן vote while contributing no value.
    """
    q = "במבצע עד 50 שח"
    llm = _FakeLLM(segments=[{"text": "במבצע", "type": "transaction_mode"}])
    ctx = _ctx_with_llm(taxonomy, mock_cache, settings, llm)
    state = GraphState(raw_q=q, clean_q=q, query_words=q.split())

    out = await extract.run(state, ctx)

    # The bogus transaction_mode segment is gone.
    assert all(s["type"] != "transaction_mode" for s in out["segments"]), out["segments"]
    # And it did not pull the query into real estate.
    assert out["category"] != "נדל״ן", out
    # No מצבי_עסקה param was fabricated from an unresolvable label.
    assert "מצבי_עסקה" not in out["params"]


@pytest.mark.asyncio
async def test_resolvable_llm_label_is_kept(taxonomy, mock_cache, settings):
    """A legitimate transaction_mode ("השכרה") still resolves and classifies as נדל״ן."""
    q = "להשכרה עד 4000 שח"
    llm = _FakeLLM(segments=[{"text": "השכרה", "type": "transaction_mode"}])
    ctx = _ctx_with_llm(taxonomy, mock_cache, settings, llm)
    state = GraphState(raw_q=q, clean_q=q, query_words=q.split())

    out = await extract.run(state, ctx)

    assert out["category"] == "נדל״ן", out
    assert any(s["type"] == "transaction_mode" for s in out["segments"]), out["segments"]
    assert out["params"].get("מצבי_עסקה") == ["השכרה"]


@pytest.mark.asyncio
async def test_freetext_llm_label_is_kept(taxonomy, mock_cache, settings):
    """Free-text types (brand) always resolve and are never dropped by the gate."""
    q = "פנדר עד 4000 שח"
    llm = _FakeLLM(segments=[{"text": "פנדר", "type": "brand"}])
    ctx = _ctx_with_llm(taxonomy, mock_cache, settings, llm)
    state = GraphState(raw_q=q, clean_q=q, query_words=q.split())

    out = await extract.run(state, ctx)

    assert any(s["type"] == "brand" for s in out["segments"]), out["segments"]
    assert out["category"] == "יד_שנייה", out
