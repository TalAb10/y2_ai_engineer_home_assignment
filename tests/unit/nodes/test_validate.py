"""Tests for the validate node: schema enforcement + confidence computation."""

import pytest
from graph.nodes.validate import run
from graph.state import GraphState


def test_valid_realestate_params(ctx):
    state = GraphState(
        category="נדל״ן",
        params={"עיר": "ירושלים", "מס׳_חדרים": 3.0, "מחיר": {"max": 1_000_000}},
        classification_certainty=1.0,
        extraction_coverage=0.9,
    )
    result = run(state, ctx)
    assert result["confidence"] > 0.5
    assert "עיר" in result["params"]


def test_unknown_fields_dropped(ctx):
    state = GraphState(
        category="נדל״ן",
        params={"עיר": "ירושלים", "injected_field": "malicious_value"},
        classification_certainty=1.0,
        extraction_coverage=0.8,
    )
    result = run(state, ctx)
    assert "injected_field" not in result["params"]
    assert "עיר" in result["params"]


def test_unknown_category(ctx):
    state = GraphState(category="unknown_cat", params={"foo": "bar"})
    result = run(state, ctx)
    assert result["params"] == {}
    assert result["confidence"] == 0.0
    assert "unknown_category" in result.get("errors", [])


def test_confidence_high_when_rules_covered(ctx):
    state = GraphState(
        category="רכב",
        params={"יצרן": "טויוטה", "דגם": "קורולה", "מחיר": {"max": 70_000}},
        classification_certainty=1.0,
        extraction_coverage=1.0,
    )
    result = run(state, ctx)
    assert result["confidence"] >= 0.7


def test_confidence_rises_with_coverage(ctx):
    low = GraphState(category="רכב", params={"יצרן": "טויוטה"},
                     classification_certainty=1.0, extraction_coverage=0.3)
    high = GraphState(category="רכב", params={"יצרן": "טויוטה"},
                      classification_certainty=1.0, extraction_coverage=0.9)
    assert run(high, ctx)["confidence"] > run(low, ctx)["confidence"]


def test_unambiguous_single_word_is_confident(ctx):
    """A one-word, fully-covered, uncontested query should score high — not be
    penalised for brevity the way the old margin/5 formula did ("טויוטה" → 0.68)."""
    state = GraphState(category="רכב", params={"יצרן": "טויוטה"},
                       classification_certainty=1.0, extraction_coverage=1.0)
    assert run(state, ctx)["confidence"] == 1.0
