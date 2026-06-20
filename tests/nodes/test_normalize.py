"""Tests for the normalize node."""

import pytest
from graph.nodes.normalize import run
from graph.state import GraphState


def make_state(clean_q: str) -> GraphState:
    return GraphState(raw_q=clean_q, clean_q=clean_q)


def test_million_normalisation(ctx):
    state = make_state("דירה עד 2 מיליון שח")
    result = run(state, ctx)
    assert "2000000" in result["clean_q"]


def test_thousand_normalisation(ctx):
    state = make_state("טויוטה עד 80 אלף שח")
    result = run(state, ctx)
    assert "80000" in result["clean_q"]


def test_area_unit_normalisation(ctx):
    state = make_state("דירה 90 מטר רבוע")
    result = run(state, ctx)
    assert "מ״ר" in result["clean_q"]


def test_km_unit_normalisation(ctx):
    state = make_state("טויוטה 50000 קמ")
    result = run(state, ctx)
    assert "ק״מ" in result["clean_q"]


def test_typo_correction(ctx):
    # "ירושליים" is in the taxonomy typo map → "ירושלים"
    state = make_state("דירה בירושליים")
    result = run(state, ctx)
    assert "ירושלים" in result["clean_q"]


def test_tokens_produced(ctx):
    state = make_state("דירת 3 חדרים בירושלים")
    result = run(state, ctx)
    assert isinstance(result["query_words"], list)
    assert len(result["query_words"]) >= 3


def test_range_alias_normalisation(ctx):
    state = make_state("דירה מינימום 2 חדרים")
    result = run(state, ctx)
    assert "מעל" in result["clean_q"]


def test_normalization_applied_records_single_word(ctx):
    state = make_state("דירה בירושליים")
    result = run(state, ctx)
    # The clitic-prefixed misspelling is recorded against the original token.
    assert result["normalization_applied"] == {"בירושליים": "בירושלים"}


def test_normalization_applied_handles_one_to_many_expansion(ctx):
    # Regression: 'תלאביב' → 'תל אביב-יפו' expands one token into two. A positional
    # zip of before/after tokens used to misalign and fabricate corrections like
    # 'דירה→אביב-יפו'. The change must be recorded only against the real source word.
    state = make_state("תלאביב דירה למכירה")
    result = run(state, ctx)
    assert result["clean_q"] == "תל אביב-יפו דירה למכירה"
    assert result["normalization_applied"] == {"תלאביב": "תל אביב-יפו"}
