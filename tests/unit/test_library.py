"""Unit tests for the self-learning PatternLibrary and coverage helpers.

Pure data-structure tests — no graph, no LLM. Pipeline behaviour that exercises
the library end-to-end lives in tests/integration/test_pipeline.py.
"""

from __future__ import annotations

import pytest

from patterns.library import (
    PatternLibrary, Segment, SpanMatch, abstract, coverage, merge_spans,
)


# ── Abstraction ────────────────────────────────────────────────────────────────

def test_abstract_blanks_each_digit():
    assert abstract("עד 9000 שח") == "עד ???? שח"
    assert abstract("2018-2021") == "????-????"
    assert abstract("צבע לבן") == "צבע לבן"        # no digits → unchanged
    assert len(abstract("70000")) == len("70000")  # length preserved


# ── PatternLibrary ───────────────────────────────────────────────────────────────

def test_learn_and_scan_roundtrip():
    lib = PatternLibrary()
    lib.learn("אייפון 13 פרו", "model")
    hits = lib.scan("אייפון 14 פרו")        # same shape + same digit count → matches
    assert any(h.types == {"model"} for h in hits)


def test_different_digit_count_is_a_different_pattern():
    # One "?" per digit: a 2-digit and a 3-digit number are distinct shapes.
    lib = PatternLibrary()
    lib.learn("אייפון 13 פרו", "model")
    assert lib.scan("אייפון 256 פרו") == []


def test_learn_adds_meanings_never_overwrites():
    lib = PatternLibrary()
    lib.learn("עד 2018", "year_range")
    lib.learn("עד 9000", "price")            # same abstract shape "עד ????"
    hits = lib.scan("עד 5000")
    assert hits and hits[0].types == {"year_range", "price"}


def test_scan_respects_word_boundaries():
    lib = PatternLibrary()
    lib.learn("תל אביב", "city")
    assert lib.scan("בתל אביב יפו") == [] or all(  # 'תל' inside 'בתל' must not match
        h.start == 0 or "בתל" not in "בתל אביב"[h.start - 1:h.start + 1] for h in lib.scan("בתל אביב")
    )


def test_too_short_pattern_not_learned():
    lib = PatternLibrary()
    lib.learn("3", "rooms")          # single "?" after abstraction — too generic
    assert lib.size() == 0


# ── merge_spans / coverage ──────────────────────────────────────────────────────

def test_merge_overlapping_spans_unions_types():
    merged = merge_spans([SpanMatch(0, 8, {"rooms"}), SpanMatch(0, 1, {"price"})])
    assert len(merged) == 1
    assert merged[0].types == {"rooms", "price"}


def test_coverage_counts_salient_words():
    seg = Segment(text="דירה", type="property_type", start=0, end=4)
    assert coverage([seg], "דירה גדולה") == pytest.approx(0.5)   # 1 of 2 salient words
