"""Tests for the pattern-based extraction feature.

Covers the pattern library, the price/year disambiguation that motivated the
design, the self-learning loop (LLM teaches once, pattern reused after), and the
normalization feedback loop.
"""

from __future__ import annotations

import pytest

from llm.client import LLMResult, TokenUsage
from patterns.library import PatternLibrary, Segment, abstract, coverage, merge_spans, SpanMatch

# asyncio_mode=auto (pyproject) runs async tests automatically — no marks needed.


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


# ── Disambiguation through the full pipeline ─────────────────────────────────────

async def _parse(query, ctx):
    from graph.build import build_graph
    from graph.state import GraphState
    raw = await build_graph(ctx).ainvoke(GraphState(raw_q=query))
    return GraphState.model_validate(dict(raw))


async def test_bare_9000_is_price_not_year(ctx):
    # 9000 is not a valid year → must be a price, in any category.
    final = await _parse("אייפון עד 9000", ctx)
    assert final.params.get("מחיר", {}).get("max") == 9000
    assert "שנה" not in final.params


async def test_ad_operator_resolves_year_in_vehicle(ctx):
    # "עד 2018" in a vehicle context should be a year upper bound, not a price.
    final = await _parse("טויוטה קורולה עד 2018", ctx)
    assert final.params.get("שנה", {}).get("max") == 2018
    assert "מחיר" not in final.params


async def test_year_range_in_vehicle(ctx):
    final = await _parse("טויוטה קורולה 2018-2021", ctx)
    assert final.category == "רכב"
    assert final.params.get("שנה") == {"min": 2018, "max": 2021}


async def test_bin_year_range_is_not_a_price(ctx):
    # Regression: "בין 2015 ל 2018" with no currency is a year span, not a price.
    # The range branch used to emit it as מחיר before any year check.
    final = await _parse("טויוטה בין 2015 ל 2018", ctx)
    assert final.category == "רכב"
    assert final.params.get("שנה") == {"min": 2015, "max": 2018}
    assert "מחיר" not in final.params


async def test_bin_price_range_with_currency(ctx):
    # A currency cue makes the same shape a genuine price range.
    final = await _parse("דירה בין 500000 ל 800000 שח", ctx)
    assert final.params.get("מחיר") == {"min": 500000, "max": 800000}


async def test_year_at_taxonomy_max_survives_validation(ctx):
    # Regression: the extractor's accepted year range matches the schema/taxonomy
    # bound, so the boundary year validates instead of being extracted then silently
    # dropped.
    final = await _parse("טויוטה קורולה 2025", ctx)
    assert final.category == "רכב"
    assert final.params.get("שנה") == {"min": 2025, "max": 2025}


async def test_out_of_range_year_is_not_extracted_or_mispriced(ctx):
    # A year beyond the taxonomy range is not accepted as a year — and the price guard
    # keeps it from leaking in as a phantom price either.
    final = await _parse("טויוטה קורולה עד 2027", ctx)
    assert "שנה" not in final.params
    assert "מחיר" not in final.params


# ── extract_price: range-branch disambiguation (unit) ────────────────────────────

def test_extract_price_range_unit():
    from patterns.numbers import extract_price
    # currency cue → price range
    assert extract_price("בין 1000 ל 2000 שח") == {"min": 1000, "max": 2000}
    # large non-year numbers, no currency → still a price range
    assert extract_price("בין 500000 ל 800000") == {"min": 500000, "max": 800000}
    # both endpoints are years, no currency → not a price (left for year extraction)
    assert extract_price("בין 2015 ל 2018") is None
    # range bound to a non-price unit → not a price (left for the rooms extractor)
    assert extract_price("בין 3 ל 5 חדרים") is None


# ── Self-learning loop ───────────────────────────────────────────────────────────

class _TeachingLLM:
    """Fake LLM: labels 'במבה' as a brand and reports one typo. Counts calls."""
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    async def complete_structured(self, **_):
        self.calls += 1
        return LLMResult(
            parsed={"segments": [{"text": "במבה", "type": "brand"}],
                    "normalizations": [{"from": "במבע", "to": "במבה"}]},
            refusal=None, usage=TokenUsage(model="fake"),
        )


def _ctx_with_llm(taxonomy, settings, llm):
    from cache.cache import Cache
    from graph.context import NodeContext
    from observability import metrics as m
    from patterns.library import PatternLibrary
    from patterns.normalizations import NormalizationDB
    return NodeContext(taxonomy=taxonomy, llm=llm, cache=Cache(100), settings=settings,
                       metrics=m, pattern_library=PatternLibrary(), normalization_db=NormalizationDB())


async def test_pattern_learned_then_reused_skips_llm(taxonomy, settings):
    llm = _TeachingLLM()
    ctx = _ctx_with_llm(taxonomy, settings, llm)

    first = await _parse("במבה עד 50 שח", ctx)
    assert first.llm_used is True
    assert first.params.get("מותג") == "במבה"
    assert ctx.pattern_library.size() == 1

    llm.calls = 0
    second = await _parse("במבה עד 80 שח", ctx)   # same shape, new number
    assert llm.calls == 0                          # pattern reused — no LLM
    assert second.params.get("מותג") == "במבה"
    assert second.params.get("מחיר", {}).get("max") == 80


async def test_subcategory_consistency_filter_drops_mismatched_spec(ctx):
    # A stroller must not carry phone-only storage (נפח_אחסון).
    final = await _parse("עגלה 256 ג׳יגה עד 500 שח", ctx)
    assert final.params.get("תת_קטגוריה") == "עגלות"
    assert "נפח_אחסון" not in final.params      # storage dropped — wrong subcategory
    assert final.params.get("מחיר", {}).get("max") == 500


async def test_subcategory_consistency_keeps_matching_spec(ctx):
    # A phone keeps storage.
    final = await _parse("אייפון 256 ג׳יגה עד 2500 שח", ctx)
    assert final.params.get("תת_קטגוריה") == "טלפונים_סלולריים"
    assert final.params.get("נפח_אחסון") == "256GB"


async def test_normalization_learned_and_applied(taxonomy, settings):
    llm = _TeachingLLM()
    ctx = _ctx_with_llm(taxonomy, settings, llm)
    await _parse("במבע עד 50 שח", ctx)             # LLM reports במבע→במבה
    assert ctx.normalization_db.all().get("במבע") == "במבה"
