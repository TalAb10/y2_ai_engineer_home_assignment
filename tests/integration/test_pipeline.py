"""End-to-end pipeline tests — full graph (sanitize → … → validate) via _parse.

Covers the behaviours that only emerge from the whole pipeline: price/year
disambiguation in context, subcategory-consistency filtering, and the
self-learning loop (LLM teaches once, deterministic path reuses it). Pure
data-structure tests live in tests/unit/.
"""

from __future__ import annotations

from llm.client import LLMResult, TokenUsage


async def _parse(query, ctx):
    from graph.build import build_graph
    from graph.state import GraphState
    raw = await build_graph(ctx).ainvoke(GraphState(raw_q=query))
    return GraphState.model_validate(dict(raw))


# ── Price / year disambiguation through the full pipeline ─────────────────────────

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
    final = await _parse("טויוטה בין 2015 ל 2018", ctx)
    assert final.category == "רכב"
    assert final.params.get("שנה") == {"min": 2015, "max": 2018}
    assert "מחיר" not in final.params


async def test_bin_price_range_with_currency(ctx):
    # A currency cue makes the same shape a genuine price range.
    final = await _parse("דירה בין 500000 ל 800000 שח", ctx)
    assert final.params.get("מחיר") == {"min": 500000, "max": 800000}


async def test_year_at_taxonomy_max_survives_validation(ctx):
    # The extractor's accepted year range matches the schema/taxonomy bound, so the
    # boundary year validates instead of being extracted then silently dropped.
    final = await _parse("טויוטה קורולה 2025", ctx)
    assert final.category == "רכב"
    assert final.params.get("שנה") == {"min": 2025, "max": 2025}


async def test_out_of_range_year_is_not_extracted_or_mispriced(ctx):
    # A year beyond the taxonomy range is not accepted as a year — and the price guard
    # keeps it from leaking in as a phantom price either.
    final = await _parse("טויוטה קורולה עד 2027", ctx)
    assert "שנה" not in final.params
    assert "מחיר" not in final.params


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


class _StrollerLLM:
    """Fake LLM: labels the uncovered gap as a subcategory + supplies an embedding."""
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    async def embed(self, text):
        return [0.1, 0.2, 0.3]

    async def complete_structured(self, **_):
        self.calls += 1
        return LLMResult(
            parsed={"segments": [{"text": "עגלת תינוק", "type": "subcategory"}],
                    "normalizations": []},
            refusal=None, usage=TokenUsage(model="fake"),
        )


class _FakeIndex:
    """Fake semantic index: always canonicalises the gap to the 'עגלות' subcategory."""
    def search(self, embedding, field_type=None, k=8, threshold=0.4):
        from types import SimpleNamespace
        return [SimpleNamespace(value="עגלות", field_type="subcategory")]


def _ctx_with_llm(taxonomy, settings, llm, semantic_index=None):
    from cache.cache import Cache
    from graph.context import NodeContext
    from observability import metrics as m
    from patterns.library import PatternLibrary
    from patterns.normalizations import NormalizationDB
    return NodeContext(taxonomy=taxonomy, llm=llm, cache=Cache(100), settings=settings,
                       metrics=m, pattern_library=PatternLibrary(), normalization_db=NormalizationDB(),
                       semantic_index=semantic_index)


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


async def test_semantic_resolution_learned_then_skips_llm(taxonomy, settings):
    # A phrase the semantic index has to canonicalise ("עגלת תינוק" → "עגלות") is
    # learned as a normalization, so the second identical-shape query resolves on the
    # deterministic path with no LLM.
    llm = _StrollerLLM()
    ctx = _ctx_with_llm(taxonomy, settings, llm, semantic_index=_FakeIndex())

    first = await _parse("עגלת תינוק כחדש עד 600 שח", ctx)
    assert first.llm_used is True
    assert first.params.get("תת_קטגוריה") == "עגלות"
    assert ctx.normalization_db.all().get("עגלת תינוק") == "עגלות"

    llm.calls = 0
    second = await _parse("עגלת תינוק כחדש עד 800 שח", ctx)   # same shape, new number
    assert llm.calls == 0                                       # canonicalised → no LLM
    assert second.params.get("תת_קטגוריה") == "עגלות"
    assert second.params.get("מחיר", {}).get("max") == 800


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
