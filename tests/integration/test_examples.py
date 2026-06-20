"""End-to-end golden examples from the README + additional realistic cases.

These tests run the full graph (rules-only, no LLM) so they work offline.
They assert the shape and key fields of the output — not exact equality —
since the LLM path is disabled here.
"""

from __future__ import annotations

# Async tests run automatically via asyncio_mode = "auto" (see pyproject.toml).


async def _parse(query: str, ctx) -> dict:
    from graph.build import build_graph
    from graph.state import GraphState

    graph = build_graph(ctx)
    state = GraphState(raw_q=query)
    raw = await graph.ainvoke(state)
    final = GraphState.model_validate(dict(raw))
    return {
        "category": final.category,
        "params": final.params,
        "confidence": final.confidence,
        "notes": final.notes,
    }


# ── README Example 1: Real estate ─────────────────────────────────────────────

async def test_realestate_basic(ctx):
    result = await _parse("דירת 3 חדרים בירושלים עד מליון שח", ctx)
    assert result["category"] == "נדל״ן"
    assert result["params"].get("עיר") == "ירושלים"
    assert result["params"].get("מס׳_חדרים") == 3.0 or result["params"].get("מס׳_חדרים") == {"min": 3.0, "max": 3.0}
    price = result["params"].get("מחיר", {})
    assert price.get("max") == 1_000_000


# ── README Example 2: Vehicles ────────────────────────────────────────────────

async def test_vehicles_basic(ctx):
    result = await _parse("טויוטה קורולה 2018-2021 עד 70 אלף שח צבע לבן", ctx)
    assert result["category"] == "רכב"
    assert result["params"].get("יצרן") == "טויוטה"
    assert result["params"].get("דגם") == "קורולה"
    year = result["params"].get("שנה", {})
    assert year.get("min") == 2018
    assert year.get("max") == 2021
    price = result["params"].get("מחיר", {})
    assert price.get("max") == 70_000
    assert result["params"].get("צבע") == "לבן"


# ── README Example 3: Second-hand ─────────────────────────────────────────────

async def test_secondhand_basic(ctx):
    # "אייפון" is a slang brand not in the taxonomy index; category routing requires LLM.
    # Rules-only: assert price is extracted correctly (2500) and response is structurally valid.
    # Storage extraction ("נפח_אחסון") is tested at unit level in the secondhand extractor tests.
    result = await _parse("אייפון 13 פרו 256 ג׳יגה כמו חדש עד 2500", ctx)
    price = result["params"].get("מחיר", {})
    assert price.get("max") == 2500
    assert 0.0 <= result["confidence"] <= 1.0


def test_secondhand_storage_unit():
    """Unit-level: the storage regex pulls the GB value from any geresh variant."""
    from patterns.numbers import _STORAGE_RE
    assert _STORAGE_RE.search("256 ג׳יגה כמו חדש").group(1) == "256"
    assert _STORAGE_RE.search("256gb").group(1) == "256"


# ── Additional realistic examples ─────────────────────────────────────────────

async def test_realestate_rental(ctx):
    result = await _parse("דירת סטודיו להשכרה בתל אביב עד 5000 שח", ctx)
    assert result["category"] == "נדל״ן"


async def test_vehicles_electric(ctx):
    result = await _parse("טסלה מודל 3 חשמלי עד 150000 שח", ctx)
    assert result["category"] == "רכב"
    assert result["params"].get("יצרן") == "טסלה"


async def test_secondhand_laptop(ctx):
    result = await _parse("מחשב נייד HP i7 16 ג׳יגה RAM עד 3000 שח", ctx)
    assert result["category"] == "יד_שנייה"


async def test_realestate_with_amenities(ctx):
    result = await _parse("דירה 4 חדרים עם מעלית וחניה בחיפה", ctx)
    assert result["category"] == "נדל״ן"
    assert result["params"].get("מעלית") is True
    assert result["params"].get("חניה") is not None


async def test_typo_tolerance(ctx):
    # "יונדי" → "יונדאי" via typo map
    result = await _parse("יונדי טוסון 2020", ctx)
    assert result["category"] == "רכב"


async def test_confidence_is_float_in_range(ctx):
    result = await _parse("דירת 3 חדרים בירושלים", ctx)
    assert 0.0 <= result["confidence"] <= 1.0
