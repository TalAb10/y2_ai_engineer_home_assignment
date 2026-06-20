"""Unit tests for the complete node — taxonomy-driven brand inference.

complete.run is synchronous and pure (state + ctx → dict), so these call it directly.
"""

from __future__ import annotations

from graph.nodes import complete
from graph.state import GraphState
from patterns.segment_types import CAT_SH


def _state(clean_q: str, params: dict, category: str = CAT_SH) -> GraphState:
    return GraphState(raw_q=clean_q, clean_q=clean_q, category=category, params=dict(params))


def test_product_alias_infers_brand(ctx):
    # "אייפון" is not a taxonomy brand, but the alias layer maps it to אפל.
    out = complete.run(_state("אייפון 13 פרו 256 ג׳יגה", {"תת_קטגוריה": "טלפונים_סלולריים"}), ctx)
    assert out["params"]["מותג"] == "אפל"


def test_direct_brand_recognized_and_subcategory_backfilled(ctx):
    # "דל" is a taxonomy brand unique to laptops → brand + subcategory + sector inferred.
    out = complete.run(_state("דל לפטופ i7", {}), ctx)
    assert out["params"]["מותג"] == "דל"
    assert out["params"]["תת_קטגוריה"] == "מחשבים_ניידים"
    assert out["params"]["סקטור"] == "אלקטרוניקה"


def test_ambiguous_brand_sets_brand_without_subcategory(ctx):
    # אפל is listed under both phones and laptops → set brand, don't guess subcategory.
    out = complete.run(_state("אפל משהו", {}), ctx)
    assert out["params"]["מותג"] == "אפל"
    assert "תת_קטגוריה" not in out["params"]


def test_brand_not_valid_for_known_subcategory_is_skipped(ctx):
    # סמסונג is phones/TVs, not laptops → must not be attached to a laptop query.
    out = complete.run(_state("סמסונג", {"תת_קטגוריה": "מחשבים_ניידים"}), ctx)
    assert out == {}


def test_no_brand_for_brandless_category(ctx):
    # Furniture has no brands in the taxonomy → nothing invented.
    out = complete.run(_state("ספה פינתית", {"תת_קטגוריה": "סלון"}), ctx)
    assert out == {}


def test_non_secondhand_is_noop(ctx):
    out = complete.run(_state("טויוטה קורולה", {"יצרן": "טויוטה"}, category="רכב"), ctx)
    assert out == {}


def test_existing_brand_is_untouched(ctx):
    out = complete.run(_state("אייפון", {"תת_קטגוריה": "טלפונים_סלולריים", "מותג": "כבר"}), ctx)
    assert out == {}
