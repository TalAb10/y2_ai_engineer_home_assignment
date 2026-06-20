"""Unit tests for the taxonomy matching layer (taxonomy/loader.py).

These helpers are the foundation of all deterministic extraction — clitic-prefix
stripping, exact/inflectional/prefix matching against canonical value sets. The
Hebrew morphology here is subtle, so it gets direct coverage rather than only
being exercised through the pipeline.
"""

from __future__ import annotations

from taxonomy.loader import match_in_set, strip_prefixes


# ── strip_prefixes ───────────────────────────────────────────────────────────────

def test_strip_prefixes_returns_original_plus_destripped():
    # "ב" is a clitic prefix; the de-prefixed stem is offered alongside the original.
    assert strip_prefixes("בתל") == ["בתל", "תל"]


def test_strip_prefixes_no_prefix():
    assert strip_prefixes("תל") == ["תל"]


def test_strip_prefixes_single_char_not_stripped():
    # A lone prefix letter has no stem to keep — nothing to strip.
    assert strip_prefixes("ב") == ["ב"]


# ── match_in_set ─────────────────────────────────────────────────────────────────

def test_exact_city_match(taxonomy):
    assert match_in_set("ירושלים", taxonomy.re_cities) == "ירושלים"


def test_clitic_prefixed_city_match(taxonomy):
    # "בירושלים" → strip "ב" → "ירושלים".
    assert match_in_set("בירושלים", taxonomy.re_cities) == "ירושלים"


def test_city_prefix_extends_to_canonical_suffix(taxonomy):
    # "תל אביב" is a prefix of the canonical "תל אביב-יפו" (extends via "-").
    assert match_in_set("תל אביב", taxonomy.re_cities) == "תל אביב-יפו"
    assert match_in_set("בתל אביב", taxonomy.re_cities) == "תל אביב-יפו"


def test_inflectional_property_match(taxonomy):
    # Construct state swaps final ה↔ת: "דירת" matches "דירה".
    assert match_in_set("דירת", taxonomy.re_property_types, whole=True) == "דירה"


def test_multiword_property_match(taxonomy):
    assert match_in_set("דירת סטודיו", taxonomy.re_property_types, whole=True) == "דירת סטודיו"


def test_no_match_returns_none(taxonomy):
    assert match_in_set("פלאפל", taxonomy.re_cities) is None
