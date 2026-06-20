"""Deterministic span discovery — the offline floor under the pattern library.

Finds non-numeric candidate spans (enums, names, city, model, slang) straight from
the taxonomy, with no LLM and no learned patterns. Its output feeds the same
merge → validate → resolve pipeline as PatternLibrary.scan, so a query works on
day one (cold cache, LLM disabled). The pattern library only adds shapes the LLM
later teaches (free-text models, novel slang, unusual phrasings).

Numbers are intentionally NOT discovered here — they are handled by the regex
extractors in the extract node (see segment_types module docstring).
"""

from __future__ import annotations

import re

from patterns.library import SpanMatch
from patterns.segment_types import _AMENITIES, _SUBCATEGORY_HINTS
from taxonomy.loader import TaxonomyIndex, lookup_with_prefixes, match_in_set, strip_prefixes

_WORD_RE = re.compile(r"\S+")
_MAX_CITY_WORDS = 3        # longest multi-word city (e.g. "ראשון לציון")
_MAX_PROPERTY_WORDS = 4   # longest multi-word property type (e.g. "דירה בבניין לשימור")

# Slang condition contractions not stored as canonical taxonomy values. The
# canonical multi-word conditions ("כמו חדש", "דורש שיפוץ") are derived from the
# taxonomy at lookup time; these are the variants users type that it doesn't list.
_CONDITION_SLANG = {"כחדש", "לחלקים"}

# Single-word taxonomy lookups: (TaxonomyIndex attribute, segment type).
# property_type is intentionally excluded — it uses multi-word window matching below
# so that "דירת סטודיו" is matched as a 2-word unit rather than "דירת" alone.
_LOOKUP_SPECS: list[tuple[str, str]] = [
    ("re_transaction_modes", "transaction_mode"),
    ("vehicle_manufacturers", "manufacturer"),
    ("vehicle_fuel_types", "fuel_type"),
    ("vehicle_gearbox_types", "gearbox"),
    ("vehicle_colors", "color"),
    ("sh_sectors", "sector"),
    ("sh_subcategories", "subcategory"),
    ("re_condition_values", "condition"),
    ("sh_conditions", "condition"),
]


def lookup_spans(clean_q: str, tax: TaxonomyIndex) -> list[SpanMatch]:
    """Return non-numeric candidate spans found by taxonomy lookup."""
    words = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(clean_q)]
    spans: list[SpanMatch] = []

    for start, end, word in words:
        for attr, seg_type in _LOOKUP_SPECS:
            if lookup_with_prefixes(word, getattr(tax, attr)):
                spans.append(SpanMatch(start, end, {seg_type}))

        for variant in strip_prefixes(word):
            if variant in _AMENITIES:
                spans.append(SpanMatch(start, end, {"amenity"}))
                break

        if not word.isdigit():   # numeric model names (Mazda "3") are too ambiguous
            for variant in strip_prefixes(word):
                if variant in tax.model_to_manufacturer:
                    spans.append(SpanMatch(start, end, {"model"}))
                    break

        if word in _SUBCATEGORY_HINTS:
            spans.append(SpanMatch(start, end, {"subcategory"}))

    # Multi-word condition phrases — must be matched as a unit before the single-word
    # loop commits "חדש" alone (which would give the wrong canonical value).
    # Canonical phrases come from the taxonomy itself (e.g. "כמו חדש", "דורש שיפוץ");
    # _CONDITION_SLANG holds contractions that are not stored as canonical values.
    condition_phrases = {
        c for c in (tax.sh_conditions | tax.re_condition_values) if " " in c
    } | _CONDITION_SLANG
    for phrase in condition_phrases:
        idx = clean_q.find(phrase)
        while idx != -1:
            end = idx + len(phrase)
            left_ok  = idx == 0 or clean_q[idx - 1] == " "
            right_ok = end == len(clean_q) or clean_q[end] == " "
            if left_ok and right_ok:
                spans.append(SpanMatch(idx, end, {"condition"}))
            idx = clean_q.find(phrase, idx + 1)

    # Multi-word property types: try 1..N word windows. whole=True so the window
    # must BE a property type, not merely contain one. This allows:
    #   "דירה"         (1 word, exact) → "דירה"
    #   "דירת"         (1 word, inflect ה↔ת via step 1.5) → "דירה"
    #   "דירת סטודיו"  (2 words, exact) → "דירת סטודיו"   (wins over 1-word "דירה")
    # Longer windows shadow shorter overlapping ones via merge_spans priority.
    for i in range(len(words)):
        for j in range(i, min(i + _MAX_PROPERTY_WORDS, len(words))):
            start, end = words[i][0], words[j][1]
            if match_in_set(clean_q[start:end], tax.re_property_types, whole=True):
                spans.append(SpanMatch(start, end, {"property_type"}))

    # Multi-word cities: try 1..N word windows against the city set. whole=True so
    # the window must BE a city (modulo prefix + stored suffix, "תל אביב" →
    # "תל אביב-יפו"), not merely contain one — otherwise "פנטהאוז ברמת גן" would
    # match the city "רמת גן" and swallow the property type into the span.
    for i in range(len(words)):
        for j in range(i, min(i + _MAX_CITY_WORDS, len(words))):
            start, end = words[i][0], words[j][1]
            if match_in_set(clean_q[start:end], tax.re_cities, whole=True):
                spans.append(SpanMatch(start, end, {"city"}))

    return spans
