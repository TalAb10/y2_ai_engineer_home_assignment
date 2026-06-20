"""Taxonomy loader — builds in-memory indexes from yad2_search_taxonomy.json.

This is the single source of truth consumed by:
  - normalize node      (typo correction)
  - span discovery      (enum lookup, model→manufacturer reverse index)
  - extract node        (category inference via keyword scoring)

Hebrew morphology note: Hebrew clitics (ב/ל/מ/ה/ו/ש/כ) attach to the front of words.
strip_prefixes() tries to normalise them before dictionary lookups so that e.g.
"בירושלים" matches "ירושלים" and "להשכרה" matches "השכרה".
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Hebrew clitic prefix stripping ────────────────────────────────────────────
# Ordered longest-first so we don't strip a single char that's part of a longer prefix.
# Shared with the normalize node so typo correction and lookup agree on prefixes.
CLITIC_PREFIXES = ["כש", "מה", "שה", "בה", "לה", "ב", "ל", "מ", "ה", "ו", "ש", "כ"]


def strip_prefixes(word: str) -> list[str]:
    """Return the original word plus all de-prefixed variants (longest-first)."""
    variants = [word]
    for prefix in CLITIC_PREFIXES:
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            variants.append(word[len(prefix):])
    return variants


def _normalise_term(term: str) -> str:
    """NFKC + strip punctuation noise around the term (not inside)."""
    return unicodedata.normalize("NFKC", term).strip()


def _flatten_strings(obj: Any) -> list[str]:
    """Recursively collect every string value inside a nested JSON structure."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        return [s for item in obj for s in _flatten_strings(item)]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _flatten_strings(v)]
    return []


# ── Main index dataclass ───────────────────────────────────────────────────────

@dataclass
class TaxonomyIndex:
    """All lookup structures derived from the taxonomy JSON."""

    # Classification scoring: canonical_term → category name
    term_to_category: dict[str, str] = field(default_factory=dict)

    # Per-category keyword sets (used for category inference scoring)
    category_terms: dict[str, set[str]] = field(default_factory=dict)

    # Typo maps: misspelling → canonical form (merged from all verticals + global)
    typo_map: dict[str, str] = field(default_factory=dict)

    # Real-estate specific
    re_property_types: set[str] = field(default_factory=set)
    re_transaction_modes: set[str] = field(default_factory=set)
    re_cities: set[str] = field(default_factory=set)
    re_amenities: set[str] = field(default_factory=set)   # boolean fields
    re_condition_values: set[str] = field(default_factory=set)

    # Vehicles specific
    vehicle_manufacturers: set[str] = field(default_factory=set)
    vehicle_models: set[str] = field(default_factory=set)
    model_to_manufacturer: dict[str, str] = field(default_factory=dict)
    vehicle_fuel_types: set[str] = field(default_factory=set)
    vehicle_gearbox_types: set[str] = field(default_factory=set)
    vehicle_colors: set[str] = field(default_factory=set)

    # Second-hand specific
    sh_sectors: set[str] = field(default_factory=set)
    sh_subcategories: set[str] = field(default_factory=set)
    sh_conditions: set[str] = field(default_factory=set)

    # Raw taxonomy (kept for LLM prompt building and schema derivation)
    raw: dict[str, Any] = field(default_factory=dict)


def load(taxonomy_path: Path) -> TaxonomyIndex:
    """Parse the taxonomy JSON and return a fully-populated TaxonomyIndex."""
    with open(taxonomy_path, encoding="utf-8") as taxonomy_file:
        data: dict[str, Any] = json.load(taxonomy_file)

    index = TaxonomyIndex(raw=data)
    categories: dict[str, Any] = data.get("קטגוריות", {})

    _load_realestate(index, categories.get("נדל״ן", {}))
    _load_vehicles(index, categories.get("רכב", {}))
    _load_secondhand(index, categories.get("יד_שנייה", {}))
    _load_typos(index, categories)

    return index


# ── Per-vertical loaders ───────────────────────────────────────────────────────

def _register(index: TaxonomyIndex, terms: list[str], category: str) -> None:
    """Add terms to the category index and to the classification lookup."""
    if category not in index.category_terms:
        index.category_terms[category] = set()
    for term_text in terms:
        normalised_term = _normalise_term(term_text)
        index.category_terms[category].add(normalised_term)
        # Only store the first mapping (taxonomy terms are unambiguous)
        index.term_to_category.setdefault(normalised_term, category)


def _load_realestate(index: TaxonomyIndex, re_data: dict[str, Any]) -> None:
    category = "נדל״ן"

    prop_types: list[str] = re_data.get("סוגי_נכס", [])
    index.re_property_types = {_normalise_term(t) for t in prop_types}
    _register(index, prop_types, category)

    tx_modes: list[str] = re_data.get("מצבי_עסקה", [])
    index.re_transaction_modes = {_normalise_term(t) for t in tx_modes}
    _register(index, tx_modes, category)

    general: dict[str, Any] = re_data.get("מאפיינים_כלליים", {})

    # Cities from examples list
    city_examples: list[str] = general.get("עיר", {}).get("דוגמאות", [])
    index.re_cities = {_normalise_term(c) for c in city_examples}
    _register(index, city_examples, category)

    # Boolean amenity field names (used by extractor keyword matching)
    boolean_fields = {"מעלית", "מחסן", "מיזוג", "ממ״ד", "גישה_לנכים", "מרפסת_שמש", "חיות_מחמד"}
    index.re_amenities = boolean_fields

    # Condition values
    condition_vals: list[str] = general.get("מצב_נכס", [])
    index.re_condition_values = {_normalise_term(v) for v in condition_vals}
    _register(index, condition_vals, category)

    # Proximity terms, furnishing, ownership — register for classification
    for key in ("קרבה", "ריהוט", "בעלות", "כיווני_אוויר", "בעלות_מקרקעין"):
        _register(index, general.get(key, []), category)


def _load_vehicles(index: TaxonomyIndex, v_data: dict[str, Any]) -> None:
    category = "רכב"

    vehicle_types: list[str] = v_data.get("סוגי_רכב", [])
    _register(index, vehicle_types, category)

    manufacturers: dict[str, Any] = v_data.get("יצרנים", {})
    for manufacturer_name, manufacturer_data in manufacturers.items():
        manufacturer = _normalise_term(manufacturer_name)
        index.vehicle_manufacturers.add(manufacturer)
        _register(index, [manufacturer_name], category)

        models: list[str] = manufacturer_data.get("דגמים", [])
        for raw_model in models:
            model = _normalise_term(raw_model)
            index.vehicle_models.add(model)
            index.model_to_manufacturer[model] = manufacturer
        # Skip purely numeric model names (e.g. Mazda "2","3","6") from the
        # classification keyword index — single digits are too ambiguous and
        # would spuriously score vehicles in any query containing a number.
        _register(index, [m for m in models if not _normalise_term(m).isdigit()], category)

        # Sub-models
        for sub_models in manufacturer_data.get("תתי_דגמים", {}).values():
            _register(index, sub_models, category)

    general: dict[str, Any] = v_data.get("מאפיינים_כלליים", {})

    fuel_types: list[str] = general.get("סוג_דלק", [])
    index.vehicle_fuel_types = {_normalise_term(t) for t in fuel_types}
    _register(index, fuel_types, category)

    gearbox_types: list[str] = general.get("תיבת_הילוכים", [])
    index.vehicle_gearbox_types = {_normalise_term(t) for t in gearbox_types}
    _register(index, gearbox_types, category)

    colors: list[str] = general.get("צבע", [])
    index.vehicle_colors = {_normalise_term(c) for c in colors}
    _register(index, colors, category)

    for key in ("בעלות", "מערכות_בטיחות", "אבזור"):
        _register(index, general.get(key, []), category)


def _load_secondhand(index: TaxonomyIndex, sh_data: dict[str, Any]) -> None:
    category = "יד_שנייה"

    sectors: dict[str, Any] = sh_data.get("סקטורים", {})
    for sector_name, sector_data in sectors.items():
        sector = _normalise_term(sector_name)
        index.sh_sectors.add(sector)
        _register(index, [sector_name], category)

        subcats: dict[str, Any] = sector_data.get("תתי_קטגוריות", {})
        for subcategory_name, subcategory_data in subcats.items():
            subcategory = _normalise_term(subcategory_name)
            index.sh_subcategories.add(subcategory)
            _register(index, [subcategory_name], category)

            # Brands, conditions, specs within each sub-category
            for key in ("מותגים", "מצב", "סוג", "טכנולוגיה", "רזולוציה", "סוגי_רהיט", "חומר"):
                _register(index, subcategory_data.get(key, []), category)

    general: dict[str, Any] = sh_data.get("מאפיינים_כלליים", {})
    conditions: list[str] = general.get("מצב", [])
    index.sh_conditions = {_normalise_term(c) for c in conditions}
    _register(index, conditions, category)
    _register(index, general.get("אזור", []), category)


def _load_typos(index: TaxonomyIndex, categories: dict[str, Any]) -> None:
    """Merge per-category typo maps into a single flat dict."""
    for category_data in categories.values():
        for wrong, correct in category_data.get("מיפוי_מילות_שגיאה", {}).items():
            index.typo_map[_normalise_term(wrong)] = _normalise_term(correct)


# ── Lookup helpers (used by normalise node and extractors) ─────────────────────

def lookup_with_prefixes(word: str, lookup_set: set[str]) -> str | None:
    """Try to find `word` in `lookup_set`, trying prefix-stripped variants if needed."""
    for variant in strip_prefixes(_normalise_term(word)):
        if variant in lookup_set:
            return variant
    return None


# ── Phrase-aware, inflection-tolerant matching ────────────────────────────────
# lookup_with_prefixes only matches a single token exactly. Real queries need two
# more things: multi-word canonical values ("דירת גן", "תל אביב-יפו") and Hebrew
# inflection ("חדשה"→"חדש", "היברידית"→"היברידי"). match_in_set() handles both and
# is the matcher every enum extractor should use.

def _inflect_equiv(a: str, b: str) -> bool:
    """True if a and b are the same word up to a short Hebrew inflectional suffix.

    Hebrew inflects gender/number with trailing ה/ת/ים/ות/ית. We treat one word as
    a match for another when the shorter is a prefix of the longer and they differ
    by at most two trailing characters — enough for "חדש"/"חדשה", "היברידי"/"היברידית"
    without collapsing unrelated short words.

    Also handles the Hebrew construct state (סמיכות): feminine nouns drop their
    final ה and add ת when preceding a noun ("דירה" → "דירת", "שכונה" → "שכונת").
    These are the same word in different grammatical roles and must be treated as
    equivalent for taxonomy lookup.
    """
    if a == b:
        return True
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    if len(shorter) >= 3 and longer.startswith(shorter) and len(longer) - len(shorter) <= 2:
        return True
    # Construct state: same length, differ only in final ה ↔ ת
    if len(a) == len(b) >= 3 and a[:-1] == b[:-1] and {a[-1], b[-1]} == {"ה", "ת"}:
        return True
    return False


def _word_equiv(a: str, b: str) -> bool:
    """Word equality tolerant of both clitic prefixes and inflectional suffixes."""
    return any(_inflect_equiv(av, bv)
               for av in strip_prefixes(a)
               for bv in strip_prefixes(b))


def _split_words(text: str) -> list[str]:
    """Split into words, treating '/' and '-' as separators.

    Canonical values pack alternates into one token: "בית פרטי/וילה" (house or villa),
    "תל אביב-יפו" (Tel Aviv, formally Tel Aviv-Yafo). Splitting on these lets a user
    who types just "בית פרטי" or "תל אביב" match the full canonical value.
    """
    return text.replace("/", " ").replace("-", " ").replace("־", " ").split()


def _is_run(needle: list[str], haystack: list[str], anchored: bool) -> bool:
    """True if `needle` appears as a consecutive run of words inside `haystack`.

    anchored=True requires the run to start at haystack[0] (used when the user typed
    a prefix of a longer canonical, e.g. "בית פרטי" of "בית פרטי/וילה").
    """
    if not needle or len(needle) > len(haystack):
        return False
    starts = [0] if anchored else range(len(haystack) - len(needle) + 1)
    return any(all(_word_equiv(needle[j], haystack[i + j]) for j in range(len(needle)))
               for i in starts)


def match_in_set(text: str, lookup_set: set[str], *, whole: bool = False) -> str | None:
    """Find the canonical value in `lookup_set` that best matches `text`.

    Resolution order, most-confident first:
      1. Whole phrase, exact or with a clitic prefix stripped.
      2. A canonical value appears fully inside the text (inflection-tolerant) —
         prefer the LONGEST such value, i.e. the most specific concept the user named.
      3. The text is a prefix of a longer canonical ("בית פרטי" → "בית פרטי/וילה",
         "תל אביב" → "תל אביב-יפו") — a fallback, preferring the shortest canonical.

    whole=True skips step 2, so the *entire* text must be the value (modulo prefix
    and the canonical's own suffix). Use it for span discovery — testing whether a
    word window IS a city, not whether a city is buried inside extra words. Without
    it, the window "פנטהאוז ברמת גן" would match the city "רמת גן".
    """
    norm = _normalise_term(text)

    def word_count(value: str) -> int:
        return len(_split_words(value))

    # 1. Whole phrase — exact, or with a clitic prefix stripped.
    #    Prefer the LONGEST matching variant (least stripping), so "בהרצליה"
    #    returns "הרצליה" (strip "ב") rather than "רצליה" (strip "בה") when both
    #    are in the set. This preserves proper nouns whose ה is part of the name.
    matches = [v for v in strip_prefixes(norm) if v in lookup_set]
    if matches:
        return max(matches, key=len)

    text_words = _split_words(norm)

    # 1.5. Single-word inflectional match: Hebrew construct state swaps final ה↔ת
    # ("דירת" matches "דירה"). Only applies when both text and candidate are one word,
    # so "דירת" never matches the multi-word "דירת גן" here — that requires step 2.
    if len(text_words) == 1:
        for candidate in lookup_set:
            cand_words = _split_words(candidate)
            if len(cand_words) == 1 and _inflect_equiv(text_words[0], cand_words[0]):
                return candidate

    # 2. A canonical value appears in full inside the text. Prefer the longest —
    #    "היברידי נטען" wins over "היברידי" when the user typed both words.
    if not whole:
        contained = [c for c in lookup_set if _is_run(_split_words(c), text_words, anchored=False)]
        if contained:
            return max(contained, key=word_count)

    # 3. The text is a prefix of a longer canonical, but ONLY when the canonical's
    #    extra characters start with a separator (/ or -), never a plain space.
    #    This allows "תל אביב" → "תל אביב-יפו" and "בית פרטי" → "בית פרטי/וילה"
    #    while blocking "דירת" → "דירת גן" (where "גן" is a semantic word, not a
    #    formal suffix — without it the phrase means something entirely different).
    def _extends_via_separator(user: str, canonical: str) -> bool:
        for variant in strip_prefixes(user):
            if canonical.startswith(variant):
                remainder = canonical[len(variant):]
                if remainder and remainder[0] in ('/', '-', '־'):
                    return True
        return False

    prefixes = [c for c in lookup_set
                if _is_run(text_words, _split_words(c), anchored=True)
                and _extends_via_separator(norm, c)]
    if prefixes:
        return min(prefixes, key=word_count)

    return None


def score_query_for_category(tokens: list[str], idx: TaxonomyIndex) -> dict[str, float]:
    """Return a {category: score} dict based on how many tokens match each category."""
    scores: dict[str, float] = {}
    for token in tokens:
        for variant in strip_prefixes(_normalise_term(token)):
            cat = idx.term_to_category.get(variant)
            if cat:
                scores[cat] = scores.get(cat, 0.0) + 1.0
                break  # count each token once
    return scores
