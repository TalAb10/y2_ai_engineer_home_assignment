"""Segment-type registry — the single source of truth for what a segment means.

Scope: **non-numeric** concepts only — enums, names, and free text (manufacturer,
model, city, color, condition, …). Numbers (price, year, km, rooms, area, floor,
storage) are handled separately by the deterministic regexes in patterns/numbers.py,
never as learned patterns. That separation is deliberate: abstracting digits to "?"
would collapse "עד 9000" (price) and "עד 2018" (year) into the same shape. Keeping
numbers on the regex path — where years are range-restricted to 1980–2025 and read
only in a vehicle context — removes the ambiguity at the source.

Every type is one entry: name → (vertical, value-extractor). The same dict drives
the LLM's allowed labels (SEGMENT_TYPE_NAMES), category inference (vertical votes),
and value extraction. `vertical` is the category a type votes for, or None for
shared types (color, model, condition) that do not imply a vertical on their own.

Value extractors return a partial params dict keyed by Pydantic *field name*
(e.g. עיר). With populate_by_name=True the validate node accepts field names and
emits the proper taxonomy aliases. An empty dict means "no value", which is also
how a candidate type fails validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from taxonomy.loader import (
    TaxonomyIndex, match_in_set, score_query_for_category, strip_prefixes,
)

# Category constants (same literals used across the codebase).
CAT_RE = "נדל״ן"
CAT_VEHICLE = "רכב"
CAT_SH = "יד_שנייה"

# Keyword → (sector, subcategory) hints for common items/slang not in the taxonomy.
# Supplements the taxonomy term index; used here and by the span discoverer.
_SUBCATEGORY_HINTS: dict[str, tuple[str, str]] = {
    "אייפון": ("אלקטרוניקה", "טלפונים_סלולריים"),
    "אפל":    ("אלקטרוניקה", "טלפונים_סלולריים"),
    "סמסונג": ("אלקטרוניקה", "טלפונים_סלולריים"),
    "גלקסי":  ("אלקטרוניקה", "טלפונים_סלולריים"),
    "טלפון":  ("אלקטרוניקה", "טלפונים_סלולריים"),
    "מחשב":   ("אלקטרוניקה", "מחשבים_ניידים"),
    "לפטופ":  ("אלקטרוניקה", "מחשבים_ניידים"),
    "מקבוק":  ("אלקטרוניקה", "מחשבים_ניידים"),
    "טלוויזיה": ("אלקטרוניקה", "טלוויזיות"),
    "ספה":    ("ריהוט", "סלון"),
    "ארון":   ("ריהוט", "חדר_שינה"),
    "מיטה":   ("ריהוט", "חדר_שינה"),
    "אופניים": ("ספורט_וקמפינג", "אופניים"),
    "גיטרה":  ("מוסיקה_וכלים", "גיטרות"),
    "פסנתר":  ("מוסיקה_וכלים", "קלידים"),
    "עגלה":   ("לתינוקות_וסופגנים", "עגלות"),
}

# Params whose schema type is a list — merged by union instead of first-wins.
LIST_KEYS = {"סוגי_נכס", "מצבי_עסקה"}


# ── Value extractors (one per concept; all reuse taxonomy lookups) ─────────────

def _property_type(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    hit = match_in_set(text, tax.re_property_types)
    return {"סוגי_נכס": [hit]} if hit else {}


def _transaction_mode(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    hit = match_in_set(text, tax.re_transaction_modes)
    return {"מצבי_עסקה": [hit]} if hit else {}


def _city(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    # match_in_set handles clitic prefixes ("בתל אביב") and multi-word canonical
    # cities stored with a suffix ("תל אביב" → "תל אביב-יפו").
    # If no taxonomy match is found, return the raw text unchanged — stripping
    # without a confirmed match could silently corrupt the city name.
    hit = match_in_set(text, tax.re_cities)
    return {"עיר": hit or text.strip()}


_AMENITIES: dict[str, str] = {
    "מעלית": "מעלית", "חניה": "חניה", "מחסן": "מחסן", "מיזוג": "מיזוג",
    "ממ״ד": "מממ_ד", "ממד": "מממ_ד", "מרפסת": "מרפסת_שמש",
    "נגיש": "גישה_לנכים", "כלבים": "חיות_מחמד", "חיות": "חיות_מחמד",
}


def _amenity(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for word in text.split():
        for variant in strip_prefixes(word):
            field_name = _AMENITIES.get(variant)
            if field_name:
                out[field_name] = True
                break
    return out


def _manufacturer(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    hit = match_in_set(text, tax.vehicle_manufacturers)
    return {"יצרן": hit} if hit else {}


def _is_bare_number(text: str) -> bool:
    """True if text is only digits and numeric punctuation (e.g. "3000", "2015-2019")."""
    stripped = text.replace(".", "").replace(",", "").replace("-", "").strip()
    return bool(stripped) and stripped.isdigit()


def _model(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    # Known car model → also backfill its manufacturer; otherwise free-text model.
    for word in text.split():
        if word.isdigit():
            continue
        for variant in strip_prefixes(word):
            manufacturer = tax.model_to_manufacturer.get(variant)
            if manufacturer:
                return {"דגם": variant, "יצרן": manufacturer}
    cleaned = text.strip()
    # Invariant: a model name is never a bare number. The prompt already tells the
    # LLM not to label standalone numbers, but we enforce it here too because this
    # is the boundary that defines the model field — and because a numeric "model"
    # would otherwise be learned by the PatternLibrary as the shape "????", poisoning
    # every future query that contains a 4-digit number. This guards both outputs.
    if not cleaned or _is_bare_number(cleaned):
        return {}
    return {"דגם": cleaned}


def _fuel(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    hit = match_in_set(text, tax.vehicle_fuel_types)
    return {"סוג_דלק": hit} if hit else {}


def _gearbox(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    hit = match_in_set(text, tax.vehicle_gearbox_types)
    return {"תיבת_הילוכים": hit} if hit else {}


def _color(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    hit = match_in_set(text, tax.vehicle_colors)
    return {"צבע": hit} if hit else {}


def _sector_or_subcategory(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    """Shared by the sector and subcategory types — hints fill both at once."""
    for word in text.split():
        hint = _SUBCATEGORY_HINTS.get(word)
        if hint:
            return {"סקטור": hint[0], "תת_קטגוריה": hint[1]}
    out: dict[str, Any] = {}
    sector = match_in_set(text, tax.sh_sectors)
    if sector:
        out["סקטור"] = sector
    subcat = match_in_set(text, tax.sh_subcategories)
    if subcat:
        out["תת_קטגוריה"] = subcat
    return out


def _brand(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    cleaned = text.strip()
    return {"מותג": cleaned} if cleaned else {}


def _condition(text: str, tax: TaxonomyIndex, cat: str) -> dict[str, Any]:
    """Condition maps to a different field per vertical — key is category-scoped."""
    if "כמו חדש" in text or "כחדש" in text:
        return {"מצב": "כמו חדש"}
    re_hit = match_in_set(text, tax.re_condition_values)
    sh_hit = match_in_set(text, tax.sh_conditions)
    if cat == CAT_RE:
        if re_hit:
            return {"מצב_נכס": re_hit}
        return {"מצב": sh_hit} if sh_hit else {}
    # second-hand context, or validation (cat="") — prefer the second-hand field
    if sh_hit:
        return {"מצב": sh_hit}
    return {"מצב_נכס": re_hit} if re_hit else {}


# ── Registry ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentType:
    name: str
    vertical: str | None
    extract: Callable[[str, TaxonomyIndex, str], dict[str, Any]]


REGISTRY: dict[str, SegmentType] = {
    seg.name: seg for seg in [
        # real estate
        SegmentType("property_type", CAT_RE, _property_type),
        SegmentType("transaction_mode", CAT_RE, _transaction_mode),
        SegmentType("city", CAT_RE, _city),
        SegmentType("amenity", CAT_RE, _amenity),
        # vehicles
        SegmentType("manufacturer", CAT_VEHICLE, _manufacturer),
        SegmentType("fuel_type", CAT_VEHICLE, _fuel),
        SegmentType("gearbox", CAT_VEHICLE, _gearbox),
        # second-hand
        SegmentType("sector", CAT_SH, _sector_or_subcategory),
        SegmentType("subcategory", CAT_SH, _sector_or_subcategory),
        SegmentType("brand", CAT_SH, _brand),
        # shared (no vertical vote)
        SegmentType("model", None, _model),
        SegmentType("color", None, _color),
        SegmentType("condition", None, _condition),
    ]
}

SEGMENT_TYPE_NAMES: list[str] = sorted(REGISTRY)


# ── Resolution helpers (used by the extract node) ──────────────────────────────

def vertical_of(seg_type: str) -> str | None:
    entry = REGISTRY.get(seg_type)
    return entry.vertical if entry else None


def extract_value(seg_type: str, text: str, tax: TaxonomyIndex, category: str) -> dict[str, Any]:
    entry = REGISTRY.get(seg_type)
    return entry.extract(text, tax, category) if entry else {}


def valid_types(text: str, candidates: set[str], tax: TaxonomyIndex) -> set[str]:
    """Keep only candidate types whose extractor actually yields a value."""
    return {t for t in candidates if extract_value(t, text, tax, "")}


def resolve_type(text: str, valid: set[str], category: str) -> str | None:
    """Pick one type for a span from its validated candidates.

    Prefer a type matching the inferred category, then a shared (None) type.
    """
    if not valid:
        return None
    if len(valid) == 1:
        return next(iter(valid))
    for t in sorted(valid):
        if vertical_of(t) == category:
            return t
    for t in sorted(valid):
        if vertical_of(t) is None:
            return t
    return sorted(valid)[0]


def infer_category(verticals: list[str | None], tax: TaxonomyIndex,
                   query_words: list[str]) -> tuple[str, float]:
    """Infer the vertical from votes, falling back to taxonomy keyword scoring.

    Returns (category, certainty) where certainty = top_votes / (top_votes +
    second_votes), a value in [0, 1] measuring how dominant the winning vertical
    is over the runner-up. One uncontested category → 1.0 regardless of how many
    words matched; a near-tie → ~0.5. Feeds the confidence formula in validate.
    """
    votes: dict[str, float] = {}
    for vertical in verticals:
        if vertical:
            votes[vertical] = votes.get(vertical, 0.0) + 1.0

    if not votes:
        votes = score_query_for_category(query_words, tax)
    if not votes:
        # Zero signal — no typed segment and no taxonomy keyword matched. Return an
        # explicit "unknown" rather than guessing a vertical; the validate node then
        # produces empty params with an "unknown category" note. Guessing a default
        # here is what made gibberish / empty / English queries look like real hits.
        return "", 0.0

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top_cat, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    # Dominance ratio in [0, 1]. second_score is 0 for an uncontested winner → 1.0.
    # top_score is always > 0 here (we returned early when there were no votes).
    certainty = top_score / (top_score + second_score)
    return top_cat, certainty


def merge_params(params: dict[str, Any], new: dict[str, Any]) -> None:
    """Merge `new` into `params` in place. List keys union; scalars first-wins."""
    for key, value in new.items():
        if key in LIST_KEYS:
            bucket = params.setdefault(key, [])
            for item in (value if isinstance(value, list) else [value]):
                if item not in bucket:
                    bucket.append(item)
        else:
            params.setdefault(key, value)


# Second-hand fields that only make sense for a specific subcategory (field names).
# Anything not listed is universal (price, brand, condition, color, …) — never dropped.
_SUBCATEGORY_SPEC_FIELDS: dict[str, set[str]] = {
    "טלפונים_סלולריים": {"נפח_אחסון"},
    "מחשבים_ניידים": {"מעבד", "זיכרון_RAM", "אחסון_GB"},
    "טלוויזיות": {"גודל_אינצ", "טכנולוגיה", "רזולוציה"},
    "אופניים": {"גודל_גלגל", "סוג"},
}
_ALL_SPEC_FIELDS: set[str] = set().union(*_SUBCATEGORY_SPEC_FIELDS.values())


def filter_by_subcategory(params: dict[str, Any], category: str) -> list[str]:
    """Drop second-hand spec fields that don't belong to the detected subcategory.

    Prevents nonsense like a stroller carrying נפח_אחסון (a phone-only field).
    In-place; returns the dropped field names. No-op outside יד_שנייה or when the
    subcategory is unknown (we don't drop on a guess).
    """
    if category != CAT_SH:
        return []
    subcategory = params.get("תת_קטגוריה")
    if not subcategory:
        return []
    allowed = _SUBCATEGORY_SPEC_FIELDS.get(subcategory, set())
    dropped = [f for f in params if f in _ALL_SPEC_FIELDS and f not in allowed]
    for field_name in dropped:
        del params[field_name]
    return dropped
