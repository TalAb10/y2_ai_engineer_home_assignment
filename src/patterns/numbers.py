"""Deterministic numeric extraction — regex over NFKC-normalised Hebrew text.

Numbers (price, rooms, area, floor, km, storage) are pulled here, never as learned
patterns: abstracting digits to "?" would collapse "עד 9000" (price) and "עד 2018"
(year) into the same shape. The normalize node has already turned "מיליון"→"1000000"
and "אלף"→"1000" before these patterns run.

Used by graph/nodes/extract.py. Each extractor returns a partial value
({"min":…, "max":…}, a float, or None) — never a full params dict.
"""

from __future__ import annotations

import re
from typing import Any

_PRICE_RE = re.compile(
    r"(?P<op>עד|מעל ל|מעל|לפחות|מ-|מ|ב)?\s*"
    r"(?P<num>\d[\d,\.]*)\s*"
    r"(?P<cur>שח)?",   # normalize node has already collapsed שח / ש״ח / ₪ / שקל → "שח"
    re.UNICODE,
)
# "ב" is a price anchor ("ב1000" = "for 1000") but also means "in" ("ב2018" = a year).
# A ב-number that looks like a year and has no currency is left for year extraction.
#
# _YEAR_HI is the price-disambiguation window only: the upper edge of "this number is
# probably a year, not a price." It is intentionally broader than the range of years we
# actually accept (see _YEAR_RE) so that a just-out-of-range year a user types — e.g.
# "עד 2026" — is still treated as a (rejected) year rather than misread as a ₪2026 price.
_YEAR_LO, _YEAR_HI = 1980, 2029
# If one of these units immediately follows a number, the number belongs to that
# field (km / area / rooms / storage) — not to price. "עד 100000 קמ" is mileage.
_NON_PRICE_UNIT_RE = re.compile(
    r"^\s*(?:ק[\"״']?מ|מ[\"״']?ר|מטר|חד|ג\S*יגה|gb)", re.IGNORECASE | re.UNICODE,
)
_PRICE_OPS_MIN = {"מעל", "מעל ל", "לפחות", "מ", "מ-"}
_RANGE_RE = re.compile(
    r"בין\s+(?P<lo>\d[\d,\.]*)\s+(?:ל-?|ל)\s*(?P<hi>\d[\d,\.]*)",
    re.UNICODE,
)
# A currency token immediately after a number proves it is a price ("…2000 שח").
_TRAILING_CURRENCY_RE = re.compile(r"^\s*שח", re.UNICODE)
_ROOMS_RE = re.compile(r"(?P<num>\d+(?:[.,]\d+)?)\s*(?:חד(?:רים?)?)", re.UNICODE)
_AREA_RE = re.compile(r"(?P<op>עד|מעל|מ-)?\s*(?P<num>\d+)\s*(?:מ[״']ר|מטר(?:\s*רבוע)?)", re.UNICODE)
# Accepted year values, bounded to the taxonomy's valid range (1980–2025). Anything
# above is not a year we recognise, so it is never extracted (no silent drop at
# validation). The schema (taxonomy.schemas) enforces the same 1980–2025 bound.
_YEAR_RE = re.compile(r"\b(19[8-9]\d|20[01]\d|202[0-5])\b")
_KM_RE = re.compile(r"(?P<op>עד|מ-)?\s*(?P<num>\d[\d,\.]*)\s*(?:ק[״']מ|קמ\b|ק\"מ)", re.UNICODE)
_FLOOR_RE = re.compile(r"קומה\s+(?P<num>\d+)", re.UNICODE)
# "256GB", "256 gb", "256 ג'יגה", "256 ג׳יגה" — any geresh/quote variant.
_STORAGE_RE = re.compile(r"(\d+)\s*(?:gb|ג\S*יגה)", re.IGNORECASE | re.UNICODE)


def _num(raw: str) -> float:
    return float(raw.replace(",", ""))


def _looks_like_year(num: float) -> bool:
    """A number in the model-year window carries no price meaning on its own.

    Used by both the range and single-number paths so they disambiguate
    year-vs-price identically: a year-looking number without a currency cue is a
    year, not a price (e.g. "עד 2018", "בין 2015 ל 2018").
    """
    return _YEAR_LO <= num <= _YEAR_HI


def extract_price(text: str) -> dict[str, Any] | None:
    """Price from explicit cues only — a bare number is never a price.

    A number is a price when it carries textual evidence:
      • a currency token right after it (שח / ש״ח / ₪ / שקל), or
      • a price operator right before it (עד / מעל / לפחות / מ / ב) that is NOT bound
        to another unit — so "עד 100000 קמ" stays mileage, "קומה 10" stays a floor.
    "ב" is a price anchor ("ב1000") except on a year-looking number ("ב2018").
    "בין X ל Y" is treated as a price range.
    """
    m = _RANGE_RE.search(text)
    if m:
        lo, hi = _num(m.group("lo")), _num(m.group("hi"))
        trailing = text[m.end():]
        has_currency = bool(_TRAILING_CURRENCY_RE.match(trailing))
        # Apply the same disambiguation the single-number path uses below: a range is
        # only a price if it carries a currency cue OR is neither a year span nor bound
        # to another unit. "בין 2015 ל 2018" (years) and "בין 3 ל 5 חדרים" (rooms) fall
        # through so they aren't reported as a price; their endpoints have no price cue
        # of their own, so the loop below won't pick them up either.
        followed_by_unit = bool(_NON_PRICE_UNIT_RE.match(trailing))
        is_year_span = _looks_like_year(lo) and _looks_like_year(hi)
        if has_currency or not (followed_by_unit or is_year_span):
            return {"min": lo, "max": hi}

    results: dict[str, float] = {}
    for m in _PRICE_RE.finditer(text):
        op = (m.group("op") or "").strip()
        has_currency = bool(m.group("cur"))
        followed_by_unit = bool(_NON_PRICE_UNIT_RE.match(text[m.end("num"):]))

        if followed_by_unit and not has_currency:
            continue                       # the number belongs to km / area / rooms
        if not has_currency and not op:
            continue                       # bare number, no cue → not a price

        num = _num(m.group("num"))
        if op in ("ב", "עד", "מעל", "מעל ל") and not has_currency and _looks_like_year(num):
            continue                       # "ב2018" / "עד 2018" → a year, not a price

        if op in _PRICE_OPS_MIN:
            results["min"] = num
        else:                              # "עד" / "ב", or currency with no operator
            results["max"] = num
    return results or None


def extract_rooms(text: str) -> float | None:
    m = _ROOMS_RE.search(text)
    return float(m.group("num").replace(",", ".")) if m else None


def extract_area(text: str) -> dict[str, Any] | None:
    m = _AREA_RE.search(text)
    if not m:
        return None
    num = float(m.group("num"))
    op = (m.group("op") or "").strip()
    if op == "עד":
        return {"max": num}
    if op in ("מעל", "מ-"):
        return {"min": num}
    return {"min": num, "max": num}


def extract_km(text: str) -> dict[str, Any] | None:
    m = _KM_RE.search(text)
    if not m:
        return None
    num = _num(m.group("num"))
    op = (m.group("op") or "").strip()
    if op in ("מ-", "מ"):
        return {"min": num}
    return {"max": num}


def extract_floor(text: str) -> int | None:
    m = _FLOOR_RE.search(text)
    return int(m.group("num")) if m else None
