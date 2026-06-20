"""Unit tests for the deterministic numeric extractors (patterns/numbers.py).

Direct function-level tests — the category-scoped wiring and price/year
disambiguation through the full graph is covered in test_pipeline.py.
"""

from __future__ import annotations

from patterns.numbers import (
    _looks_like_year, extract_area, extract_floor, extract_km, extract_price, extract_rooms,
)


# ── extract_price ────────────────────────────────────────────────────────────────

def test_price_currency_and_operator_cues():
    assert extract_price("עד 5000 שח") == {"max": 5000}
    assert extract_price("מעל 5000 שח") == {"min": 5000}
    assert extract_price("5000 שח") == {"max": 5000}        # currency, no operator → max


def test_price_bare_number_is_none():
    # No currency and no price operator → not a price (e.g. a model number).
    assert extract_price("מאזדה 3") is None
    assert extract_price("256") is None


def test_price_range_disambiguation():
    assert extract_price("בין 1000 ל 2000 שח") == {"min": 1000, "max": 2000}  # currency → price
    assert extract_price("בין 500000 ל 800000") == {"min": 500000, "max": 800000}  # non-year → price
    assert extract_price("בין 2015 ל 2018") is None        # year span, no currency → not a price
    assert extract_price("בין 3 ל 5 חדרים") is None        # bound to a unit → not a price


def test_year_looking_number_without_currency_is_not_a_price():
    assert extract_price("עד 2018") is None                # year, not price
    assert extract_price("עד 9000") == {"max": 9000}       # 9000 is not a year → price


# ── _looks_like_year (the price/year disambiguation window) ──────────────────────

def test_looks_like_year_window():
    assert _looks_like_year(2018)
    assert _looks_like_year(2025)
    assert _looks_like_year(2029)
    assert not _looks_like_year(1979)
    assert not _looks_like_year(2030)
    assert not _looks_like_year(9000)


# ── other numeric fields ─────────────────────────────────────────────────────────

def test_extract_rooms():
    assert extract_rooms("3 חדרים") == 3.0
    assert extract_rooms("3.5 חדרים") == 3.5
    assert extract_rooms("דירה גדולה") is None


def test_extract_area_operators():
    assert extract_area("עד 100 מ״ר") == {"max": 100}
    assert extract_area("מעל 80 מ״ר") == {"min": 80}
    assert extract_area("90 מ״ר") == {"min": 90, "max": 90}


def test_extract_km():
    assert extract_km("עד 80000 ק״מ") == {"max": 80000}
    assert extract_km("מ-50000 ק״מ") == {"min": 50000}


def test_extract_floor():
    assert extract_floor("קומה 5") == 5
    assert extract_floor("דירה") is None
