"""Node: extract — pattern-first, LLM only for the gaps.

This node both classifies and extracts: the category falls out of the segment
types and the numeric fields found, so there is no separate classify node.

Two extraction tracks run side by side:
  • Numbers (price, year, km, rooms, area, floor, storage) → deterministic regex
    over the whole query, category-scoped. Always runs, offline-safe. Year is read
    only in a vehicle context. extract_price skips year-range values (1980–2029)
    without a currency cue, so "עד 2018" → שנה.max, "עד 9000" → מחיר.max.
  • Everything else (names, enums, city, slang) → typed *segments*, discovered
    deterministically from the taxonomy and from the learned PatternLibrary, then
    resolved to one type each. If too little of the query is covered, the LLM
    labels only the gaps; its new segments and clear typos are learned for next time.

The LLM never extracts values — it only labels meaning. Identical query shapes
with different numbers reuse learned patterns and skip the LLM entirely.
"""

from __future__ import annotations

import re
from typing import Any

from patterns.numbers import (
    _AREA_RE, _FLOOR_RE, _KM_RE, _PRICE_RE, _RANGE_RE, _ROOMS_RE, _STORAGE_RE, _YEAR_RE,
    extract_area, extract_floor, extract_km, extract_price, extract_rooms,
)
from graph.context import NodeContext
from graph.state import GraphState
from llm.prompts import SEGMENTATION_SCHEMA, SEGMENTATION_SYSTEM
from patterns import discover, library, segment_types
from patterns.library import Segment
from patterns.segment_types import CAT_RE, CAT_SH, CAT_VEHICLE



async def run(state: GraphState, ctx: NodeContext) -> dict:
    clean_q = state.clean_q
    lib = ctx.pattern_library
    tax = ctx.taxonomy

    # ── Non-numeric segments: discover + learned patterns → validated candidates ─
    candidates = discover.lookup_spans(clean_q, tax) + lib.scan(clean_q)
    validated = _validate(clean_q, library.merge_spans(candidates), tax)
    definite = [next(iter(types)) for _, types in validated if len(types) == 1]
    numeric_verticals = _numeric_verticals(clean_q)

    # Provisional category drives multi-type span resolution + the LLM gate.
    category, _ = segment_types.infer_category(
        [segment_types.vertical_of(t) for t in definite] + numeric_verticals,
        tax, state.query_words,
    )
    segments = _resolve(clean_q, validated, category)

    num_params, num_spans = _extract_numeric(clean_q, category)
    coverage = library.coverage(segments, clean_q, num_spans)

    notes: list[str] = []
    llm_used = False
    taxonomy_hints: dict[str, list[str]] = {}

    # ── LLM gap-fill (only when patterns + rules leave too much uncovered) ─────
    if coverage < ctx.settings.pattern_coverage_threshold and ctx.llm.is_available():
        gap_text = _gap_text(clean_q, segments, num_spans)
        annotated = library.annotate(clean_q, segments)

        # Semantic search: two queries merged — full query gives context,
        # gap text gives precision on what's specifically missing.
        # Results are passed to the LLM so it knows which canonical values exist.
        taxonomy_hints = await _semantic_hints(gap_text, clean_q, ctx)  # stored in state for debug

        parsed = await _llm_segment(
            full_query=clean_q,
            annotated=annotated,
            gap_text=gap_text,
            taxonomy_hints=taxonomy_hints,
            ctx=ctx,
        )
        llm_used = True
        if parsed:
            new_segments = _segments_from_llm(parsed.get("segments", []), clean_q, segments,
                                              taxonomy_hints)
            for seg in new_segments:
                # Keep an LLM segment only if its value-extractor resolves to a taxonomy
                # value — the same gate the deterministic path applies in _validate().
                # An unresolvable enum label (e.g. "במבצע" mislabelled transaction_mode)
                # would otherwise cast a category vote while contributing no value,
                # misclassifying the query. Free-text types (city/model/brand/condition)
                # always resolve.
                if not segment_types.valid_types(seg.text, {seg.type}, tax):
                    continue
                segments.append(seg)
                # Learn the raw surface form (not the canonical replacement) so the
                # library recognises the same user phrasing next time, skipping the LLM.
                # raw_text == text means it already passed the gate above; only when a
                # semantic hint changed the text do we re-validate the surface form, so a
                # plausible-but-wrong label is never learned.
                if seg.raw_text == seg.text or segment_types.valid_types(seg.raw_text, {seg.type}, tax):
                    lib.learn(seg.raw_text, seg.type)
            learned = _learn_normalizations(parsed.get("normalizations", []), ctx)
            if learned:
                notes.append("נלמדו נורמליזציות: " + ", ".join(learned))

    # ── Final category + value extraction ──────────────────────────────────────
    verticals = [segment_types.vertical_of(s.type) for s in segments] + numeric_verticals
    category, certainty = segment_types.infer_category(verticals, tax, state.query_words)
    num_params, num_spans = _extract_numeric(clean_q, category)

    params: dict[str, Any] = {}
    for seg in segments:
        segment_types.merge_params(params, segment_types.extract_value(seg.type, seg.text, tax, category))
    for key, value in num_params.items():
        segment_types.merge_params(params, {key: value})

    # Drop spec fields inconsistent with the detected subcategory (e.g. storage on a stroller).
    dropped = segment_types.filter_by_subcategory(params, category)
    if dropped:
        notes.append("שדות שאינם תואמים לתת-הקטגוריה הוסרו: " + ", ".join(dropped))

    return {
        "category": category,
        "classification_certainty": certainty,
        "params": params,
        "segments": [seg.to_dict() for seg in segments],
        "extraction_coverage": library.coverage(segments, clean_q, num_spans),
        "llm_used": llm_used,
        "taxonomy_hints": taxonomy_hints,
        "notes": state.notes + notes,
    }


# ── Segment resolution ──────────────────────────────────────────────────────────

def _validate(clean_q: str, merged, tax) -> list[tuple]:
    """Drop candidate spans/types whose value-extractor yields nothing."""
    out: list[tuple] = []
    for span in merged:
        types = segment_types.valid_types(clean_q[span.start:span.end], span.types, tax)
        if types:
            out.append((span, types))
    return out


def _resolve(clean_q: str, validated: list[tuple], category: str) -> list[Segment]:
    segments: list[Segment] = []
    for span, types in validated:
        text = clean_q[span.start:span.end]
        seg_type = segment_types.resolve_type(text, types, category)
        if seg_type:
            segments.append(Segment(text=text, type=seg_type,
                                    start=span.start, end=span.end, source="pattern"))
    return segments


# ── Numeric extraction (deterministic, category-scoped) ─────────────────────────

def _numeric_verticals(clean_q: str) -> list[str]:
    """Category votes from anchored numeric units present in the query."""
    verticals: list[str] = []
    if _ROOMS_RE.search(clean_q) or _AREA_RE.search(clean_q) or _FLOOR_RE.search(clean_q):
        verticals.append(CAT_RE)
    if _KM_RE.search(clean_q):
        verticals.append(CAT_VEHICLE)
    if _STORAGE_RE.search(clean_q):
        verticals.append(CAT_SH)
    return verticals


def _extract_numeric(clean_q: str, category: str) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    """Extract numeric fields for the category. Returns (params, covered_spans)."""
    params: dict[str, Any] = {}
    spans: list[tuple[int, int]] = []

    # extract_price now requires a price cue, so a bare number (e.g. "מאזדה 3") is
    # never a price — no extra guard needed here.
    price = extract_price(clean_q)
    if price:
        params["מחיר"] = price
        spans += _regex_spans(_RANGE_RE, clean_q) + _regex_spans(_PRICE_RE, clean_q)

    if category == CAT_RE:
        rooms = extract_rooms(clean_q)
        if rooms is not None:
            params["מס_חדרים"] = rooms
            spans += _regex_spans(_ROOMS_RE, clean_q)
        area = extract_area(clean_q)
        if area:
            params["מ_ר_בנוי"] = area
            spans += _regex_spans(_AREA_RE, clean_q)
        floor = extract_floor(clean_q)
        if floor is not None:
            params["קומה"] = floor
            spans += _regex_spans(_FLOOR_RE, clean_q)

    elif category == CAT_VEHICLE:
        km = extract_km(clean_q)
        if km:
            params["ק_מ"] = km
            spans += _regex_spans(_KM_RE, clean_q)
        year, year_spans = _extract_year(clean_q)
        if year:
            params["שנה"] = year
            spans += year_spans

    elif category == CAT_SH:
        match = _STORAGE_RE.search(clean_q)
        if match:
            params["נפח_אחסון"] = f"{match.group(1)}GB"
            spans.append((match.start(), match.end()))

    return params, spans


def _extract_year(clean_q: str) -> tuple[dict[str, int] | None, list[tuple[int, int]]]:
    """Accepted years (1980–2025, per _YEAR_RE) in a vehicle query — extract_price already skips year-range values."""
    years: list[int] = []
    spans: list[tuple[int, int]] = []
    for match in _YEAR_RE.finditer(clean_q):
        years.append(int(match.group()))
        spans.append((match.start(), match.end()))
    if not years:
        return None, []
    return {"min": min(years), "max": max(years)}, spans


def _regex_spans(regex: re.Pattern, text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in regex.finditer(text) if m.group().strip()]


# ── Semantic taxonomy hints ──────────────────────────────────────────────────────

def _gap_text(clean_q: str, segments: list[Segment], num_spans: list[tuple[int, int]]) -> str:
    """Return the uncovered words from the query — what the LLM needs to label."""
    covered = library._covered_chars(segments, len(clean_q), num_spans)
    return " ".join(
        m.group() for m in re.finditer(r"\S+", clean_q)
        if library.is_salient_token(m.group())
        and not all(i in covered for i in range(m.start(), m.end()))
    )


async def _semantic_hints(gap_text: str, clean_q: str, ctx: NodeContext) -> dict[str, list[str]]:
    """Two-query semantic search: full query for context, gap text for precision.

    Both embeddings are searched against the taxonomy index. Gap results come first
    (higher precision for what's missing); full-query results fill in context-aware
    suggestions for types not already covered by the gap search.
    """
    if ctx.semantic_index is None or not gap_text.strip():
        return {}

    # Run both embeddings concurrently.
    import asyncio as _asyncio
    gap_emb, ctx_emb = await _asyncio.gather(
        ctx.llm.embed(gap_text),
        ctx.llm.embed(clean_q),
    )
    hints: dict[str, list[str]] = {}

    # Gap search first — precision for what's specifically missing.
    if gap_emb:
        for m in ctx.semantic_index.search(gap_emb, field_type=None, k=8, threshold=0.4):
            if m.value not in hints.get(m.field_type, []):
                hints.setdefault(m.field_type, []).append(m.value)

    # Full-query search — context for types the gap search may have missed.
    if ctx_emb:
        for m in ctx.semantic_index.search(ctx_emb, field_type=None, k=8, threshold=0.45):
            if m.value not in hints.get(m.field_type, []):
                hints.setdefault(m.field_type, []).append(m.value)

    return hints


# ── LLM gap-fill ────────────────────────────────────────────────────────────────

async def _llm_segment(
    full_query: str,
    annotated: str,
    gap_text: str,
    taxonomy_hints: dict[str, list[str]],
    ctx: NodeContext,
) -> dict | None:
    """Segmentation call — LLM receives the full picture:

      - Full query (for context)
      - Already-identified parts annotated inline as [text](type)
      - Remaining uncovered text
      - Semantic search suggestions: canonical taxonomy values that likely apply

    The LLM labels the remaining gaps and can use the suggested values in
    its normalizations output to bridge morphological variants.
    """
    hint_lines = "\n".join(
        f"  {field_type}: {', '.join(values)}"
        for field_type, values in taxonomy_hints.items()
    )
    context_block = (
        f"Full query: {full_query}\n"
        f"Already identified: {annotated}\n"
        f"Remaining to label: {gap_text or '(none)'}\n"
    )
    if hint_lines:
        context_block += f"Semantic search suggests these taxonomy values may apply:\n{hint_lines}\n"

    model = ctx.settings.llm_model
    result = await ctx.llm.complete_structured(
        system_prompt=SEGMENTATION_SYSTEM,
        user_content=context_block,
        json_schema=SEGMENTATION_SCHEMA,
        schema_name="query_segments",
        model=model,
    )
    status = "success" if result.ok else ("refusal" if result.refusal else "error")
    ctx.metrics.record_llm_usage(
        model=result.usage.model or model, status=status,
        input_tok=result.usage.input_tokens, output_tok=result.usage.output_tokens,
        cost=result.usage.cost_usd,
    )
    return result.parsed if result.ok else None


def _resolves(seg_type: str, text: str, taxonomy_hints: dict[str, list[str]]) -> bool:
    """Return True if the text is already a known canonical value for this type.

    A value is considered resolvable when it appears verbatim (case-insensitive)
    in the semantic hints for its type — meaning the taxonomy already contains it.
    Types like 'city', 'model', 'brand' are always resolvable (free-text fields).
    """
    free_text_types = {"city", "model", "brand", "condition"}
    if seg_type in free_text_types:
        return True
    known = [v.lower() for v in taxonomy_hints.get(seg_type, [])]
    return text.lower() in known


def _segments_from_llm(
    raw_segments: list[dict],
    clean_q: str,
    existing: list[Segment],
    taxonomy_hints: dict[str, list[str]],
) -> list[Segment]:
    """Turn LLM-labelled chunks into Segments, locating each in clean_q.

    taxonomy_hints (already computed from one upfront semantic search) maps
    field_type → [canonical values]. When the LLM returns a segment text that
    doesn't exactly match a known value, the first hint for that type is used
    as the resolved text, bridging morphological variants like עגלת → עגלות.
    """
    covered: set[int] = set()
    for seg in existing:
        if seg.start >= 0:
            covered.update(range(seg.start, seg.end))

    out: list[Segment] = []
    for item in raw_segments:
        text = (item.get("text") or "").strip()
        seg_type = item.get("type") or ""
        if not text or seg_type not in segment_types.REGISTRY:
            continue

        # Try to resolve the raw text. Fall back to the semantic hint only when the
        # raw text yields nothing in the taxonomy (e.g. "עגלת תינוק" → "עגלות").
        resolved_text = text
        if not _resolves(seg_type, text, taxonomy_hints):
            for hint in taxonomy_hints.get(seg_type, []):
                if hint.lower() != text.lower():
                    resolved_text = hint
                    break

        start = _find_free(clean_q, text, covered)
        if start >= 0:
            end = start + len(text)
            covered.update(range(start, end))
            out.append(Segment(text=resolved_text, type=seg_type,
                               start=start, end=end, source="llm", raw_text=text))
        else:
            out.append(Segment(text=resolved_text, type=seg_type, source="llm", raw_text=text))
    return out


def _find_free(clean_q: str, text: str, covered: set[int]) -> int:
    """First whole-word occurrence of text in clean_q not overlapping covered."""
    start = clean_q.find(text)
    while start != -1:
        end = start + len(text)
        if covered.isdisjoint(range(start, end)) and library._is_word_boundary(clean_q, start, end):
            return start
        start = clean_q.find(text, start + 1)
    return -1


def _learn_normalizations(raw: list[dict], ctx: NodeContext) -> list[str]:
    learned: list[str] = []
    for item in raw:
        wrong = (item.get("from") or "").strip()
        correct = (item.get("to") or "").strip()
        if wrong and correct and wrong != correct:
            ctx.normalization_db.learn(wrong, correct)
            learned.append(f"{wrong}→{correct}")
    return learned
