"""Node: normalize

Responsibility: linguistic normalisation of the sanitised query.

  1. Apply typo map (static taxonomy + LLM-learned corrections) — fixes misspellings
  2. Normalise Hebrew number words: מיליון → 1000000, אלף → 1000
  3. Normalise unit variants: מ׳/מטר/מטרים → מ״ר (where context implies area)
  4. Normalise ק״מ variants
  5. Tokenise → produces state.query_words (whitespace split, lowercased)

Hebrew morphology: clitic-prefix stripping is handled at lookup time
(in loader.py strip_prefixes()), not here — we preserve the original form
in query_words so the sanitize log retains readable context.
"""

from __future__ import annotations

import re
import unicodedata

from graph.context import NodeContext
from graph.state import GraphState
from taxonomy.loader import CLITIC_PREFIXES

# ── Number-word normalisation ─────────────────────────────────────────────────
_MILLION_RE = re.compile(r"(?:(\d+(?:[.,]\d+)?)\s*)?מי?ליון", re.UNICODE)  # optional digit prefix; bare "מליון" → 1,000,000
_THOUSAND_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*אלף", re.UNICODE)

# ── Unit normalisation ────────────────────────────────────────────────────────
_AREA_UNIT_RE = re.compile(r"\bמ[׳']\b|\bמטר(?:ים)?\b", re.UNICODE)
_KM_UNIT_RE = re.compile(r"\bקמ\b|\bק\"מ\b|\bק'מ\b", re.UNICODE)
# Collapse every shekel form to one token so the price pattern only matches "שח".
_CURRENCY_RE = re.compile(r"ש[\"״'׳]?ח|שקל(?:ים)?|₪", re.UNICODE)

# ── Range marker normalisation (for downstream regex) ────────────────────────
# We keep the Hebrew words; the _helpers.py regexes understand them.
# This pass only collapses common abbreviations.
_RANGE_ALIASES = [
    (re.compile(r"\bעד\s+כ-\b", re.UNICODE), "עד"),
    (re.compile(r"\bלפחות\b", re.UNICODE), "מעל"),
    (re.compile(r"\bמינימום\b", re.UNICODE), "מעל"),
    (re.compile(r"\bמקסימום\b", re.UNICODE), "עד"),
]


def _apply_typo_map(text: str, typo_map: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Typo correction that handles Hebrew clitic prefixes.

    Hebrew prefixes (ב/ל/מ/ה/ו/ש/כ) attach directly to words so \b won't match
    'ירושליים' inside 'בירושליים'.  We split on whitespace, try to correct each
    token (stripping known prefixes before lookup), then reassemble.  The prefix
    list is shared with taxonomy.loader so lookups and corrections stay in sync.

    Returns (corrected_text, {original_word: correction}). The change map is built
    here, per word, rather than by zipping the before/after token lists: a single
    typo can expand into several words (e.g. 'תלאביב' → 'תל אביב-יפו'), which shifts
    every later token and would corrupt a positional zip.
    """
    changes: dict[str, str] = {}

    def _correct_word(word: str) -> str:
        corrected = word
        if word in typo_map:
            corrected = typo_map[word]
        else:
            # Try stripping clitic prefixes and correcting the remainder
            for prefix in CLITIC_PREFIXES:
                if word.startswith(prefix) and len(word) > len(prefix) + 1:
                    stem = word[len(prefix):]
                    if stem in typo_map:
                        corrected = prefix + typo_map[stem]
                        break
        if corrected != word:
            changes[word] = corrected
        return corrected

    corrected_text = " ".join(_correct_word(word) for word in text.split())
    return corrected_text, changes


def run(state: GraphState, ctx: NodeContext) -> dict:
    """Normalise clean_q and produce tokens list."""
    text = state.clean_q

    # 1. Typo corrections: static taxonomy map + dynamically learned corrections.
    #    Learned entries win on conflict (they reflect real misses the LLM caught).
    typo_map = {**ctx.taxonomy.typo_map, **ctx.normalization_db.all()}
    text, normalization_applied = _apply_typo_map(text, typo_map)

    # 2. Number-word → digit
    def _replace_million(m: re.Match) -> str:
        raw = m.group(1)
        val = float(raw.replace(",", ".")) * 1_000_000 if raw else 1_000_000
        return str(int(val))

    def _replace_thousand(m: re.Match) -> str:
        val = float(m.group(1).replace(",", ".")) * 1_000
        return str(int(val))

    text = _MILLION_RE.sub(_replace_million, text)
    text = _THOUSAND_RE.sub(_replace_thousand, text)

    # 3. Unit aliases
    text = _AREA_UNIT_RE.sub("מ״ר", text)
    text = _KM_UNIT_RE.sub("ק״מ", text)
    text = _CURRENCY_RE.sub("שח", text)

    # 4. Range aliases
    for pattern, replacement in _RANGE_ALIASES:
        text = pattern.sub(replacement, text)

    # 5. Tokenise (simple whitespace split; punctuation kept with token)
    query_words = [word for word in text.split() if word]

    return {
        "clean_q": text,
        "query_words": query_words,
        "normalization_applied": normalization_applied,
    }
