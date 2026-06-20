"""PatternLibrary — the self-learning core.

A *pattern* is a query fragment with its digits blanked out, one "?" per digit:

    "עד 70000 שח"  →  "עד ????? שח"
    "2018-2021"    →  "????-????"
    "טויוטה"        →  "טויוטה"          (no digits → unchanged)

The library maps each pattern to the *set* of segment types it has ever meant.
A set, not a single value, because the same shape can mean different things:

    "עד ????"  →  {"price", "year_range"}   ← "עד 9000" is price, "עד 2018" could be a year

Resolution (which type actually applies to a given span) happens later in the
extract node, using value-range validation + the surrounding context. The library
only remembers shapes; it never decides meaning on its own.

Abstraction is length-preserving (each digit becomes exactly one "?"), so a span
[start:end) found in the abstracted query is the identical span in the original
text — no re-alignment needed.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

_DIGIT_RE = re.compile(r"[0-9]")
_WORD_RE = re.compile(r"\S+")

# A pattern shorter than this is too generic to match safely (e.g. a lone "?").
_MIN_PATTERN_LEN = 2


def abstract(text: str) -> str:
    """Blank every digit to '?' (length-preserving)."""
    return _DIGIT_RE.sub("?", text)


def is_salient_token(token: str) -> bool:
    """True if a token carries standalone search meaning.

    Single-character tokens are dropped as noise UNLESS they are digits. A lone
    Hebrew letter is almost always a clitic prefix (ב/ל/מ/ה/ו/ש/כ) that means
    nothing on its own, but a lone digit ("3" in "מודל 3" / "מאזדה 3") is a real
    model or spec number that must be kept. This is the single definition of
    "salient", used by both coverage scoring and the LLM gap-text builder.
    """
    return len(token) > 1 or token.isdigit()


@dataclass
class Segment:
    """A resolved, meaningful chunk of the query.

    start/end are character offsets in clean_q; both are -1 for LLM segments
    whose text could not be located verbatim (they still extract a value, but
    do not count toward coverage).

    raw_text holds the original query text before canonical substitution.
    This is what gets learned into the pattern library, so future queries with
    the same surface form are recognised — not the canonical replacement.
    """
    text: str
    type: str
    start: int = -1
    end: int = -1
    source: str = "pattern"   # "pattern" | "llm"
    raw_text: str = ""

    def __post_init__(self) -> None:
        if not self.raw_text:
            self.raw_text = self.text

    def to_dict(self) -> dict:
        return {"text": self.text, "type": self.type,
                "start": self.start, "end": self.end, "source": self.source}


@dataclass
class SpanMatch:
    """A pattern hit: a span in clean_q plus every type that shape can mean."""
    start: int
    end: int
    types: set[str] = field(default_factory=set)


class PatternLibrary:
    """Thread-safe in-memory store of pattern → {segment types}.

    Never evicts and never overwrites — learning only ever *adds* a meaning.
    """

    def __init__(self) -> None:
        self._patterns: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def learn(self, text: str, segment_type: str) -> None:
        pattern = abstract(text).strip()
        if len(pattern) < _MIN_PATTERN_LEN or not segment_type:
            return
        with self._lock:
            self._patterns.setdefault(pattern, set()).add(segment_type)

    def scan(self, clean_q: str) -> list[SpanMatch]:
        """Find every known pattern as a whole-word substring of clean_q.

        Returns raw (possibly overlapping) hits, each carrying the full set of
        types its pattern can mean. Overlap resolution happens later via
        merge_spans, so pattern and discoverer hits are merged together.
        """
        abstracted = abstract(clean_q)

        # Snapshot under the lock so the scan loop runs without holding it.
        # frozenset avoids copying types for patterns that never match.
        with self._lock:
            snapshot = [(p, frozenset(t)) for p, t in self._patterns.items()]

        hits: list[SpanMatch] = []
        for pattern, types in snapshot:
            start = abstracted.find(pattern)
            while start != -1:
                end = start + len(pattern)
                if _is_word_boundary(clean_q, start, end):
                    hits.append(SpanMatch(start=start, end=end, types=set(types)))
                start = abstracted.find(pattern, start + 1)
        return hits

    def size(self) -> int:
        with self._lock:
            return len(self._patterns)


def merge_spans(spans: list[SpanMatch]) -> list[SpanMatch]:
    """Merge overlapping spans into non-overlapping ones, unioning their types.

    Two candidates that cover overlapping characters (e.g. a city phrase and a
    word inside it, or two readings of the same number) become one span spanning
    both, carrying every candidate type. The extract node then resolves each
    merged span to a single type.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, -s.end))
    merged: list[SpanMatch] = [SpanMatch(ordered[0].start, ordered[0].end, set(ordered[0].types))]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start < last.end:           # overlaps the current cluster
            last.end = max(last.end, span.end)
            last.types |= span.types
        else:
            merged.append(SpanMatch(span.start, span.end, set(span.types)))
    return merged


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """True if [start:end) starts and ends on a whitespace/string boundary.

    Prevents matching 'תל' inside 'בתל' or '????' inside '????-????'.
    """
    left_ok = start == 0 or text[start - 1] == " "
    right_ok = end == len(text) or text[end] == " "
    return left_ok and right_ok


def _covered_chars(segments: list[Segment], max_len: int,
                   extra_spans: list[tuple[int, int]] = ()) -> set[int]:
    """Build the set of character positions covered by segments and extra spans.

    Clamps all offsets to [0, max_len) so a malformed segment or LLM response
    with an out-of-range end never materialises a huge range into memory.
    """
    covered: set[int] = set()
    for seg in segments:
        if seg.start >= 0:
            covered.update(range(seg.start, min(seg.end, max_len)))
    for start, end in extra_spans:
        covered.update(range(max(0, start), min(end, max_len)))
    return covered


def coverage(segments: list[Segment], clean_q: str,
             extra_spans: list[tuple[int, int]] = ()) -> float:
    """Fraction of salient words (see is_salient_token) fully covered by a segment or extra span.

    `extra_spans` lets the caller fold in non-segment coverage (e.g. the character
    ranges of deterministic numeric matches) so price/year/rooms count too.
    """
    covered = _covered_chars(segments, len(clean_q), extra_spans)

    total = hits = 0
    for match in _WORD_RE.finditer(clean_q):
        if not is_salient_token(match.group()):
            continue
        total += 1
        if all(i in covered for i in range(match.start(), match.end())):
            hits += 1
    return 1.0 if total == 0 else hits / total


def annotate(clean_q: str, segments: list[Segment]) -> str:
    """Render the query with matched spans marked as [text](type), for the LLM.

    e.g. "טויוטה [2018-2021](year_range) [עד 70000 שח](price) אוטומטי"
    """
    spans = sorted((s for s in segments if s.start >= 0), key=lambda s: s.start)
    out: list[str] = []
    pos = 0
    for seg in spans:
        if seg.start < pos:
            continue
        out.append(clean_q[pos:seg.start])
        out.append(f"[{seg.text}]({seg.type})")
        pos = seg.end
    out.append(clean_q[pos:])
    return "".join(out)
