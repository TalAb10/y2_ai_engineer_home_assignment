"""NormalizationDB — typos the LLM corrected, fed back into the normalize node.

The LLM receives already-normalised text, so when it still has to mentally correct
a word, that word is a gap in our static typo_map. The LLM reports such corrections
(only clear, obvious ones) and we remember them here. The normalize node merges
this with the static taxonomy typo_map, so the next identical typo is fixed before
the LLM is ever consulted.

Holds only dynamically-learned entries; the static taxonomy typo_map stays the
authoritative base and is merged in at lookup time.
"""

from __future__ import annotations

import threading


class NormalizationDB:
    """Thread-safe in-memory map of misspelling → canonical form."""

    def __init__(self) -> None:
        self._map: dict[str, str] = {}
        self._lock = threading.Lock()

    def learn(self, wrong: str, correct: str) -> None:
        wrong, correct = wrong.strip(), correct.strip()
        if wrong and correct and wrong != correct:
            with self._lock:
                self._map[wrong] = correct

    def all(self) -> dict[str, str]:
        with self._lock:
            return dict(self._map)

    def size(self) -> int:
        with self._lock:
            return len(self._map)
