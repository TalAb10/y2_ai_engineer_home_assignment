"""Semantic taxonomy index — FAISS nearest-neighbour over embedded taxonomy values.

At build time (scripts/build_taxonomy_index.py):
  - Every canonical taxonomy value is embedded with text-embedding-3-small.
  - The embeddings + metadata are saved to disk as two files:
      taxonomy.faiss   — FAISS IndexFlatIP (cosine similarity via L2-normalised vectors)
      taxonomy.meta.json — parallel list of {value, field_type} dicts

At runtime:
  - The index is loaded once at startup and injected into NodeContext.
  - After the LLM labels a segment (e.g. text="עגלת", type="subcategory"),
    semantic_search() returns the closest canonical value ("עגלות"), bridging
    Hebrew morphological and spelling variants that exact lookup would miss.

Similarity threshold (default 0.45): below this the raw text is kept as-is so
ambiguous or clearly wrong matches don't silently corrupt results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "taxonomy.faiss"
DEFAULT_META_PATH  = Path(__file__).resolve().parent.parent.parent / "taxonomy.meta.json"
SIMILARITY_THRESHOLD = 0.45


@dataclass(frozen=True)
class TaxonomyMatch:
    value: str        # canonical taxonomy value (e.g. "עגלות")
    field_type: str   # which lookup set it came from (e.g. "subcategory")
    score: float      # cosine similarity [0, 1]


class SemanticTaxonomyIndex:
    """Loaded FAISS index with parallel metadata list."""

    def __init__(self, index, entries: list[dict[str, str]]) -> None:
        self._index   = index
        self._entries = entries   # [{value, field_type}, ...]

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        embedding: list[float],
        field_type: str | None = None,
        k: int = 1,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[TaxonomyMatch]:
        """Return up to k closest canonical values, optionally filtered by type.

        Returns an empty list when no match exceeds the similarity threshold.
        """
        vec = np.array([embedding], dtype="float32")
        _l2_normalize(vec)

        # Search a bit wider than k so we have room to filter by type.
        search_k = min(k * 10 if field_type else k, self._index.ntotal)
        scores, indices = self._index.search(vec, search_k)

        results: list[TaxonomyMatch] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < threshold:
                break
            entry = self._entries[idx]
            if field_type and entry["field_type"] != field_type:
                continue
            results.append(TaxonomyMatch(
                value=entry["value"],
                field_type=entry["field_type"],
                score=float(score),
            ))
            if len(results) >= k:
                break
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, index_path: Path, meta_path: Path) -> None:
        import faiss
        faiss.write_index(self._index, str(index_path))
        meta_path.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved semantic index: %d entries → %s", len(self._entries), index_path)

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> SemanticTaxonomyIndex:
        import faiss
        index   = faiss.read_index(str(index_path))
        entries = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info("Loaded semantic index: %d entries from %s", len(entries), index_path)
        return cls(index, entries)

    @classmethod
    def build(
        cls,
        entries: list[dict[str, str]],   # [{value, field_type}, ...]
        embeddings: list[list[float]],
    ) -> SemanticTaxonomyIndex:
        """Build a new index from pre-computed embeddings."""
        import faiss
        mat = np.array(embeddings, dtype="float32")
        _l2_normalize(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        return cls(index, entries)

    @property
    def size(self) -> int:
        return self._index.ntotal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _l2_normalize(mat: np.ndarray) -> None:
    """Normalise rows in place so dot-product == cosine similarity."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat /= norms


def collect_entries(taxonomy_index) -> list[dict[str, str]]:
    """Collect every canonical taxonomy value with its field_type label."""
    entries: list[dict[str, str]] = []

    def _add(values: set[str] | list[str], field_type: str) -> None:
        for v in values:
            if v and v.strip():
                entries.append({"value": v.strip(), "field_type": field_type})

    _add(taxonomy_index.vehicle_manufacturers, "manufacturer")
    _add(taxonomy_index.vehicle_models,        "model")
    _add(taxonomy_index.vehicle_fuel_types,    "fuel_type")
    _add(taxonomy_index.vehicle_gearbox_types, "gearbox")
    _add(taxonomy_index.vehicle_colors,        "color")
    _add(taxonomy_index.re_property_types,     "property_type")
    _add(taxonomy_index.re_transaction_modes,  "transaction_mode")
    _add(taxonomy_index.re_cities,             "city")
    _add(taxonomy_index.re_condition_values,   "condition")
    _add(taxonomy_index.sh_sectors,            "sector")
    _add(taxonomy_index.sh_subcategories,      "subcategory")
    _add(taxonomy_index.sh_conditions,         "condition")

    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for e in entries:
        key = (e["value"], e["field_type"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique
