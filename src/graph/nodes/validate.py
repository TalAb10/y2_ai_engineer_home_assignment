"""Node: validate

Responsibility: schema enforcement + confidence computation.

  1. Run params through the per-vertical Pydantic schema:
       - Any key not declared in the schema is silently dropped (injection guard).
       - Values are type-coerced and enum-checked.
  2. Compute final confidence score from:
       - classification_certainty (how dominant the winning category was, in [0,1])
       - extraction_coverage (fraction of tokens matched by rules)
  3. Append normalisation notes (e.g. "city name corrected").

After this node the state is the final API response shape.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from graph.context import NodeContext
from graph.state import GraphState
from taxonomy.schemas import CATEGORY_TO_SCHEMA

logger = logging.getLogger(__name__)

# ── Confidence formula weights ─────────────────────────────────────────────────
# Confidence blends two signals, each already in [0, 1]:
#   • classification_certainty — how dominant the winning category was
#   • extraction_coverage      — how much of the query the rules understood
# Coverage is weighted higher because "we understood the query" is a stronger
# correctness signal than "the category was unambiguous".
_CLASSIFICATION_WEIGHT = 0.4
_COVERAGE_WEIGHT = 0.6


def _compute_confidence(classification_certainty: float, extraction_coverage: float) -> float:
    """Heuristic confidence in [0, 1]. Both inputs are already in [0, 1]; clamp defensively."""
    certainty_score = min(max(classification_certainty, 0.0), 1.0)
    coverage_score = min(max(extraction_coverage, 0.0), 1.0)
    raw = _CLASSIFICATION_WEIGHT * certainty_score + _COVERAGE_WEIGHT * coverage_score
    return round(raw, 3)


def _drop_none_values(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove None values so the JSON response stays clean."""
    return {k: (_drop_none_values(v) if isinstance(v, dict) else v) for k, v in d.items() if v is not None}


def run(state: GraphState, ctx: NodeContext) -> dict:
    """Validate params against the vertical schema and compute confidence."""
    category = state.category
    notes: list[str] = list(state.notes)

    schema_class = CATEGORY_TO_SCHEMA.get(category)
    if schema_class is None:
        return {
            "params": {},
            "confidence": 0.0,
            "notes": notes + [f"קטגוריה לא ידועה: {category}"],
            "errors": state.errors + ["unknown_category"],
        }

    validated_params: dict[str, Any] = {}
    try:
        parsed = schema_class.model_validate(state.params)
        validated_params = _drop_none_values(
            parsed.model_dump(by_alias=True, exclude_none=True)
        )
    except ValidationError as exc:
        # Log the validation errors but return whatever we could extract
        logger.warning("Validation errors for category=%s: %s", category, exc)
        notes.append("חלק מהשדות לא עמדו בסכמה ונמחקו")
        # Best-effort: pass through only known valid fields
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            notes.append(f"שדה לא תקין הוסר: {loc}")
        # Re-validate with only known keys (field names + their aliases)
        allowed_keys: set[str] = set(schema_class.model_fields.keys())
        for field_info in schema_class.model_fields.values():
            alias = getattr(field_info, "alias", None)
            if alias:
                allowed_keys.add(alias)
        try:
            filtered = {k: v for k, v in state.params.items() if k in allowed_keys}
            parsed = schema_class.model_validate(filtered)
            validated_params = _drop_none_values(parsed.model_dump(by_alias=True, exclude_none=True))
        except ValidationError:
            validated_params = {}

    confidence = _compute_confidence(
        classification_certainty=state.classification_certainty,
        extraction_coverage=state.extraction_coverage,
    )

    return {
        "params": validated_params,
        "confidence": confidence,
        "notes": notes,
    }
