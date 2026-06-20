"""Data-driven tests over the labelled dataset (cases.json).

Each case becomes one parametrized pytest case, run through the full graph offline
(rules-only via the shared `ctx` fixture). This is the seed of the larger
precision/recall dataset described in ARCHITECTURE.md.

Run just these:   pytest tests/dataset
One case:         pytest tests/dataset -k re_001
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from graph.build import build_graph
from graph.state import GraphState

CASES: list[dict] = json.loads(
    (Path(__file__).parent / "cases.json").read_text(encoding="utf-8")
)["cases"]


async def _parse(query: str, ctx) -> GraphState:
    raw = await build_graph(ctx).ainvoke(GraphState(raw_q=query))
    return GraphState.model_validate(dict(raw))


def _assert_params(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Subset match: every expected key present with the right value.

    A dict value (a numeric range) is compared key-by-key, so a case can assert
    only `max` without pinning `min`.
    """
    for key, exp in expected.items():
        assert key in actual, f"missing param {key!r} (got {actual})"
        got = actual[key]
        if isinstance(exp, dict):
            for sub_key, sub_val in exp.items():
                got_sub = got.get(sub_key) if isinstance(got, dict) else None
                assert got_sub == sub_val, f"{key}[{sub_key!r}] expected {sub_val!r}, got {got_sub!r}"
        else:
            assert got == exp, f"{key} expected {exp!r}, got {got!r}"


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
async def test_case(case: dict, ctx) -> None:
    result = await _parse(case["query"], ctx)

    expected_category = case.get("expected_category")
    if expected_category is not None:
        assert result.category == expected_category, \
            f"category expected {expected_category!r}, got {result.category!r}"

    _assert_params(case.get("expected_params", {}), result.params)

    for key in case.get("expected_absent_params", []):
        assert key not in result.params, f"param {key!r} should be absent (got {result.params})"

    for flag in case.get("expected_flags", []):
        assert flag in result.security_flags, \
            f"missing security flag {flag!r} (got {result.security_flags})"
