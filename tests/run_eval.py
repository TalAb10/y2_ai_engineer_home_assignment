"""
Batch evaluation runner for eval_cases.json.

Usage:
    python tests/run_eval.py                    # rules-only (offline, no API key needed)
    python tests/run_eval.py --llm              # use real LLM (requires OPENAI_API_KEY in .env)
    python tests/run_eval.py --tags realestate  # run only cases with this tag
    python tests/run_eval.py --id veh_001       # run a single case by id

Each case checks:
  - category matches expected_category  (or allows any category when expected_category is null)
  - every key in expected_params is present with the right value
  - every key in expected_absent_params is NOT in the output
  - every flag in expected_flags is present in security_flags

Exit code: 0 if all selected cases pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows so box-drawing characters render correctly.
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = Path(__file__).parent / "eval_cases.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
SEP    = "-" * 60   # ASCII separator — safe on all terminals


def _load_cases(tag_filter: str | None, id_filter: str | None) -> list[dict]:
    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    cases = data["cases"]
    if id_filter:
        cases = [c for c in cases if c["id"] == id_filter]
    if tag_filter:
        cases = [c for c in cases if tag_filter in c.get("tags", [])]
    return cases


def _build_ctx(use_llm: bool):
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    from config import Settings
    from taxonomy.loader import load
    from llm.client import create_llm_client, NoOpLLMClient
    from cache.cache import create_cache
    from graph.context import NodeContext
    from observability import metrics as m
    from patterns.library import PatternLibrary
    from patterns.normalizations import NormalizationDB

    settings = Settings(taxonomy_path=REPO_ROOT / "yad2_search_taxonomy.json")
    if not use_llm:
        settings = Settings(
            openai_api_key="",
            llm_enabled=False,
            taxonomy_path=REPO_ROOT / "yad2_search_taxonomy.json",
        )

    sem_index = None
    if use_llm:
        from taxonomy.semantic_index import DEFAULT_INDEX_PATH, DEFAULT_META_PATH, SemanticTaxonomyIndex
        if DEFAULT_INDEX_PATH.exists() and DEFAULT_META_PATH.exists():
            sem_index = SemanticTaxonomyIndex.load(DEFAULT_INDEX_PATH, DEFAULT_META_PATH)

    return NodeContext(
        taxonomy=load(settings.taxonomy_path),
        llm=create_llm_client(settings),
        cache=create_cache(settings),
        settings=settings,
        metrics=m,
        pattern_library=PatternLibrary(),
        normalization_db=NormalizationDB(),
        semantic_index=sem_index,
    )


async def _parse(query: str, ctx) -> dict:
    from graph.build import build_graph
    from graph.state import GraphState

    raw = await build_graph(ctx).ainvoke(GraphState(raw_q=query))
    final = GraphState.model_validate(dict(raw))
    return {
        "category":       final.category,
        "params":         final.params,
        "confidence":     final.confidence,
        "security_flags": final.security_flags,
        "llm_used":       final.llm_used,
        "notes":          final.notes,
    }


def _check_params(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Return a list of failure messages; empty list means all pass."""
    failures: list[str] = []
    for key, exp_val in expected.items():
        if key not in actual:
            failures.append(f"  missing param  {key!r}")
            continue
        got = actual[key]
        if isinstance(exp_val, dict):
            # Range check — only compare keys that are present in the expected dict.
            for sub_key, sub_val in exp_val.items():
                got_sub = got.get(sub_key) if isinstance(got, dict) else None
                if got_sub != sub_val:
                    failures.append(f"  {key}[{sub_key!r}]  expected {sub_val!r}, got {got_sub!r}")
        elif got != exp_val:
            failures.append(f"  {key}  expected {exp_val!r}, got {got!r}")
    return failures


def _print_result(case: dict, result: dict, elapsed: float, failures: list[str]) -> bool:
    passed = len(failures) == 0
    icon   = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    tag_str = ", ".join(case.get("tags", []))

    print(f"\n{icon}  {BOLD}{case['id']}{RESET}  [{tag_str}]  {elapsed*1000:.0f}ms")
    print(f"   query:     {case['query'][:80]!r}")
    print(f"   expected:  category={case.get('expected_category')!r}")
    print(f"   got:       category={result['category']!r}  confidence={result['confidence']:.2f}"
          f"  llm={'yes' if result['llm_used'] else 'no'}"
          f"  flags={result['security_flags']}")

    if result["params"]:
        print(f"   params:    {json.dumps(result['params'], ensure_ascii=False)}")

    if not passed:
        for msg in failures:
            print(f"   {RED}{msg}{RESET}")
    if case.get("notes"):
        print(f"   notes:     {case['notes']}")

    return passed


async def _run(cases: list[dict], ctx) -> tuple[int, int]:
    passed = failed = 0

    for case in cases:
        t0 = time.perf_counter()
        try:
            result = await _parse(case["query"], ctx)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"\n{RED}✗  {case['id']}  EXCEPTION: {exc}{RESET}")
            failed += 1
            continue
        elapsed = time.perf_counter() - t0

        failures: list[str] = []

        # 1. Category check
        exp_cat = case.get("expected_category")
        if exp_cat is not None and result["category"] != exp_cat:
            failures.append(f"  category  expected {exp_cat!r}, got {result['category']!r}")

        # 2. Required params
        param_failures = _check_params(case.get("expected_params", {}), result["params"])
        failures.extend(param_failures)

        # 3. Absent params (subcategory consistency)
        for key in case.get("expected_absent_params", []):
            if key in result["params"]:
                failures.append(f"  param {key!r} should be absent but was present")

        # 4. Security flags
        for flag in case.get("expected_flags", []):
            if flag not in result["security_flags"]:
                failures.append(f"  missing security flag {flag!r} (got {result['security_flags']})")

        ok = _print_result(case, result, elapsed, failures)
        if ok:
            passed += 1
        else:
            failed += 1

    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch query evaluation runner")
    parser.add_argument("--llm",  action="store_true", help="Enable real LLM calls (requires OPENAI_API_KEY)")
    parser.add_argument("--tags", metavar="TAG",       help="Run only cases with this tag")
    parser.add_argument("--id",   metavar="ID",        help="Run only the case with this id")
    args = parser.parse_args()

    cases = _load_cases(tag_filter=args.tags, id_filter=args.id)
    if not cases:
        print("No matching cases found.")
        return 1

    mode = f"{CYAN}LLM enabled{RESET}" if args.llm else f"{YELLOW}rules-only (offline){RESET}"
    print(f"\n{BOLD}Yad2 batch eval — {len(cases)} cases — {mode}{RESET}")
    print(SEP)

    ctx = _build_ctx(use_llm=args.llm)
    passed, failed = asyncio.run(_run(cases, ctx))

    print("\n" + "─" * 60)
    total = passed + failed
    color = GREEN if failed == 0 else RED
    print(f"{color}{BOLD}{passed}/{total} passed{RESET}"
          + (f"  {RED}{failed} failed{RESET}" if failed else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
