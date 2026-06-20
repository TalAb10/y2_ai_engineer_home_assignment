"""Node: sanitize — input hygiene + prompt-injection screening before any LLM call.

  1. NFKC unicode normalisation
  2. Strip emojis / control / direction-override characters
  3. Enforce max length
  4. Scan for injection markers → security_flags (semantic categories)

A flagged query still flows through the pipeline, but:
  - the security_check node runs an LLM deepcheck that can block it, and
  - the extract node never escalates a flagged query to the value LLM.
Every flag is logged and counted so security dashboards can alert on them.

Why categories and not full-sentence regexes: we match short, high-signal markers
(a delimiter tag, a role-override phrase, an evasion verb) that have no legitimate
place in a property/car/second-hand search. English is matched case-insensitively;
Hebrew uses whitespace boundaries because \\b is unreliable at Hebrew word edges.
"""

from __future__ import annotations

import re
import unicodedata

from graph.context import NodeContext
from graph.state import GraphState
from observability.logging_config import log_security_event

# Strip: control chars (keep \t \n \r), emoji ranges, invisible/RTL overrides.
_STRIP_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\U0001F300-\U0001FAFF"
    r"​-‏‪-‮"
    r"]",
    re.UNICODE,
)

# Injection markers grouped by category. Any hit is a risk signal.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("instruction_override", re.compile(
        r"\b(ignore|disregard|override|bypass|forget)\b"
        r"|(?<!\S)(התעלם|תתעלם|עקוף|שכח|תשכח)(?!\S)", re.IGNORECASE | re.UNICODE)),
    ("role_injection", re.compile(
        r"\b(you are (now|a)|act as|pretend|jailbreak|new instructions?)\b"
        r"|(?<!\S)תשחק(?!\S)", re.IGNORECASE | re.UNICODE)),
    ("prompt_extraction", re.compile(
        r"\b(system prompt|print your|repeat your|reveal|prompt)\b"
        r"|(?<!\S)(פרומפט|גלה)(?!\S)", re.IGNORECASE | re.UNICODE)),
    ("delimiter_injection", re.compile(
        r"</?system>|<\|.*?\|>|\[/?INST\]", re.IGNORECASE | re.UNICODE)),
]


def run(state: GraphState, ctx: NodeContext) -> dict:
    """Sanitize raw_q → clean_q + security_flags."""
    text = unicodedata.normalize("NFKC", state.raw_q)
    text = _STRIP_RE.sub("", text).strip()

    max_chars = ctx.settings.max_input_chars
    if len(text) > max_chars:
        text = text[:max_chars]

    flags: list[str] = []
    for flag, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            flags.append(flag)
            log_security_event(flag_type=flag, query_snippet=text[:80],
                               details=f"injection marker: {flag}")
            ctx.metrics.security_events_total.labels(flag_type=flag).inc()

    return {"clean_q": text, "security_flags": flags}
