"""Structured JSON logging configuration.

Every log record emitted via the standard `logging` module is formatted as a
single JSON line so it can be ingested by any log aggregator (Loki, Datadog, etc.).

Two special helpers are provided:
  - log_parse_decision()   → records every classification/extraction decision
  - log_security_event()   → records sanitization flag triggers (separate log field
                              makes it easy to build security dashboards / alerts)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


def setup_logging(level: str = "INFO") -> None:
    """Call once at application startup to install the JSON formatter."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


_parse_logger = logging.getLogger("yad2.parse")
_security_logger = logging.getLogger("yad2.security")


def log_parse_decision(
    query: str,
    category: str | None,
    confidence: float,
    cache_hit: bool,
    llm_used: bool,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    _parse_logger.info(
        "parse_decision",
        extra={
            "query_len": len(query),
            "category": category,
            "confidence": confidence,
            "cache_hit": cache_hit,
            "llm_used": llm_used,
            "model": model,
            **(extra or {}),
        },
    )


def log_security_event(
    flag_type: str,
    query_snippet: str,
    details: str = "",
) -> None:
    _security_logger.warning(
        "security_event",
        extra={
            "flag_type": flag_type,
            # Truncate to avoid storing large malicious payloads in logs
            "query_snippet": query_snippet[:80],
            "details": details,
        },
    )


def log_injection_confirmed(
    query_snippet: str,
    flags: list[str],
    reason: str = "llm_confirmed_injection",
) -> None:
    _security_logger.error(
        "injection_confirmed",
        extra={
            "query_snippet": query_snippet[:80],
            "flags": flags,
            "reason": reason,
        },
    )
