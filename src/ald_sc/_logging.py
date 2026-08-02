"""Structured logging configuration for ALD-SC.

Configures ``structlog`` to render log events to stdout with a readable
console format. Importing this module (or the package) configures
structlog once; subsequent imports are no-ops.

This keeps library code free of ``print()`` (per AGENTS.md §10) while
still giving notebooks human-readable per-epoch progress.
"""

from __future__ import annotations

import logging

import structlog

_CONFIGURED = False


def configure_logging() -> None:
    """Configure structlog with a console renderer (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _CONFIGURED = True


configure_logging()
