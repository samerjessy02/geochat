"""
agents/logging_config.py — one place to configure and obtain loggers.

Every module obtains its logger via :func:`get_logger`, which guarantees the
root logging config has been applied exactly once. Behaviour is driven by
environment (see ``config.py``):

    LOG_LEVEL     DEBUG | INFO | WARNING | ERROR      (default INFO)
    LOG_TO_FILE   also write to a rotating file        (default false)
    LOG_FILE      path of that file                    (default geochat.log)
    LOG_JSON      emit one JSON object per line         (default false)

Two conveniences on top of the stdlib logger:

    get_logger(name)      -> a namespaced logger ("geochat.<name>")
    log_step(logger, msg) -> context manager that logs "▶ start" / "✓ done
                             (Nms)" around a block, and "✗ failed (Nms)" if it
                             raises — so a single query's journey through the
                             pipeline reads as a timed, indented trace.

Set ``LOG_LEVEL=DEBUG`` to see per-stage detail (scores, counts, SQL, chosen
web tier); INFO gives a clean high-level trace of every step.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

from config import settings

_ROOT_NAME = "geochat"
_configured = False


class _JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object (for log shippers)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any structured extras passed via logger.info(..., extra={...}).
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure the ``geochat`` logger tree once (idempotent)."""
    global _configured
    if _configured:
        return

    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False

    if settings.log_json:
        fmt: logging.Formatter = _JsonFormatter()
    else:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if settings.log_to_file:
        file_handler = RotatingFileHandler(
            settings.log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Mirror every record into the queryable SQLite store for the Activity Logs UI.
    try:
        from agents.log_store import SQLiteLogHandler, init_log_db
        init_log_db()
        logger.addHandler(SQLiteLogHandler())
    except Exception as e:  # noqa: BLE001 — never block startup on the log store
        logging.getLogger(_ROOT_NAME).warning("log store unavailable: %s", e)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, ensuring logging is configured first."""
    setup_logging()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


@contextmanager
def log_step(logger: logging.Logger, message: str, level: int = logging.INFO):
    """Log the start, successful end (with duration), or failure of a block.

    Usage::

        with log_step(log, "hybrid retrieval"):
            hits = hybrid_search(query)
    """
    start = time.perf_counter()
    logger.log(level, "▶ %s", message, extra={"ctx_target": message})
    try:
        yield
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception("✗ %s — failed after %.0fms", message, elapsed,
                         extra={"ctx_target": message, "ctx_duration_ms": round(elapsed, 1)})
        raise
    else:
        elapsed = (time.perf_counter() - start) * 1000
        logger.log(level, "✓ %s — %.0fms", message, elapsed,
                   extra={"ctx_target": message, "ctx_duration_ms": round(elapsed, 1)})


def snippet(text: str, length: int = 80) -> str:
    """Collapse whitespace and truncate text for safe single-line logging."""
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= length else collapsed[: length - 1] + "…"
