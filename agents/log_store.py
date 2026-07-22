"""
agents/log_store.py — queryable structured-log store for the Activity Logs UI.

A ``logging.Handler`` (:class:`SQLiteLogHandler`) mirrors every ``geochat.*`` log
record into a SQLite table so the UI can filter/search without parsing the raw
log file. Each row captures: timestamp, module (a coarse tag derived from the
logger name), logger, level, message, target, duration_ms, error_type and a full
trace (on failures).

The console + rotating-file handlers in ``logging_config`` are unchanged; this is
an additional sink. Writes are guarded by a lock (logging can run on FastAPI's
threadpool) and failures never propagate (``handleError`` prints to stderr).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone

_DB_PATH = os.path.join("logs", "geochat_logs.db")
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

# Map a logger-name suffix to a coarse module tag shown in the UI.
_MODULE_MAP = {
    "intent_router": "AGENT_ROUTER",
    "knowledge_pipeline": "RAG_RETRIEVAL",
    "hybrid_retriever": "RAG_RETRIEVAL",
    "vector_store": "RAG_RETRIEVAL",
    "retrieval_validator": "RAG_RETRIEVAL",
    "evaluation": "RAG_RETRIEVAL",
    "embeddings": "EMBEDDINGS",
    "web_fallback": "WEB_FETCH",
    "llm_client": "LLM_CALL",
    "column_describer": "LLM_CALL",
    "routing": "ROUTING",
    "spatial_search": "POSTGIS_QUERY",
    "dataset_lookup": "POSTGIS_QUERY",
    "geojson_validator": "VALIDATION",
    "document_loader": "INGEST",
    "chunker": "INGEST",
    "doc_ingest": "INGEST",
    "semantic_cache": "CACHE",
    "memory": "MEMORY",
    "guardrails": "GUARDRAILS",
    "api": "API",
}

_DUR_RE = re.compile(r"[—-]\s*([\d.]+)\s*ms")


def module_for(logger_name: str) -> str:
    suffix = (logger_name or "").split(".")[-1]
    return _MODULE_MAP.get(suffix, suffix.upper() or "APP")


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS logs (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 ts REAL, iso TEXT, module TEXT, logger TEXT, level TEXT,
                 message TEXT, target TEXT, duration_ms REAL, error_type TEXT, trace TEXT)"""
        )
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts)")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_module ON logs(module)")
        _conn.commit()
    return _conn


def init_log_db() -> None:
    """Create the store (call once at startup)."""
    with _lock:
        _connect()


class SQLiteLogHandler(logging.Handler):
    """Persist every log record into the ``logs`` table."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            duration = getattr(record, "ctx_duration_ms", None)
            target = getattr(record, "ctx_target", None)
            error_type = trace = None
            if record.exc_info and record.exc_info[0] is not None:
                error_type = record.exc_info[0].__name__
                trace = logging.Formatter().formatException(record.exc_info)
            if duration is None:
                m = _DUR_RE.search(msg)
                if m:
                    duration = float(m.group(1))
            iso = datetime.fromtimestamp(record.created, timezone.utc).astimezone().isoformat()
            with _lock:
                conn = _connect()
                conn.execute(
                    "INSERT INTO logs (ts,iso,module,logger,level,message,target,duration_ms,error_type,trace) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (record.created, iso, module_for(record.name), record.name,
                     record.levelname, msg, target, duration, error_type, trace),
                )
                conn.commit()
        except Exception:  # noqa: BLE001 — never let logging crash the app
            self.handleError(record)


_COLS = ["id", "iso", "module", "logger", "level", "message", "target",
         "duration_ms", "error_type", "trace"]


def query_logs(search: str | None = None, modules: list[str] | None = None,
               levels: list[str] | None = None, start: float | None = None,
               end: float | None = None, limit: int = 200, offset: int = 0) -> dict:
    """Filtered log query (filters combine with AND), newest first."""
    where: list[str] = []
    args: list = []
    if search:
        where.append("(message LIKE ? OR target LIKE ? OR logger LIKE ?)")
        like = f"%{search}%"
        args += [like, like, like]
    if modules:
        where.append("module IN (%s)" % ",".join("?" * len(modules)))
        args += modules
    if levels:
        where.append("level IN (%s)" % ",".join("?" * len(levels)))
        args += [l.upper() for l in levels]
    if start is not None:
        where.append("ts >= ?"); args.append(start)
    if end is not None:
        where.append("ts <= ?"); args.append(end)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    limit = max(1, min(int(limit), 100000))
    with _lock:
        conn = _connect()
        total = conn.execute(f"SELECT COUNT(*) FROM logs {clause}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT {','.join(_COLS)} FROM logs {clause} ORDER BY ts DESC LIMIT ? OFFSET ?",
            args + [limit, int(offset)],
        ).fetchall()
    return {"total": total, "count": len(rows),
            "logs": [dict(zip(_COLS, r)) for r in rows]}


def distinct_modules() -> list[str]:
    with _lock:
        conn = _connect()
        rows = conn.execute("SELECT DISTINCT module FROM logs ORDER BY module").fetchall()
    return [r[0] for r in rows]


def clear_logs() -> int:
    with _lock:
        conn = _connect()
        n = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        conn.execute("DELETE FROM logs")
        conn.commit()
    return n
