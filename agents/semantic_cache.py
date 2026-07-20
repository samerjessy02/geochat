"""
agents/semantic_cache.py — semantic (embedding-similarity) response cache.

Caches whole ``/query`` responses keyed on the **embedding of the query**, not
the raw string. So two differently-worded but semantically-equivalent questions
— "what is NBE's location" and "locate NBE" — resolve to the same cached answer
and skip the entire downstream pipeline (intent classification, SQL generation,
retrieval, LLM generation).

Why semantic and not keyword:
    An exact-match (hash-the-string) cache only hits on identical text, so it
    misses paraphrases — the common case for real users. Here we embed the query
    and compare it against previously-seen query embeddings with cosine
    similarity; anything at or above ``cache_threshold`` is a hit. The query is
    still embedded on every lookup (that's how similarity is measured), but a hit
    then skips the two expensive stages — retrieval and LLM generation.

Scope:
    Answers depend on *what* is being searched, so each entry is tagged with a
    ``scope`` (selected datasets + collection). A query only matches cached
    entries recorded under the same scope, so switching datasets never returns a
    stale answer from a different selection.

Freshness:
    Ingesting or deleting a document/dataset changes what the correct answer is,
    so those paths call :func:`clear`. An optional TTL provides a time-based
    safety net. Embeddings are L2-normalized upstream, so cosine similarity is a
    plain dot product.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from config import settings
from agents.logging_config import get_logger, snippet

log = get_logger("semantic_cache")


@dataclass
class _Entry:
    vector: list[float]
    scope: str
    value: dict
    query: str
    ts: float


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Vectors are pre-normalized, so this is a dot product
    (with a defensive re-normalization in case a caller passes raw vectors)."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    denom = (na ** 0.5) * (nb ** 0.5)
    # Fast path: already unit vectors -> denom ~ 1.0.
    return dot / denom if abs(denom - 1.0) > 1e-6 else dot


class SemanticCache:
    """In-memory nearest-neighbour cache over query embeddings."""

    def __init__(self, threshold: float, max_entries: int, ttl: int) -> None:
        self.threshold = threshold
        self.max_entries = max(1, max_entries)
        self.ttl = ttl
        self._entries: list[_Entry] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _is_fresh(self, entry: _Entry, now: float) -> bool:
        return self.ttl <= 0 or (now - entry.ts) <= self.ttl

    def get(self, vector: list[float], scope: str) -> tuple[dict, float, str] | None:
        """Return ``(value, similarity, matched_query)`` for the best in-scope
        entry at/above threshold, else ``None``. Expired entries are skipped."""
        now = time.time()
        with self._lock:
            best: _Entry | None = None
            best_sim = -1.0
            for e in self._entries:
                if e.scope != scope or not self._is_fresh(e, now):
                    continue
                sim = _cosine(vector, e.vector)
                if sim > best_sim:
                    best_sim, best = sim, e
            if best is not None and best_sim >= self.threshold:
                self.hits += 1
                best.ts = now  # LRU: refresh on hit
                return best.value, best_sim, best.query
            self.misses += 1
            return None

    def put(self, vector: list[float], scope: str, value: dict, query: str) -> None:
        """Store a response. Evicts the oldest entry when over capacity, and any
        expired entries encountered along the way."""
        now = time.time()
        with self._lock:
            if self.ttl > 0:
                self._entries = [e for e in self._entries if self._is_fresh(e, now)]
            self._entries.append(_Entry(vector=vector, scope=scope, value=value, query=query, ts=now))
            if len(self._entries) > self.max_entries:
                # drop least-recently-used (smallest ts)
                oldest = min(range(len(self._entries)), key=lambda i: self._entries[i].ts)
                self._entries.pop(oldest)

    def clear(self) -> int:
        """Forget everything (called when documents/datasets change). Returns the
        number of entries dropped."""
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
        if n:
            log.info("semantic cache cleared (%d entr%s)", n, "y" if n == 1 else "ies")
        return n

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "enabled": True,
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "threshold": self.threshold,
            }


# Process-wide singleton, configured from settings.
_cache = SemanticCache(
    threshold=settings.cache_threshold,
    max_entries=settings.cache_max_entries,
    ttl=settings.cache_ttl,
)


def get_cache() -> SemanticCache:
    return _cache


def scope_key(dataset_ids: list[str] | None, collection: str | None) -> str:
    """A stable tag for the retrieval scope an answer depends on."""
    ds = ",".join(sorted(dataset_ids or []))
    return f"ds=[{ds}]|col={collection or ''}"


def invalidate(reason: str = "") -> int:
    """Clear the cache because the underlying data changed."""
    if reason:
        log.info("invalidating semantic cache: %s", reason)
    return _cache.clear()
