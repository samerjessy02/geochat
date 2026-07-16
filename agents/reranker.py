"""
agents/reranker.py — optional cross-encoder reranking.

RRF gives a strong first-stage ranking, but a cross-encoder (which jointly
encodes the query and each passage) is a more accurate final-stage judge. It is
slower, so it only runs on the small fused candidate set and only when
``RERANKER_ENABLED`` is true.

The model is loaded lazily and cached. If sentence-transformers/torch are not
installed, :func:`rerank` degrades gracefully by returning the input order.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from config import settings
from agents.logging_config import get_logger

log = get_logger("reranker")

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _get_cross_encoder() -> "CrossEncoder":
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def rerank(query: str, hits: list[dict], *, top_k: int | None = None) -> list[dict]:
    """Reorder ``hits`` by cross-encoder relevance to ``query``.

    Adds a ``rerank_score`` to each hit. Returns the top ``top_k`` (or all).
    On any failure (e.g. missing dependency) the original order is preserved so
    reranking never becomes a hard dependency of retrieval.
    """
    if not hits:
        return hits
    try:
        model = _get_cross_encoder()
        pairs = [(query, h.get("text", "")) for h in hits]
        scores = model.predict(pairs)
    except Exception as e:  # noqa: BLE001 — reranking is best-effort
        log.warning("reranker unavailable (%s) — keeping fused order", e)
        return hits[: top_k or len(hits)]

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    reranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return reranked[: top_k or len(reranked)]
