"""
agents/hybrid_retriever.py — true hybrid search (dense + BM25 + RRF).

Combines two complementary retrievers and fuses their rankings with weighted
Reciprocal Rank Fusion (RRF):

  * Dense: cosine similarity over multilingual-e5 embeddings (Qdrant).
  * Sparse: BM25 keyword scoring (``rank_bm25``) over the same corpus.

RRF is rank-based rather than score-based, so it is robust to the two
retrievers producing scores on totally different scales. Weights
(``DENSE_WEIGHT`` / ``BM25_WEIGHT``) let you bias toward one retriever, and
``RRF_K`` damps the influence of very high ranks.

The fusion math (:func:`reciprocal_rank_fusion`) is a pure function with no I/O
so it can be unit-tested in isolation. The BM25 corpus is lazily built from the
vector store and cached; :func:`invalidate_bm25_cache` should be called after
ingesting new documents.

Each returned hit carries ``dense_score``, ``bm25_score`` and the fused
``score`` so callers can inspect why a chunk ranked where it did.
"""

from __future__ import annotations

import re
from functools import lru_cache

from config import settings
from agents.logging_config import get_logger, snippet

log = get_logger("hybrid_retriever")

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Unicode-aware lowercase tokenizer (keeps Arabic and Latin word runs)."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse several ranked id-lists into one, using weighted RRF.

    Args:
        ranked_lists: each inner list is ids ordered best-first from one retriever.
        weights: per-list weight (defaults to 1.0 each). Must match ``ranked_lists``.
        k: RRF damping constant; contribution of rank ``r`` is ``weight / (k + r)``.

    Returns:
        ``[(id, fused_score), ...]`` sorted by descending fused score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists length")

    scores: dict[str, float] = {}
    for ranking, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[list[dict], object]:
    """Build (and cache) the BM25 index over every stored chunk.

    Returns the list of chunk payloads and a fitted ``BM25Okapi`` index aligned
    by position. Cached until :func:`invalidate_bm25_cache` is called.
    """
    from rank_bm25 import BM25Okapi

    from agents import vector_store

    corpus = list(vector_store.scroll_all())
    tokenized = [tokenize(c.get("text", "")) for c in corpus]
    # BM25Okapi requires a non-empty corpus; guard with a sentinel token.
    bm25 = BM25Okapi(tokenized or [["__empty__"]])
    log.info("built BM25 keyword index over %d chunk(s)", len(corpus))
    return corpus, bm25


def invalidate_bm25_cache() -> None:
    """Drop the cached BM25 corpus so the next search rebuilds it (call after ingest)."""
    _load_corpus.cache_clear()
    log.debug("BM25 cache invalidated — will rebuild on next retrieval")


def _bm25_search(query: str, top_k: int) -> list[dict]:
    corpus, bm25 = _load_corpus()
    if not corpus:
        return []
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits = []
    for i in ranked:
        if scores[i] <= 0:
            continue
        payload = corpus[i]
        hits.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source") or payload.get("title", ""),
                "bm25_score": float(scores[i]),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
            }
        )
    return hits


def _chunk_key(hit: dict) -> str:
    """Stable identity for a hit across retrievers (chunk_id, else text hash)."""
    md = hit.get("metadata", {})
    return md.get("chunk_id") or str(hash(hit.get("text", "")))


def hybrid_search(
    query: str,
    *,
    top_k: int | None = None,
    candidate_k: int | None = None,
    filters: dict | None = None,
    rerank: bool | None = None,
) -> list[dict]:
    """Run dense + BM25 retrieval and fuse with RRF.

    Args:
        query: natural-language query.
        top_k: number of final results (default ``RETRIEVAL_TOP_K``).
        candidate_k: pool size fetched from each retriever before fusion.
        filters: optional metadata filter (e.g. ``{"collection": "policies"}``).
        rerank: force cross-encoder reranking on/off (default: ``RERANKER_ENABLED``).

    Returns:
        Fused, ranked hit dicts with ``dense_score``, ``bm25_score``, ``score``.
    """
    from agents import vector_store

    final_k = top_k or settings.retrieval_top_k
    pool = candidate_k or settings.retrieval_candidate_k
    log.info("hybrid retrieval for '%s' (pool=%d, top_k=%d)", snippet(query), pool, final_k)

    dense_hits = vector_store.similarity_search(query, top_k=pool, filters=filters)
    for h in dense_hits:
        h["dense_score"] = h.pop("score", 0.0)
    bm25_hits = _bm25_search(query, top_k=pool)
    log.debug("dense=%d hit(s), bm25=%d hit(s)", len(dense_hits), len(bm25_hits))

    # Index hits by identity and record each retriever's ranking.
    by_key: dict[str, dict] = {}
    dense_ranking, bm25_ranking = [], []
    for h in dense_hits:
        key = _chunk_key(h)
        by_key.setdefault(key, {}).update(h)
        by_key[key].setdefault("bm25_score", 0.0)
        dense_ranking.append(key)
    for h in bm25_hits:
        key = _chunk_key(h)
        entry = by_key.setdefault(key, {})
        entry.update({k: v for k, v in h.items() if k not in entry or k == "bm25_score"})
        entry.setdefault("dense_score", 0.0)
        bm25_ranking.append(key)

    fused = reciprocal_rank_fusion(
        [dense_ranking, bm25_ranking],
        weights=[settings.dense_weight, settings.bm25_weight],
        k=settings.rrf_k,
    )

    results = []
    for key, fused_score in fused:
        hit = by_key.get(key, {})
        hit["score"] = fused_score
        hit.setdefault("dense_score", 0.0)
        hit.setdefault("bm25_score", 0.0)
        results.append(hit)

    do_rerank = settings.reranker_enabled if rerank is None else rerank
    if do_rerank and results:
        from agents.reranker import rerank as _rerank

        log.info("reranking %d fused candidate(s) with cross-encoder", len(results))
        results = _rerank(query, results, top_k=final_k)
        log.info("hybrid retrieval returned %d result(s) (reranked)", len(results))
        return results

    final = results[:final_k]
    log.info("hybrid retrieval returned %d result(s) (RRF fused)", len(final))
    return final
