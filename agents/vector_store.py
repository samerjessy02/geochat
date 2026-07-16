"""
agents/vector_store.py — Qdrant persistence + dense similarity search.

Stores one point per chunk: the dense embedding as the vector, and the chunk
text plus its full provenance metadata as the payload. Exposes:

    ensure_collection()                    idempotent collection creation
    upsert_chunks(chunks, vectors)         persist embedded chunks
    dense_search(vector, top_k, filters)   vector search -> list[Hit]
    similarity_search(query, top_k, ...)   embed query + dense_search (convenience)
    scroll_all(filters)                    stream every stored chunk (for BM25)
    delete_by_document(document_id)        remove a document's chunks
    count()                                number of stored points

``similarity_search`` returns dicts shaped ``{text, source, score, metadata}``
which the existing ``rag_agent.py`` already consumes.

The Qdrant client is created lazily and cached, so importing this module does
not require a running Qdrant unless a call is actually made.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any, Iterator

from config import settings
from agents.logging_config import get_logger

log = get_logger("vector_store")

# Deterministic namespace so the same chunk_id always maps to the same point id.
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


@lru_cache(maxsize=1)
def get_client():
    """Return a cached QdrantClient built from configuration.

    A generous request ``timeout`` is set because writing batches of 1024-dim
    vectors can exceed the client's 5s default (which surfaces as
    "write operation timed out").
    """
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )


def ensure_collection() -> None:
    """Create the configured collection if it does not already exist (idempotent)."""
    from qdrant_client.models import Distance, VectorParams

    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
        log.info("created Qdrant collection '%s' (dim=%d, cosine)",
                 settings.qdrant_collection, settings.embedding_dim)


def _build_filter(filters: dict[str, Any] | None):
    """Translate a flat ``{field: value|list}`` dict into a Qdrant filter."""
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions) if conditions else None


def upsert_chunks(chunks: list, vectors: list[list[float]]) -> int:
    """Persist ``chunks`` (list of :class:`agents.chunker.Chunk`) with ``vectors``.

    Returns the number of points written. Point ids are derived from each
    ``chunk_id`` so re-ingesting identical content overwrites in place rather
    than duplicating.
    """
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must be the same length")
    if not chunks:
        return 0

    from qdrant_client.models import PointStruct

    ensure_collection()
    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {"text": chunk.text, **chunk.metadata()}
        points.append(PointStruct(id=_point_id(chunk.chunk_id), vector=vector, payload=payload))

    client = get_client()
    batch = max(1, settings.qdrant_upsert_batch)
    # Upsert in batches so a single large write can't exceed the socket timeout.
    for i in range(0, len(points), batch):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i : i + batch],
            wait=True,
        )
    log.info("upserted %d point(s) into '%s' (batch=%d)", len(points), settings.qdrant_collection, batch)
    return len(points)


def _hit_to_dict(payload: dict, score: float) -> dict:
    return {
        "text": payload.get("text", ""),
        "source": payload.get("source") or payload.get("title", ""),
        "score": float(score),
        "metadata": {k: v for k, v in payload.items() if k != "text"},
    }


def dense_search(
    query_vector: list[float],
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
) -> list[dict]:
    """Cosine-similarity search over stored chunks. Returns ranked hit dicts."""
    client = get_client()
    k = top_k or settings.retrieval_top_k
    qfilter = _build_filter(filters)

    # qdrant-client >= 1.12 removed the old `.search()` in favour of
    # `.query_points()`. Prefer the new API, fall back for older clients.
    if hasattr(client, "query_points"):
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=k,
            query_filter=qfilter,
            with_payload=True,
        ).points
    else:  # pragma: no cover — legacy client
        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=k,
            query_filter=qfilter,
            with_payload=True,
        )

    hits = [_hit_to_dict(r.payload or {}, r.score) for r in results]
    top = hits[0]["score"] if hits else 0.0
    log.debug("dense search: %d/%d hit(s), top score=%.3f%s",
              len(hits), k, top, f", filters={filters}" if filters else "")
    return hits


def similarity_search(
    query: str,
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
) -> list[dict]:
    """Embed ``query`` and run :func:`dense_search`. Convenience for callers
    (e.g. the web-fallback agent) that only need dense retrieval."""
    from agents.embeddings import embed_query

    return dense_search(embed_query(query), top_k=top_k, filters=filters)


def scroll_all(filters: dict[str, Any] | None = None, batch: int = 256) -> Iterator[dict]:
    """Yield every stored chunk's payload (text + metadata). Used to (re)build
    the BM25 keyword index, which needs the full corpus."""
    client = get_client()
    offset = None
    qfilter = _build_filter(filters)
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=qfilter,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            yield p.payload or {}
        if offset is None:
            break


def delete_by_document(document_id: str) -> None:
    """Delete all chunks belonging to ``document_id``."""
    from qdrant_client.models import FilterSelector

    client = get_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=FilterSelector(filter=_build_filter({"document_id": document_id})),
    )
    log.info("deleted all chunks for document_id=%s", document_id)


def count() -> int:
    """Return the number of points currently stored (0 if collection absent)."""
    try:
        return get_client().count(collection_name=settings.qdrant_collection).count
    except Exception:  # noqa: BLE001 — collection may not exist yet
        return 0
