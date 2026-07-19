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
    _ensure_payload_indexes(client)


def _ensure_payload_indexes(client) -> None:
    """Create keyword payload indexes so metadata filters (document_id, etc.) work.

    Qdrant rejects filtering/deleting by a payload field that has no index
    ("Index required but not found for ..."). This runs idempotently on both new
    and existing collections; already-indexed fields raise and are ignored.
    """
    try:
        from qdrant_client.models import PayloadSchemaType
    except Exception:  # noqa: BLE001
        return
    for field in ("document_id", "collection", "section", "chunk_id"):
        try:
            client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:  # noqa: BLE001 — already exists / non-fatal
            pass


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


def chunks_of_document(document_id: str, collection: str | None = None) -> list[dict]:
    """Return all stored chunk payloads for ``document_id`` (filtered client-side).

    Filtering happens in Python rather than server-side, so it works regardless
    of whether a Qdrant payload index exists on ``document_id``. Fine for the
    small per-document scans used by exhaustive queries.
    """
    out: list[dict] = []
    for p in scroll_all():
        if p.get("document_id") != document_id:
            continue
        if collection and p.get("collection") != collection:
            continue
        out.append(p)
    return out


def delete_by_document(document_id: str) -> None:
    """Delete all chunks belonging to ``document_id``.

    Deletes by point ID (recomputed from each chunk's ``chunk_id``) rather than
    a server-side payload filter — so it works even when no payload index exists
    on ``document_id`` (Qdrant rejects filter-deletes without one).
    """
    from qdrant_client.models import PointIdsList

    ids = [
        _point_id(p["chunk_id"])
        for p in chunks_of_document(document_id)
        if p.get("chunk_id")
    ]
    if not ids:
        log.info("no chunks to delete for document_id=%s", document_id)
        return
    get_client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=PointIdsList(points=ids),
    )
    log.info("deleted %d chunk(s) for document_id=%s", len(ids), document_id)


def delete_orphans(known_document_ids: set[str]) -> dict:
    """Delete chunks whose ``document_id`` is not in ``known_document_ids``.

    Used to clean up vectors left behind when a document's metadata row was
    removed but its chunks weren't. Deletes by point ID (no index needed).
    """
    from qdrant_client.models import PointIdsList

    ids: list[str] = []
    docs: set[str] = set()
    for p in scroll_all():
        did = p.get("document_id")
        if did and did not in known_document_ids and p.get("chunk_id"):
            ids.append(_point_id(p["chunk_id"]))
            docs.add(did)
    if ids:
        get_client().delete(
            collection_name=settings.qdrant_collection,
            points_selector=PointIdsList(points=ids),
        )
    log.info("purged %d orphan chunk(s) across %d document(s)", len(ids), len(docs))
    return {"orphan_documents": len(docs), "chunks_deleted": len(ids)}


def count() -> int:
    """Return the number of points currently stored (0 if collection absent)."""
    try:
        return get_client().count(collection_name=settings.qdrant_collection).count
    except Exception:  # noqa: BLE001 — collection may not exist yet
        return 0
