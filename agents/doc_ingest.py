"""
agents/doc_ingest.py — orchestrate document ingestion end-to-end.

Pipeline (matches the spec's "Document Processing" section):

    load_document -> chunk_pages -> embed_passages -> vector_store.upsert_chunks
                                                   \-> record document metadata in Postgres

A lightweight ``rag_documents`` table in Postgres holds one row per uploaded
document (id, title, source, filetype, page/chunk counts, upload date). The
per-chunk metadata lives in the Qdrant payload; this table just gives the API a
cheap way to list/manage documents and enforce collection filtering.

An in-document prompt-injection screen runs before anything is embedded, so
instructions hidden inside an uploaded file (``"ignore previous instructions"``
and friends) are flagged and stripped rather than silently indexed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import text

from db import engine
from agents import document_loader, embeddings, vector_store
from agents.chunker import chunk_pages
from agents.guardrails import scan_document_text
from agents.logging_config import get_logger, log_step

log = get_logger("doc_ingest")

_DDL = """
CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    filetype TEXT NOT NULL,
    collection TEXT,
    page_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    upload_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    version INTEGER NOT NULL DEFAULT 1
);
"""


class IngestError(Exception):
    """Raised when a document cannot be ingested."""


@dataclass
class IngestResult:
    document_id: str
    title: str
    source: str
    filetype: str
    page_count: int
    chunk_count: int
    upload_date: str
    injection_flags: list[str] = field(default_factory=list)


def init_documents_table() -> None:
    """Create the ``rag_documents`` table (call once at startup)."""
    with engine.begin() as conn:
        conn.execute(text(_DDL))


def ingest_document(
    filename: str,
    raw: bytes,
    *,
    title: str | None = None,
    collection: str | None = None,
) -> IngestResult:
    """Parse, chunk, embed and store a single uploaded document.

    ``collection`` is an optional logical grouping (used for metadata-filtered
    retrieval). Returns an :class:`IngestResult` including any prompt-injection
    strings that were detected and stripped from the source text.
    """
    filetype = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    doc_title = title or filename
    document_id = str(uuid.uuid4())
    log.info("ingest start: '%s' (id=%s, collection=%s)", doc_title, document_id, collection)

    try:
        with log_step(log, f"parse '{filename}'"):
            pages = document_loader.load_document(filename, raw)
    except document_loader.DocumentLoadError as e:
        raise IngestError(str(e)) from e

    # --- Retrieval/prompt-injection guardrail on the raw document text -------
    injection_flags: list[str] = []
    cleaned_pages = []
    for page in pages:
        safe_text, flags = scan_document_text(page.text)
        injection_flags.extend(flags)
        page.text = safe_text
        if safe_text.strip():
            cleaned_pages.append(page)
    if injection_flags:
        log.warning("guardrail neutralized %s in '%s'", sorted(set(injection_flags)), filename)
    if not cleaned_pages:
        raise IngestError("Document had no content left after safety filtering.")

    upload_date = datetime.now(timezone.utc).isoformat()

    chunks = chunk_pages(
        cleaned_pages,
        document_id=document_id,
        title=doc_title,
        source=filename,
        upload_date=upload_date,
        extra={"collection": collection} if collection else None,
    )
    if not chunks:
        raise IngestError("Document produced no chunks.")
    log.info("chunked into %d chunk(s)", len(chunks))

    with log_step(log, f"embed {len(chunks)} chunk(s) with {embeddings.settings.embedding_model}"):
        vectors = embeddings.embed_passages([c.text for c in chunks])
    with log_step(log, "upsert to Qdrant"):
        stored = vector_store.upsert_chunks(chunks, vectors)

    page_count = max((p.page_number for p in cleaned_pages), default=0)
    log.info("ingest done: '%s' — %d chunk(s) across %d page(s) stored", doc_title, stored, page_count)
    with engine.begin() as conn:
        conn.execute(
            text(
                """INSERT INTO rag_documents
                       (id, title, source, filetype, collection, page_count, chunk_count, upload_date)
                   VALUES (:id, :title, :source, :filetype, :collection, :page_count, :chunk_count, :upload_date)"""
            ),
            {
                "id": document_id,
                "title": doc_title,
                "source": filename,
                "filetype": filetype,
                "collection": collection,
                "page_count": page_count,
                "chunk_count": stored,
                "upload_date": upload_date,
            },
        )

    return IngestResult(
        document_id=document_id,
        title=doc_title,
        source=filename,
        filetype=filetype,
        page_count=page_count,
        chunk_count=stored,
        upload_date=upload_date,
        injection_flags=sorted(set(injection_flags)),
    )


def list_documents() -> list[dict]:
    """Return metadata for all ingested documents (newest first)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """SELECT id, title, source, filetype, collection, page_count, chunk_count,
                          upload_date, version
                   FROM rag_documents ORDER BY upload_date DESC"""
            )
        )
        return [dict(r._mapping) for r in rows]


def delete_document(document_id: str) -> bool:
    """Remove a document from both the Qdrant index and Postgres metadata.

    Deletes the vectors FIRST so that if Qdrant deletion fails, the metadata row
    is kept (the document stays listed and the user can retry) rather than being
    orphaned with dangling chunks.
    """
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id FROM rag_documents WHERE id = :id"), {"id": document_id}
        ).fetchone()
    if not row:
        return False

    vector_store.delete_by_document(document_id)   # remove chunks first
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM rag_documents WHERE id = :id"), {"id": document_id})
    return True


def purge_orphan_chunks() -> dict:
    """Delete Qdrant chunks whose document is no longer registered in Postgres."""
    with engine.connect() as conn:
        known = {str(r[0]) for r in conn.execute(text("SELECT id FROM rag_documents"))}
    return vector_store.delete_orphans(known)
