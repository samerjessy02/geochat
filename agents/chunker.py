"""
agents/chunker.py — heading-aware + recursive chunking with provenance metadata.

Two strategies, selected by ``CHUNK_STRATEGY`` (see ``config.py``):

* **heading/section** (default via ``auto``) — for catalog-style documents where
  each record is a short heading (an entity name) followed by a descriptive
  paragraph (e.g. the "Bean House / Costa / Cilantro" cafés PDF). Each section
  becomes exactly one chunk, and the heading (entity name) is prepended to the
  chunk text ("contextual chunking") so both dense and BM25 retrieval always see
  the name. This keeps per-entity questions precise — a query about "Costa"
  retrieves only Costa's chunk, not a blob of several cafés. A section larger
  than ``chunk_size`` is further split with the recursive splitter, with the
  heading re-attached to every sub-chunk.

* **recursive** — LangChain's ``RecursiveCharacterTextSplitter`` (paragraph →
  line → sentence → word → char) with overlap, for prose/unstructured documents.

``auto`` uses heading chunking when a page has ≥2 detected sections, otherwise
falls back to recursive — so structured catalogs and free-form prose both chunk
well without manual configuration.

Public surface kept stable:
    DEFAULT_SEPARATORS   separator ladder (Latin + Arabic punctuation)
    _split_text(...)     LangChain-backed recursive splitter (also used by web_fallback)
    chunk_pages(...)     LoadedPage[] -> Chunk[] with metadata
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

from config import settings
from agents.document_loader import LoadedPage
from agents.logging_config import get_logger

log = get_logger("chunker")

# Ordered coarsest → finest; includes Arabic sentence punctuation for bilingual docs.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "؟ ", "? ", "؛ ", "; ", ", ", " ", ""]

# Heading heuristics (for section chunking).
_HEADING_MAX_CHARS = 64
_HEADING_MAX_WORDS = 8
_TERMINAL_PUNCT = (".", "!", "?", ":", ";", ",", "،", "؛")


@dataclass
class Chunk:
    """A single retrievable unit of text plus its provenance metadata."""

    text: str
    chunk_index: int
    document_id: str
    title: str
    source: str
    page: int
    section: str | None = None
    upload_date: str = ""
    chunk_id: str = ""
    extra: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        """Return the flat metadata payload stored alongside the vector."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "upload_date": self.upload_date,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            **self.extra,
        }


# --------------------------------------------------------------------------- #
# recursive splitter (LangChain)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=8)
def _get_splitter(chunk_size: int, chunk_overlap: int, separators: tuple[str, ...]):
    """Build (and cache) a LangChain RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
        keep_separator=True,
        length_function=len,
        is_separator_regex=False,
    )


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Split ``text`` into overlapping pieces using LangChain's recursive splitter.

    Kept standalone (historical signature) because the web fallback also chunks
    scraped pages through it.
    """
    text = (text or "").strip()
    if not text:
        return []
    seps = tuple(separators or DEFAULT_SEPARATORS)
    return [c for c in _get_splitter(chunk_size, chunk_overlap, seps).split_text(text) if c.strip()]


# --------------------------------------------------------------------------- #
# heading / section detection
# --------------------------------------------------------------------------- #

def _looks_like_heading(line: str) -> bool:
    """Heuristic: is ``line`` a section heading (a short entity-name title)?

    A heading is a short line (few words, no terminal punctuation) that starts
    with an uppercase letter — e.g. "Bean House", "Costa", "Cilantro". Full
    sentences (which end in a period, are long, or start lowercase) are not
    headings.
    """
    s = line.strip()
    if not s or len(s) > _HEADING_MAX_CHARS:
        return False
    if len(s.split()) > _HEADING_MAX_WORDS:
        return False
    if s.endswith(_TERMINAL_PUNCT):
        return False
    first_alpha = next((ch for ch in s if ch.isalpha()), "")
    if not first_alpha:
        return False
    # Arabic has no case; accept it. For cased scripts, require an uppercase start.
    if first_alpha.isascii() and not first_alpha.isupper():
        return False
    return True


def _heading_sections(text: str) -> list[tuple[str, str]]:
    """Segment ``text`` into ``(heading, body)`` sections.

    Only sections that have a non-empty body are returned, so a stray title with
    no following paragraph doesn't create an empty chunk.
    """
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    body_lines: list[str] = []

    def flush() -> None:
        nonlocal body_lines, current_heading
        if current_heading is not None:
            body = "\n".join(body_lines).strip()
            if body:
                sections.append((current_heading, body))
        body_lines = []

    for line in text.splitlines():
        if _looks_like_heading(line):
            flush()
            current_heading = line.strip()
        elif line.strip() or body_lines:
            body_lines.append(line)
    flush()
    return sections


def _with_context(heading: str, text: str) -> str:
    """Prepend the heading to ``text`` unless it already leads with it."""
    text = text.strip()
    if text.lower().startswith(heading.lower()):
        return text
    return f"{heading} — {text}"


def _section_units(text: str, size: int, overlap: int) -> list[tuple[str, str]] | None:
    """Return ``[(chunk_text, heading), ...]`` for catalog-style ``text``.

    Returns ``None`` when the text doesn't look like a catalog (< 2 sections), so
    the caller can fall back to recursive chunking.
    """
    sections = _heading_sections(text)
    if len(sections) < 2:
        return None
    units: list[tuple[str, str]] = []
    for heading, body in sections:
        unit = _with_context(heading, body)
        if len(unit) <= size:
            units.append((unit, heading))
        else:
            # Oversized section: split the body, re-attach the heading to each piece.
            for piece in _split_text(body, size, overlap, DEFAULT_SEPARATORS):
                units.append((_with_context(heading, piece), heading))
    return units


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def _units_for_page(page: LoadedPage, size: int, overlap: int) -> list[tuple[str, str | None]]:
    """Produce ``[(text, section), ...]`` for one page per the configured strategy."""
    strategy = settings.chunk_strategy
    if strategy in ("auto", "heading"):
        units = _section_units(page.text, size, overlap)
        if units is not None:
            return units  # list[(text, heading)]
        # strategy == "heading" but nothing detected -> fall back to recursive too.
    return [(piece, page.section) for piece in _split_text(page.text, size, overlap, DEFAULT_SEPARATORS)]


def chunk_pages(
    pages: list[LoadedPage],
    *,
    document_id: str,
    title: str,
    source: str,
    upload_date: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    extra: dict | None = None,
) -> list[Chunk]:
    """Split every :class:`LoadedPage` into :class:`Chunk` objects with metadata.

    Uses heading/section chunking for catalog-style pages (each entity → one
    chunk, name prepended) and recursive size-based chunking otherwise. Each
    ``chunk_id`` is derived from the document id, page, and a content hash so
    re-ingesting identical content overwrites in place rather than duplicating.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    upload = upload_date or datetime.now(timezone.utc).isoformat()

    chunks: list[Chunk] = []
    index = 0
    section_pages = 0
    for page in pages:
        units = _units_for_page(page, size, overlap)
        # Track whether this page used heading sections (section differs from page.section).
        if units and any(sec and sec != page.section for _, sec in units):
            section_pages += 1
        for piece, section in units:
            digest = hashlib.sha1(f"{document_id}:{page.page_number}:{piece}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    text=piece,
                    chunk_index=index,
                    document_id=document_id,
                    title=title,
                    source=source,
                    page=page.page_number,
                    section=section,
                    upload_date=upload,
                    chunk_id=f"{document_id}-{digest}",
                    extra=dict(extra or {}),
                )
            )
            index += 1

    log.info(
        "chunked '%s' -> %d chunk(s) [%s]",
        title, len(chunks),
        f"heading sections on {section_pages}/{len(pages)} page(s)" if section_pages else "recursive",
    )
    return chunks
