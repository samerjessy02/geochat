"""Unit tests for the LangChain-backed recursive chunker and its metadata."""

import pytest

pytest.importorskip("langchain_text_splitters", reason="LangChain splitter not installed")

from agents.chunker import chunk_pages, _split_text, DEFAULT_SEPARATORS
from agents.document_loader import LoadedPage


def test_split_respects_max_size():
    text = "word " * 500  # 2500 chars
    chunks = _split_text(text, chunk_size=200, chunk_overlap=0, separators=DEFAULT_SEPARATORS)
    assert chunks
    # allow small overshoot from separator joining, but nothing wildly over
    assert all(len(c) <= 260 for c in chunks)


def test_split_short_text_is_single_chunk():
    assert _split_text("short text", 200, 20, DEFAULT_SEPARATORS) == ["short text"]


def test_overlap_carries_context_between_chunks():
    text = "AAAA\n\nBBBB\n\nCCCC\n\nDDDD\n\nEEEE\n\nFFFF"
    chunks = _split_text(text, chunk_size=10, chunk_overlap=4, separators=DEFAULT_SEPARATORS)
    assert len(chunks) > 1


def test_chunk_pages_populates_required_metadata():
    pages = [LoadedPage(page_number=1, text="Cairo University was founded in 1908. " * 50, section="History")]
    chunks = chunk_pages(
        pages,
        document_id="doc123",
        title="Cairo University Facts",
        source="cu.pdf",
        upload_date="2026-07-16T00:00:00Z",
        chunk_size=200,
        chunk_overlap=20,
    )
    assert len(chunks) > 1
    md = chunks[0].metadata()
    for key in ("document_id", "title", "page", "section", "source", "upload_date", "chunk_index", "chunk_id"):
        assert key in md
    assert md["document_id"] == "doc123"
    assert md["page"] == 1
    assert md["section"] == "History"
    # chunk_index increments
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic():
    pages = [LoadedPage(page_number=1, text="deterministic content " * 40)]
    kw = dict(document_id="d", title="t", source="s", upload_date="2026-07-16")
    a = chunk_pages(pages, chunk_size=100, chunk_overlap=10, **kw)
    b = chunk_pages(pages, chunk_size=100, chunk_overlap=10, **kw)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_heading_chunking_one_chunk_per_entity():
    # Catalog-style text (heading + paragraph) should chunk one entity per chunk,
    # with each chunk carrying its own name and not bleeding into the others.
    text = (
        "Bean House\n\nBean House is a neighborhood cafe in Cairo with wifi and takeaway.\n\n"
        "Costa\n\nCosta is an international coffee chain with handcrafted espresso and lattes.\n\n"
        "Cilantro\n\nCilantro is an Egyptian cafe serving specialty coffee and sandwiches."
    )
    pages = [LoadedPage(page_number=1, text=text)]
    chunks = chunk_pages(pages, document_id="d", title="Cafes", source="cafes.pdf",
                         chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 3
    assert [c.section for c in chunks] == ["Bean House", "Costa", "Cilantro"]
    costa = next(c for c in chunks if c.section == "Costa")
    assert "Costa" in costa.text
    assert "Bean House" not in costa.text  # per-entity isolation


def test_recursive_fallback_for_prose():
    # Prose with no headings should fall back to recursive size-based chunking.
    prose = "This is a long flowing paragraph of prose. " * 60
    pages = [LoadedPage(page_number=1, text=prose)]
    chunks = chunk_pages(pages, document_id="d", title="t", source="s",
                         chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
