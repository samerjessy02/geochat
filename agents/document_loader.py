"""
agents/document_loader.py — parse uploaded documents into pages via LangChain.

Uses LangChain Community document loaders so ingestion benefits from a
well-maintained, uniform parsing layer:

    PDF   -> PyPDFLoader        (one Document per page, page number preserved)
    DOCX  -> Docx2txtLoader
    TXT   -> TextLoader
    MD    -> TextLoader         (raw Markdown text; the splitter handles structure)
    HTML  -> BSHTMLLoader       (BeautifulSoup text extraction)
    CSV   -> CSVLoader          (one Document per row: "col: value" lines)

Each LangChain ``Document`` is normalized into a :class:`LoadedPage` so the rest
of the pipeline (chunker, ingestion, retrieval) is unchanged. LangChain loaders
read from a path, so the uploaded bytes are written to a short-lived temp file.

LangChain is imported lazily inside the loader functions, keeping module import
cheap and isolating the heavy dependency to the moment a document is parsed.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from agents.logging_config import get_logger

log = get_logger("document_loader")

SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt", "md", "markdown", "html", "htm", "csv"}


class DocumentLoadError(Exception):
    """Raised when a document cannot be parsed."""


@dataclass
class LoadedPage:
    """One logical page/section of a document."""

    page_number: int
    text: str
    section: str | None = None


def _ext(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def _langchain_documents(ext: str, path: str) -> list:
    """Dispatch to the appropriate LangChain loader and return its Documents."""
    from langchain_community.document_loaders import (
        BSHTMLLoader,
        CSVLoader,
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
    )

    if ext == "pdf":
        return PyPDFLoader(path).load()
    if ext == "docx":
        return Docx2txtLoader(path).load()
    if ext in ("txt", "md", "markdown"):
        return TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
    if ext in ("html", "htm"):
        return BSHTMLLoader(path, open_encoding="utf-8", get_text_separator=" ").load()
    if ext == "csv":
        return CSVLoader(path, encoding="utf-8").load()
    raise DocumentLoadError(f"Unsupported document type '.{ext}'.")


def _to_pages(docs: list) -> list[LoadedPage]:
    """Normalize LangChain Documents into :class:`LoadedPage` objects.

    Preserves a PDF's 0-indexed ``page`` metadata (shifted to 1-based); for
    formats without a page concept, positions are numbered sequentially.
    """
    pages: list[LoadedPage] = []
    for i, doc in enumerate(docs, start=1):
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        meta = getattr(doc, "metadata", {}) or {}
        page_no = meta.get("page")
        page_number = (page_no + 1) if isinstance(page_no, int) else i
        section = meta.get("category") or meta.get("section") or meta.get("title")
        pages.append(LoadedPage(page_number=page_number, text=text, section=section))
    return pages


def load_document(filename: str, raw: bytes) -> list[LoadedPage]:
    """Parse ``raw`` bytes of ``filename`` into a list of :class:`LoadedPage`.

    Writes the bytes to a temp file, runs the matching LangChain loader, and
    normalizes the result. Raises :class:`DocumentLoadError` for unsupported
    types or parse failures.
    """
    ext = _ext(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported document type '.{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    log.info("parsing '%s' (.%s, %d bytes) via LangChain loader", filename, ext, len(raw))
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        docs = _langchain_documents(ext, path)
    except DocumentLoadError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize loader-specific errors
        raise DocumentLoadError(f"Failed to parse '{filename}': {e}") from e
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    pages = _to_pages(docs)
    if not pages:
        raise DocumentLoadError(
            f"'{filename}' yielded no extractable text (a scanned PDF would need OCR)."
        )
    log.info("parsed '%s' into %d page(s)", filename, len(pages))
    return pages
