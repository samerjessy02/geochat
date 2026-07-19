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

from config import settings
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


# --------------------------------------------------------------------------- #
# PDF: pypdf first, Docling fallback for scanned / complex layouts
# --------------------------------------------------------------------------- #

# pypdf emits one text line per *visual* line, so a wrapped paragraph arrives as
# several short lines — sometimes a single word ("The", "Customers") on its own
# line. Left as-is, the heading-aware chunker mistakes those fragments for
# section headings. Reflow stitches wrapped lines back into whole paragraphs so
# only *real* headings (an entity name on its own line) survive as separate lines.

_TERMINAL_PUNCT = (".", "!", "?", ":", ";", "،", "؛", '."', '!"', '?"', ".)", '.”')
# Words that are never a standalone heading — a genuine heading (an entity name)
# has at least one distinctive word, so a short line made up only of these is a
# wrapped sentence fragment and must be joined to its paragraph, not kept apart.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for", "at",
    "by", "with", "as", "so", "our", "its", "it", "is", "are", "was", "were", "this",
    "that", "these", "those", "from", "here", "there", "their", "they", "we", "you",
    "he", "she", "his", "her", "your", "whether", "however", "meanwhile", "also",
    "then", "when", "while", "each", "every", "both", "either", "neither", "who",
    "what", "which", "how", "customers", "visitors", "whatever", "whenever",
    "guests", "clients", "people", "everyone", "they", "them",
}


def _reflow_is_heading(line: str) -> bool:
    """Would ``line`` (alone) read as a real section heading, not a wrapped fragment?

    Short, title-cased, no terminal punctuation, and carrying at least one
    distinctive (non-stopword) token — e.g. "Bean House", "Cilantro". A bare
    "The" or "Customers" fails the stopword test and stays part of its paragraph.
    """
    s = line.strip()
    if not s or len(s) > 64 or len(s.split()) > 8:
        return False
    if s.endswith(_TERMINAL_PUNCT):
        return False
    # A heading is a bare title — it carries no internal sentence break.
    if any(p in s for p in (". ", "! ", "? ", "; ", "، ", "؛ ")):
        return False
    first_alpha = next((ch for ch in s if ch.isalpha()), "")
    if not first_alpha:
        return False
    if first_alpha.isascii() and not first_alpha.isupper():
        return False
    words = [w.strip(".,;:!?'\"()[]").lower() for w in s.split()]
    if words and all(w in _STOPWORDS or not w for w in words):
        return False
    return True


def _reflow_text(text: str) -> str:
    """Rejoin pypdf's wrapped lines into paragraphs, preserving real headings.

    A blank line or a heading-looking line starts a new block; every other line
    is appended to the current paragraph (de-hyphenating words split across a
    line break). Output is blocks separated by blank lines, which the
    heading-aware chunker then segments cleanly.
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(" ".join(buf).strip())
            buf.clear()

    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            flush()
            continue
        if _reflow_is_heading(line):
            flush()
            out.append(line)
            continue
        if buf and buf[-1].endswith("-"):
            buf[-1] = buf[-1][:-1] + line   # de-hyphenate split word
        else:
            buf.append(line)
    flush()

    # Second pass: pypdf may still isolate a bare fragment (e.g. a lone "The"
    # left on its own line by a spurious blank line). A fragment is any block
    # that is NOT a heading and does not end a sentence — carry it forward and
    # fold it into the following block so it never stands alone as a "heading".
    merged: list[str] = []
    carry = ""
    for block in out:
        if _reflow_is_heading(block):
            if carry:
                merged.append(carry)
                carry = ""
            merged.append(block)
            continue
        carry = f"{carry} {block}".strip() if carry else block
        if carry.rstrip().endswith(_TERMINAL_PUNCT):
            merged.append(carry)
            carry = ""
    if carry:
        merged.append(carry)
    return "\n\n".join(b for b in merged if b)


def _pdf_pypdf(path: str) -> list[LoadedPage]:
    """Extract a PDF with pypdf (via LangChain PyPDFLoader) — fast, no models.

    pypdf's raw output breaks paragraphs at every visual line; ``_reflow_text``
    stitches them back together so the chunker sees whole paragraphs and only
    real headings.
    """
    from langchain_community.document_loaders import PyPDFLoader

    docs = PyPDFLoader(path).load()
    for doc in docs:
        doc.page_content = _reflow_text(getattr(doc, "page_content", "") or "")
    return _to_pages(docs)


def _pdf_docling(path: str) -> list[LoadedPage]:
    """Extract a PDF with Docling (layout-aware, OCR) -> Markdown as one page.

    Docling emits clean Markdown with real headings, which also feeds the
    heading-aware chunker well. Requires the optional ``docling`` package.
    """
    from docling.document_converter import DocumentConverter  # lazy, heavy

    result = DocumentConverter().convert(path)
    text = (result.document.export_to_markdown() or "").strip()
    return [LoadedPage(page_number=1, text=text)] if text else []


def _pypdf_ok(pages: list[LoadedPage]) -> bool:
    """Quality gate: is pypdf's extraction good enough, or should we escalate?

    Two cheap signals computed from pypdf's own output:
      * density  — average characters per page (a scanned/image PDF yields ~0)
      * sanity   — whitespace ratio (broken extraction runs words together)
    Below either threshold, the PDF is likely scanned or badly laid out.
    """
    if not pages:
        return False
    joined = "\n".join(p.text for p in pages)
    total = len(joined)
    chars_per_page = total / max(1, len(pages))
    ws_ratio = joined.count(" ") / max(1, total)
    if chars_per_page < settings.pdf_min_chars_per_page:
        log.info("pypdf gate: %.0f chars/page < %d -> escalate", chars_per_page, settings.pdf_min_chars_per_page)
        return False
    if ws_ratio < settings.pdf_min_whitespace_ratio:
        log.info("pypdf gate: whitespace ratio %.3f too low (jumbled) -> escalate", ws_ratio)
        return False
    return True


def _load_pdf(path: str, filename: str) -> list[LoadedPage]:
    """Load a PDF: pypdf by default, escalating to Docling when configured/needed."""
    strategy = settings.pdf_loader

    if strategy == "docling":
        return _merge_pages(_pdf_docling(path))

    # pypdf first (default path)
    try:
        pages = _pdf_pypdf(path)
    except Exception as e:  # noqa: BLE001
        log.warning("pypdf failed on '%s' (%s)", filename, e)
        pages = []

    if strategy == "pypdf":
        if not pages:
            raise DocumentLoadError(f"pypdf extracted no text from '{filename}' (it may be scanned).")
        return _merge_pages(pages)

    # strategy == "auto": accept pypdf if it passes the quality gate (run on the
    # per-page output, before merging, so density is measured per real PDF page).
    if _pypdf_ok(pages):
        log.info("pdf '%s': pypdf output accepted (%d page(s))", filename, len(pages))
        return _merge_pages(pages)

    # escalate to Docling
    log.info("pdf '%s': pypdf output weak -> trying Docling fallback", filename)
    try:
        dpages = _pdf_docling(path)
    except Exception as e:  # noqa: BLE001 — Docling not installed / failed
        log.warning("Docling fallback unavailable for '%s' (%s)", filename, e)
        dpages = []
    if dpages:
        log.info("pdf '%s': extracted with Docling (%d page(s))", filename, len(dpages))
        return _merge_pages(dpages)

    # Docling didn't help — use whatever pypdf produced, else give up.
    if pages:
        log.info("pdf '%s': falling back to pypdf output (Docling unavailable)", filename)
        return _merge_pages(pages)
    raise DocumentLoadError(
        f"Could not extract text from '{filename}': pypdf empty and Docling unavailable "
        "(install `docling` to handle scanned/complex PDFs)."
    )


def _merge_pages(pages: list[LoadedPage]) -> list[LoadedPage]:
    """Collapse a PDF's pages into ONE logical page.

    A Word document loads as a single page, so the heading-aware chunker sees the
    whole catalog and detects every entity. A PDF loads as one page *per PDF
    page*, which makes heading detection run per-page — a page holding a single
    entity would fall below the "≥2 sections" bar and drop to recursive chunking.
    Merging makes a PDF chunk with the exact same technique as a Word document.
    """
    real = [p for p in pages if (p.text or "").strip()]
    if len(real) <= 1:
        return real
    text = "\n\n".join(p.text.strip() for p in real)
    return [LoadedPage(page_number=1, text=text, section=real[0].section)]


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

    log.info("parsing '%s' (.%s, %d bytes)", filename, ext, len(raw))
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        if ext == "pdf":
            pages = _load_pdf(path, filename)   # pypdf -> Docling fallback
        else:
            pages = _to_pages(_langchain_documents(ext, path))
    except DocumentLoadError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize loader-specific errors
        raise DocumentLoadError(f"Failed to parse '{filename}': {e}") from e
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if not pages:
        raise DocumentLoadError(
            f"'{filename}' yielded no extractable text (a scanned PDF would need OCR)."
        )
    log.info("parsed '%s' into %d page(s)", filename, len(pages))
    return pages
