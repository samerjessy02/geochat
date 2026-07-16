"""Unit tests for the LangChain-backed document loaders (txt/md/html/csv).

These require the LangChain community loaders; the suite skips them cleanly
when those optional dependencies are not installed.
"""

import pytest

pytest.importorskip("langchain_community", reason="LangChain loaders not installed")

from agents.document_loader import load_document, DocumentLoadError


def test_txt_loader():
    pages = load_document("notes.txt", b"Hello world.\nSecond line.")
    assert len(pages) >= 1
    assert "Hello world." in pages[0].text


def test_csv_loader_renders_rows():
    csv_bytes = b"name,students\nCairo University,250000\nAin Shams,180000\n"
    pages = load_document("data.csv", csv_bytes)
    text = "\n".join(p.text for p in pages)
    # CSVLoader emits one Document per row as "col: value" lines.
    assert "Cairo University" in text
    assert "students: 250000" in text


def test_html_loader_extracts_text():
    html = b"<html><body><h1>Museums</h1><p>Real content here</p></body></html>"
    pages = load_document("page.html", html)
    assert "Real content here" in pages[0].text


def test_markdown_loader():
    md = b"# Title\n\nSome **bold** text about museums."
    pages = load_document("doc.md", md)
    assert "museums" in "\n".join(p.text for p in pages)


def test_unsupported_type_raises():
    with pytest.raises(DocumentLoadError):
        load_document("archive.zip", b"binary")
