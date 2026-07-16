"""
agents/embeddings.py — sentence-transformer embeddings with correct prefixes.

Wraps a single SentenceTransformer model (default
``intfloat/multilingual-e5-large``, 1024-dim, Arabic + English capable) and
exposes two intent-aware methods:

    embed_query(text)      -> list[float]
    embed_passages(texts)  -> list[list[float]]

e5-family models are trained with asymmetric ``"query: "`` / ``"passage: "``
prefixes; using the wrong prefix (or none) measurably degrades retrieval. Those
prefixes are configurable so the module also works with bge-style models by
setting ``EMBEDDING_QUERY_PREFIX``/``EMBEDDING_PASSAGE_PREFIX`` appropriately.

The model is loaded lazily and cached on first use so importing this module (and
the pure-logic modules that depend on config) stays cheap and torch-free until an
embedding is actually requested.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import TYPE_CHECKING

from config import settings
from agents.logging_config import get_logger

log = get_logger("embeddings")

if TYPE_CHECKING:  # avoid importing torch/sentence-transformers at module load
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    log.info("loading embedding model '%s' (first use may download it)…", settings.embedding_model)
    start = time.perf_counter()
    model = SentenceTransformer(settings.embedding_model)
    log.info("embedding model ready in %.1fs", time.perf_counter() - start)
    return model


def _prefix(text: str, prefix: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        return text
    return f"{prefix} {text}" if not text.startswith(prefix) else text


def embed_query(text: str) -> list[float]:
    """Embed a search query (applies the query prefix, L2-normalizes)."""
    model = _get_model()
    vec = model.encode(
        _prefix(text, settings.embedding_query_prefix),
        normalize_embeddings=settings.embedding_normalize,
        convert_to_numpy=True,
    )
    return vec.tolist()


def embed_passages(texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
    """Embed document passages (applies the passage prefix, L2-normalizes)."""
    if not texts:
        return []
    model = _get_model()
    prefixed = [_prefix(t, settings.embedding_passage_prefix) for t in texts]
    start = time.perf_counter()
    vectors = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=settings.embedding_normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    log.debug("embedded %d passage(s) in %.0fms", len(texts), (time.perf_counter() - start) * 1000)
    return [v.tolist() for v in vectors]


def embedding_dimension() -> int:
    """Return the configured embedding dimension (used to size the collection)."""
    return settings.embedding_dim
