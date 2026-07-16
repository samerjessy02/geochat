"""
config.py — centralized, environment-driven configuration for GeoChat.

Every tunable knob in the RAG/geo stack is read from environment variables
(loaded from a local ``.env`` during development) and exposed through a single
frozen :class:`Settings` object, ``settings``. Modules should import that object
rather than calling ``os.getenv`` directly so that configuration lives in one
auditable place and can be overridden per-deployment without code changes.

See ``.env.example`` for the full list of supported variables and defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all runtime configuration.

    Instances are created via :func:`get_settings` (cached) and read from the
    process environment exactly once. Treat every field as read-only.
    """

    # --- Databases -------------------------------------------------------
    db_url: str = field(default_factory=lambda: os.getenv("DB_URL", ""))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "geochat_documents"))
    # Client request timeout (seconds). The default of 5s is too low for writing
    # batches of high-dimensional vectors and causes "write operation timed out".
    qdrant_timeout: int = field(default_factory=lambda: _get_int("QDRANT_TIMEOUT", 60))
    qdrant_upsert_batch: int = field(default_factory=lambda: _get_int("QDRANT_UPSERT_BATCH", 64))

    # --- LLM provider ----------------------------------------------------
    # Modular interface — one of: groq | anthropic | openai | gemini.
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    # Kept as API_KEY for backwards-compat with the existing Groq setup.
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("API_KEY") or os.getenv("LLM_API_KEY") or None)
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.1))
    llm_max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 1024))

    # --- Embeddings ------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    )
    embedding_dim: int = field(default_factory=lambda: _get_int("EMBEDDING_DIM", 1024))
    embedding_normalize: bool = field(default_factory=lambda: _get_bool("EMBEDDING_NORMALIZE", True))
    # e5 models require "query: " / "passage: " prefixes; bge uses a query instruction.
    embedding_query_prefix: str = field(default_factory=lambda: os.getenv("EMBEDDING_QUERY_PREFIX", "query: "))
    embedding_passage_prefix: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")
    )

    # --- Chunking --------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 120))
    # Strategy: "auto"      -> one chunk per heading/section when a document
    #                          looks like a catalog, else recursive size-based.
    #           "heading"   -> force heading/section chunking (auto-falls back
    #                          to recursive when no headings are found).
    #           "recursive" -> always size-based RecursiveCharacterTextSplitter.
    chunk_strategy: str = field(default_factory=lambda: os.getenv("CHUNK_STRATEGY", "auto").lower())

    # --- Retrieval -------------------------------------------------------
    retrieval_top_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_TOP_K", 5))
    retrieval_candidate_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_CANDIDATE_K", 20))
    # Relative weights for Reciprocal Rank Fusion (dense vs. keyword).
    dense_weight: float = field(default_factory=lambda: _get_float("DENSE_WEIGHT", 1.0))
    bm25_weight: float = field(default_factory=lambda: _get_float("BM25_WEIGHT", 1.0))
    rrf_k: int = field(default_factory=lambda: _get_int("RRF_K", 60))

    # --- Reranking (optional) -------------------------------------------
    reranker_enabled: bool = field(default_factory=lambda: _get_bool("RERANKER_ENABLED", False))
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # --- Retrieval validation thresholds --------------------------------
    # Minimum blended context-relevance score (0..1) required to trust local
    # retrieval; below this the router falls through to web fallback.
    min_context_score: float = field(default_factory=lambda: _get_float("MIN_CONTEXT_SCORE", 0.35))
    min_keyword_overlap: float = field(default_factory=lambda: _get_float("MIN_KEYWORD_OVERLAP", 0.05))

    # --- Web fallback ----------------------------------------------------
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY") or None)
    web_fallback_enabled: bool = field(default_factory=lambda: _get_bool("WEB_FALLBACK_ENABLED", True))
    web_max_pages: int = field(default_factory=lambda: _get_int("WEB_MAX_PAGES", 3))

    # --- Guardrails ------------------------------------------------------
    guardrails_enabled: bool = field(default_factory=lambda: _get_bool("GUARDRAILS_ENABLED", True))
    max_query_chars: int = field(default_factory=lambda: _get_int("MAX_QUERY_CHARS", 2000))

    # --- Evaluation ------------------------------------------------------
    evaluation_enabled: bool = field(default_factory=lambda: _get_bool("EVALUATION_ENABLED", True))
    # Minimum faithfulness (0..1). If the evaluator finds the answer is less than
    # this fraction supported by the retrieved context, the answer is rejected as
    # a safe refusal rather than surfaced (guards against hallucinated facts).
    min_faithfulness: float = field(default_factory=lambda: _get_float("MIN_FAITHFULNESS", 0.5))

    # --- Logging ---------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_to_file: bool = field(default_factory=lambda: _get_bool("LOG_TO_FILE", False))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "geochat.log"))
    log_json: bool = field(default_factory=lambda: _get_bool("LOG_JSON", False))

    @property
    def is_groq(self) -> bool:
        return self.llm_provider == "groq"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton (cached)."""
    return Settings()


# Convenience module-level handle so callers can ``from config import settings``.
settings = get_settings()
