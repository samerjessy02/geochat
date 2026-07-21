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

    # --- SBG / Bedrock primary (optional) --------------------------------
    # A managed chat gateway (Bedrock-backed). When configured (base URL + key),
    # it becomes the PRIMARY model; every call falls back automatically to the
    # LLM_PROVIDER above if the gateway is unreachable or errors. Leave
    # SBG_API_BASE/SBG_API_KEY unset to keep using LLM_PROVIDER only.
    sbg_enabled: bool = field(default_factory=lambda: _get_bool("SBG_ENABLED", True))
    sbg_api_base: str = field(default_factory=lambda: os.getenv("SBG_API_BASE", "").rstrip("/"))
    sbg_chat_path: str = field(default_factory=lambda: os.getenv("SBG_CHAT_PATH", "/student/chat"))
    sbg_api_key: str | None = field(default_factory=lambda: os.getenv("SBG_API_KEY") or None)
    sbg_model_id: str = field(
        default_factory=lambda: os.getenv("SBG_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
    )
    sbg_timeout: int = field(default_factory=lambda: _get_int("SBG_TIMEOUT", 60))

    @property
    def sbg_available(self) -> bool:
        """True when the SBG gateway is configured enough to try as primary."""
        return bool(self.sbg_enabled and self.sbg_api_base and self.sbg_api_key)

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

    # --- PDF extraction --------------------------------------------------
    # "auto"    -> pypdf first; fall back to Docling when pypdf's output looks
    #              poor (scanned / complex layout). "pypdf" / "docling" force one.
    pdf_loader: str = field(default_factory=lambda: os.getenv("PDF_LOADER", "auto").lower())
    # Quality gate on pypdf output (below either threshold -> escalate to Docling).
    pdf_min_chars_per_page: int = field(default_factory=lambda: _get_int("PDF_MIN_CHARS_PER_PAGE", 40))
    pdf_min_whitespace_ratio: float = field(default_factory=lambda: _get_float("PDF_MIN_WHITESPACE_RATIO", 0.05))

    # --- Retrieval -------------------------------------------------------
    retrieval_top_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_TOP_K", 5))
    retrieval_candidate_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_CANDIDATE_K", 20))
    # For count / "list all" / negation queries: if the matched document has at
    # most this many chunks, feed ALL of them (not just top-k) so the model can
    # reason over the complete set.
    exhaustive_max_chunks: int = field(default_factory=lambda: _get_int("EXHAUSTIVE_MAX_CHUNKS", 40))
    # For count / "list all" / negation queries: if the matched document has at
    # most this many chunks, feed ALL of them (not just top-k) so the model can
    # reason over the complete set.
    exhaustive_max_chunks: int = field(default_factory=lambda: _get_int("EXHAUSTIVE_MAX_CHUNKS", 40))
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
    # Scraper engine: "auto" (Firecrawl if a key is set, else httpx+BeautifulSoup),
    # "firecrawl", or "httpx".
    scraper: str = field(default_factory=lambda: os.getenv("SCRAPER", "auto").lower())
    firecrawl_api_key: str | None = field(default_factory=lambda: os.getenv("FIRECRAWL_API_KEY") or None)
    # Firecrawl's current API is v2 (v1 is legacy). Override if the base changes.
    firecrawl_api_base: str = field(
        default_factory=lambda: os.getenv("FIRECRAWL_API_BASE", "https://api.firecrawl.dev/v2").rstrip("/")
    )
    firecrawl_crawl_limit: int = field(default_factory=lambda: _get_int("FIRECRAWL_CRAWL_LIMIT", 10))
    firecrawl_timeout: int = field(default_factory=lambda: _get_int("FIRECRAWL_TIMEOUT", 90))
    # Multi-page strategy: explore the top-N ranked pages and follow up to M of
    # each page's most relevant links (one level deeper). FIRECRAWL_CRAWL_LIMIT
    # is the overall scrape budget.
    firecrawl_top_pages: int = field(default_factory=lambda: _get_int("FIRECRAWL_TOP_PAGES", 3))
    firecrawl_deep_links: int = field(default_factory=lambda: _get_int("FIRECRAWL_DEEP_LINKS", 2))

    # --- Conversational memory ------------------------------------------
    # ConversationBufferWindowMemory: keep only the last K interactions
    # (one interaction = a user turn + the assistant's reply). Bounded by turn
    # count so history stays small and never crowds out retrieved context.
    memory_enabled: bool = field(default_factory=lambda: _get_bool("MEMORY_ENABLED", True))
    memory_window_k: int = field(default_factory=lambda: _get_int("MEMORY_WINDOW_K", 5))
    # Rewrite a follow-up ("does it deliver?") into a standalone question using
    # the window, so retrieval + intent classification see a self-contained query.
    memory_condense: bool = field(default_factory=lambda: _get_bool("MEMORY_CONDENSE", True))

    # --- Semantic response cache ----------------------------------------
    # Cache full /query responses keyed on the query's EMBEDDING (semantic
    # meaning), not the raw string — so "what is NBE's location" and "locate NBE"
    # hit the same entry. A hit skips intent classification, SQL generation,
    # retrieval and LLM generation entirely.
    cache_enabled: bool = field(default_factory=lambda: _get_bool("CACHE_ENABLED", True))
    # Cosine similarity (0..1) a new query must reach against a cached query to be
    # a hit. Higher = stricter (fewer false hits); e5 paraphrases sit ~0.90–0.97.
    cache_threshold: float = field(default_factory=lambda: _get_float("CACHE_SIMILARITY_THRESHOLD", 0.93))
    cache_max_entries: int = field(default_factory=lambda: _get_int("CACHE_MAX_ENTRIES", 512))
    # Entry lifetime in seconds; 0 disables time-based expiry (rely on
    # invalidation when documents/datasets change).
    cache_ttl: int = field(default_factory=lambda: _get_int("CACHE_TTL_SECONDS", 0))

    # --- Routing (Valhalla) ----------------------------------------------
    # Self-hosted routing engine for walk/drive routes and isochrones. Default is
    # the port the gis-ops Valhalla docker image serves on. Change if you mapped a
    # different host port or run it elsewhere.
    routing_enabled: bool = field(default_factory=lambda: _get_bool("ROUTING_ENABLED", True))
    valhalla_url: str = field(default_factory=lambda: os.getenv("VALHALLA_URL", "http://localhost:8002").rstrip("/"))
    valhalla_timeout: int = field(default_factory=lambda: _get_int("VALHALLA_TIMEOUT", 15))

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
