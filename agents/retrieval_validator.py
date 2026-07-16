"""
agents/retrieval_validator.py — is the retrieved context actually relevant?

Before the LLM is allowed to answer, this module quantifies how well the
retrieved chunks match the query. It combines three cheap, transparent signals
so the router can decide whether to trust local retrieval or fall through to the
web fallback (the spec's "Retrieval Validation" step):

    top_score        best fused retrieval score (retriever's own confidence)
    keyword_overlap  fraction of query content-words present in the context
    mean_score       average fused score across the retrieved chunks

These are blended into a single ``context_score`` in [0, 1]. The pure-Python
implementation has no external dependencies, so it is fully unit-testable and
adds negligible latency. An optional LLM grader (used elsewhere in
``rag_agent``) can layer on top for harder cases.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import settings
from agents.hybrid_retriever import tokenize
from agents.logging_config import get_logger

log = get_logger("retrieval_validator")

# Common bilingual stopwords stripped before computing keyword overlap so that
# function words don't inflate the score.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "how", "what", "when", "where", "who", "which", "many",
    "much", "do", "does", "did", "this", "that", "these", "those", "with", "about",
    "من", "في", "على", "الى", "إلى", "عن", "ما", "كم", "هل", "و", "او", "أو", "هذا",
    "هذه", "التي", "الذي", "كيف", "متى", "اين", "أين",
}


@dataclass
class ValidationResult:
    """Outcome of validating retrieved context against a query."""

    context_score: float
    keyword_overlap: float
    top_score: float
    mean_score: float
    is_sufficient: bool
    num_chunks: int

    def as_dict(self) -> dict:
        return {
            "context_score": round(self.context_score, 4),
            "keyword_overlap": round(self.keyword_overlap, 4),
            "top_score": round(self.top_score, 4),
            "mean_score": round(self.mean_score, 4),
            "is_sufficient": self.is_sufficient,
            "num_chunks": self.num_chunks,
        }


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1}


def keyword_overlap(query: str, context: str) -> float:
    """Fraction of the query's content-word set that appears in ``context``."""
    q = _content_tokens(query)
    if not q:
        return 0.0
    c = _content_tokens(context)
    return len(q & c) / len(q)


def _normalize_scores(hits: list[dict]) -> list[float]:
    """Map fused/dense scores into a comparable 0..1 range.

    Dense cosine scores are already ~0..1; RRF fused scores are small positive
    numbers. We min-max normalize within the result set so the blend is stable
    regardless of which scoring path produced them.
    """
    raw = [float(h.get("score", h.get("dense_score", 0.0))) for h in hits]
    if not raw:
        return []
    lo, hi = min(raw), max(raw)
    if hi <= lo:
        return [1.0 for _ in raw]
    return [(r - lo) / (hi - lo) for r in raw]


def validate(query: str, hits: list[dict]) -> ValidationResult:
    """Score how relevant ``hits`` are to ``query`` and decide sufficiency.

    ``is_sufficient`` is True when the blended ``context_score`` clears
    ``MIN_CONTEXT_SCORE`` and keyword overlap clears ``MIN_KEYWORD_OVERLAP`` —
    both must hold, so a lexically-empty but vector-close match still triggers
    fallback rather than a confident-but-ungrounded answer.
    """
    if not hits:
        log.info("retrieval validation: 0 chunks -> insufficient (will try web fallback)")
        return ValidationResult(0.0, 0.0, 0.0, 0.0, False, 0)

    context = "\n".join(h.get("text", "") for h in hits)
    overlap = keyword_overlap(query, context)

    norm = _normalize_scores(hits)
    # Use the top hit's *absolute* dense score when available as a confidence anchor.
    dense_scores = [float(h.get("dense_score", 0.0)) for h in hits]
    top_score = max(dense_scores) if any(dense_scores) else (max(norm) if norm else 0.0)
    mean_score = sum(norm) / len(norm) if norm else 0.0

    # Blend: retrieval confidence (top dense score) + lexical grounding (overlap).
    context_score = 0.6 * top_score + 0.4 * overlap

    is_sufficient = (
        context_score >= settings.min_context_score
        and overlap >= settings.min_keyword_overlap
    )
    log.info(
        "retrieval validation: context_score=%.2f (min %.2f), overlap=%.2f (min %.2f), "
        "top=%.2f, %d chunk(s) -> %s",
        context_score, settings.min_context_score, overlap, settings.min_keyword_overlap,
        top_score, len(hits), "SUFFICIENT" if is_sufficient else "INSUFFICIENT",
    )
    return ValidationResult(
        context_score=context_score,
        keyword_overlap=overlap,
        top_score=top_score,
        mean_score=mean_score,
        is_sufficient=is_sufficient,
        num_chunks=len(hits),
    )
