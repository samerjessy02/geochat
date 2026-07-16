"""
agents/evaluation.py — automatic RAG evaluation (RAGAS-style).

After every generated answer the pipeline scores three dimensions the spec
requires, plus the retrieval score already produced upstream:

    answer_relevancy   Does the answer address the question? (LLM judge, 0..1)
    context_relevancy  Were the retrieved chunks relevant? (LLM judge, 0..1)
    faithfulness       Is every claim in the answer supported by the context?
                       (LLM judge that extracts claims and verifies each, 0..1)
    retrieval_score    Passed through from retrieval validation.

The judges use the modular :class:`LLMClient`, so they run on whatever provider
is configured. Evaluation is best-effort: any judge failure yields ``None`` for
that metric rather than breaking the answer path. A cheap lexical fallback for
``answer_relevancy`` keeps a signal even when the LLM judge is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import settings
from agents.llm_client import get_llm, LLMError
from agents.retrieval_validator import keyword_overlap
from agents.logging_config import get_logger

log = get_logger("evaluation")


@dataclass
class EvaluationScores:
    """Container for the four evaluation metrics (any may be ``None``)."""

    answer_relevancy: float | None = None
    context_relevancy: float | None = None
    faithfulness: float | None = None
    retrieval_score: float | None = None
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "answer_relevancy": self.answer_relevancy,
            "context_relevancy": self.context_relevancy,
            "faithfulness": self.faithfulness,
            "retrieval_score": self.retrieval_score,
            "details": self.details,
        }


def _clamp01(value) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _judge_answer_relevancy(question: str, answer: str) -> float | None:
    prompt = (
        "On a scale from 0.0 to 1.0, how well does the ANSWER directly and completely "
        "address the QUESTION? 1.0 = fully answers it, 0.0 = irrelevant.\n\n"
        f"QUESTION: {question}\n\nANSWER: {answer}\n\n"
        'Return ONLY JSON: {"score": <float>}'
    )
    try:
        out = get_llm().complete_json([{"role": "user", "content": prompt}])
        return _clamp01(out.get("score"))
    except (LLMError, Exception):  # noqa: BLE001
        # Lexical fallback: overlap of question content-words with the answer.
        return round(keyword_overlap(question, answer), 4) if answer else 0.0


def _judge_context_relevancy(question: str, context: str) -> float | None:
    prompt = (
        "On a scale from 0.0 to 1.0, how relevant is the retrieved CONTEXT for answering "
        "the QUESTION? Judge the context only, not any answer.\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context[:6000]}\n\n"
        'Return ONLY JSON: {"score": <float>}'
    )
    try:
        out = get_llm().complete_json([{"role": "user", "content": prompt}])
        return _clamp01(out.get("score"))
    except (LLMError, Exception):  # noqa: BLE001
        return None


def _judge_faithfulness(answer: str, context: str) -> tuple[float | None, dict]:
    prompt = (
        "Extract the distinct factual claims in the ANSWER, then decide for each whether it "
        "is supported by the CONTEXT. Return the fraction supported.\n\n"
        f"CONTEXT:\n{context[:6000]}\n\nANSWER: {answer}\n\n"
        'Return ONLY JSON: {"claims": [{"claim": "...", "supported": true|false}], "score": <float>}'
    )
    try:
        out = get_llm().complete_json([{"role": "user", "content": prompt}])
        score = out.get("score")
        claims = out.get("claims", [])
        if score is None and claims:
            supported = sum(1 for c in claims if c.get("supported"))
            score = supported / len(claims)
        return _clamp01(score), {"claims": claims}
    except (LLMError, Exception):  # noqa: BLE001
        return None, {}


def evaluate(
    question: str,
    answer: str,
    context: str,
    *,
    retrieval_score: float | None = None,
    found: bool = True,
) -> EvaluationScores:
    """Score an answer across the RAG evaluation dimensions.

    Skipped (returns empty scores with ``retrieval_score`` passed through) when
    ``EVALUATION_ENABLED`` is false, the answer reports ``found=False``, or there
    is no context to judge against.
    """
    scores = EvaluationScores(retrieval_score=retrieval_score)
    if not settings.evaluation_enabled or not found or not answer:
        if not settings.evaluation_enabled:
            log.debug("evaluation disabled (EVALUATION_ENABLED=false)")
        return scores

    scores.answer_relevancy = _judge_answer_relevancy(question, answer)
    if context.strip():
        scores.context_relevancy = _judge_context_relevancy(question, context)
        scores.faithfulness, scores.details = _judge_faithfulness(answer, context)
    return scores
