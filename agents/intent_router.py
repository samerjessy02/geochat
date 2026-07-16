"""
agents/intent_router.py — classify query intent and decide the pipeline.

Four intents (per the spec):

    MAP        spatial retrieval only  -> PostGIS SQL, render features on the map.
    KNOWLEDGE  textual answer only     -> hybrid RAG (+ web fallback), no map.
    HYBRID     both                    -> run spatial AND RAG, return features +
                                          a grounded natural-language answer.
    UNKNOWN    ambiguous               -> ask a clarifying question.

Classification uses the modular LLM for accuracy, with a transparent
rule-based heuristic (:func:`heuristic_intent`) as both a fast path and a
fallback when the LLM is unavailable. The heuristic is pure and unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.llm_client import get_llm, LLMError
from agents.logging_config import get_logger, snippet

log = get_logger("intent_router")

MAP = "MAP"
KNOWLEDGE = "KNOWLEDGE"
HYBRID = "HYBRID"
UNKNOWN = "UNKNOWN"
VALID_INTENTS = {MAP, KNOWLEDGE, HYBRID, UNKNOWN}

# Verbs/nouns that signal a request to place things on the map.
_MAP_CUES = [
    r"\b(show|find|display|plot|map|locate|list|where|near|nearby|around|within|closest|nearest)\b",
    r"\b(cafes?|coffee|restaurants?|museums?|hospitals?|universit(y|ies)|schools?|"
    r"pharmac(y|ies)|parks?|stations?|landmarks?|mosques?|churches?|fire ?stations?|places?)\b",
    r"\b(street|district|downtown|city|area|road|square|neighbou?rhood)\b",
]
# Phrases that signal a request for facts/explanation (knowledge).
_KNOWLEDGE_CUES = [
    r"\b(how many|how much|when|why|who|what (is|are|were|was)|tell me about|history|"
    r"established|founded|enrol(l)?ed|students?|opening hours|hours|about|describe|"
    r"explain|information|details?|phone|contact|rating|reviews?|price|menu)\b",
]

_MAP_RE = [re.compile(p, re.IGNORECASE) for p in _MAP_CUES]
_KNOWLEDGE_RE = [re.compile(p, re.IGNORECASE) for p in _KNOWLEDGE_CUES]


@dataclass
class IntentResult:
    """Resolved intent plus the routing flags the API acts on."""

    intent: str
    entity_focus: str | None = None
    confidence: float = 0.0
    reasoning: str = ""
    clarifying_question: str | None = None

    @property
    def needs_map(self) -> bool:
        return self.intent in (MAP, HYBRID)

    @property
    def needs_rag(self) -> bool:
        return self.intent in (KNOWLEDGE, HYBRID)

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "entity_focus": self.entity_focus,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "clarifying_question": self.clarifying_question,
            "needs_map": self.needs_map,
            "needs_rag": self.needs_rag,
        }


def heuristic_intent(query: str) -> IntentResult:
    """Fast, dependency-free intent guess from lexical cues.

    Used as a fallback and to sanity-check the LLM. Returns UNKNOWN with low
    confidence when the signal is too weak to be trusted.
    """
    text = query or ""
    map_hits = sum(1 for r in _MAP_RE if r.search(text))
    knowledge_hits = sum(1 for r in _KNOWLEDGE_RE if r.search(text))

    if map_hits >= 2 and knowledge_hits >= 1:
        intent, conf = HYBRID, 0.6
    elif map_hits >= 1 and knowledge_hits == 0:
        intent, conf = MAP, 0.6
    elif knowledge_hits >= 1 and map_hits <= 1:
        intent, conf = KNOWLEDGE, 0.55
    elif map_hits >= 1:
        intent, conf = MAP, 0.5
    else:
        intent, conf = UNKNOWN, 0.3
    return IntentResult(intent=intent, confidence=conf, reasoning="lexical heuristic")


_SYSTEM = (
    "You are an intent classifier for a geospatial assistant. Classify the user's query "
    "into exactly one intent:\n"
    "- MAP: wants places shown/found/plotted on a map (spatial only), e.g. 'show cafes on X street'.\n"
    "- KNOWLEDGE: wants facts/explanation about a place or topic (no map needed), e.g. 'when was X built'.\n"
    "- HYBRID: wants BOTH a map AND textual facts.\n"
    "- UNKNOWN: too ambiguous to route; needs clarification.\n"
    "GUIDELINES:\n"
    "- 'which/what <places> have/has/with <attribute>' (e.g. 'which cafes have handcrafted "
    "beverages') → HYBRID: the user wants to know WHICH places (list/plot) AND the qualifying "
    "detail, and that detail may live in documents rather than map columns.\n"
    "- Be consistent: near-identical phrasings must get the SAME intent. Ignore minor wording "
    "differences ('which cafes has X' == 'cafes with X').\n"
    "Also extract the main entity/place the query focuses on, if any.\n"
    'Return ONLY JSON: {"intent": "MAP|KNOWLEDGE|HYBRID|UNKNOWN", "entity_focus": "..."|null, '
    '"confidence": <0..1>, "reasoning": "...", "clarifying_question": "..."|null}'
)


def classify_intent(query: str, *, use_llm: bool = True) -> IntentResult:
    """Classify ``query`` into a routing decision.

    Tries the LLM classifier first (when ``use_llm``), validates its output, and
    falls back to :func:`heuristic_intent` on any error or malformed response.
    For UNKNOWN, ensures a clarifying question is present.
    """
    log.info("classifying intent for: '%s'", snippet(query))
    if not use_llm:
        r = heuristic_intent(query)
        log.info("intent=%s (heuristic, conf=%.2f)", r.intent, r.confidence)
        return r

    try:
        out = get_llm().complete_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": query}]
        )
        intent = str(out.get("intent", "")).upper()
        if intent not in VALID_INTENTS:
            log.warning("LLM returned invalid intent %r — falling back to heuristic", intent)
            return heuristic_intent(query)
        result = IntentResult(
            intent=intent,
            entity_focus=(out.get("entity_focus") or None),
            confidence=float(out.get("confidence", 0.5)),
            reasoning=str(out.get("reasoning", ""))[:300],
            clarifying_question=out.get("clarifying_question") or None,
        )
        if result.intent == UNKNOWN and not result.clarifying_question:
            result.clarifying_question = (
                "Could you clarify what you'd like — should I show places on the map, "
                "give you information, or both?"
            )
        log.info("intent=%s (LLM, conf=%.2f, entity=%s)",
                 result.intent, result.confidence, result.entity_focus or "-")
        return result
    except (LLMError, Exception) as e:  # noqa: BLE001
        log.warning("intent LLM classification failed (%s) — using heuristic", e)
        return heuristic_intent(query)
