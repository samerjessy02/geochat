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
ANALYTICS = "ANALYTICS"     # aggregated statistics / analytical insight (Analytics panel)
UNKNOWN = "UNKNOWN"
VALID_INTENTS = {MAP, KNOWLEDGE, HYBRID, ANALYTICS, UNKNOWN}

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

# --- ANALYTICS cues: aggregated statistics / analytical insight -------------
# Proximity / "near me" / "within N meters" must stay MAP, never analytics.
_PROXIMITY_RE = re.compile(
    r"\b(within\s+\d+\s*(m|meters?|metres?|km|kilomet(?:er|re)s?)|near\s+me|nearest|closest|"
    r"radius|walking\s+distance|next\s+to|around\s+me)\b", re.IGNORECASE)
_AGG_RE = re.compile(
    r"\b(count|number\s+of|how\s+many|sum|total|average|avg|mean|median|percentage|percent|"
    r"proportion|share|distribution|breakdown|statistics|stats)\b", re.IGNORECASE)
_GROUP_RE = re.compile(
    r"\b(per|by|in\s+each|for\s+each|across|grouped?\s+by)\s+"
    r"(district|governorate|category|categories|area|region|zone|neighbou?rhood|city|type|"
    r"group|layer|dataset|[a-z]+)\b", re.IGNORECASE)
_TOPN_RE = re.compile(r"\b(top|bottom)\s+\d+\b", re.IGNORECASE)
_RANK_RE = re.compile(r"\b(most|least|fewest|highest|lowest|maximum|minimum)\b", re.IGNORECASE)
_COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?)\b", re.IGNORECASE)
_DISTRIB_RE = re.compile(r"\bdistribution\b", re.IGNORECASE)


def _looks_analytics(text: str) -> bool:
    """True for aggregation/grouping/ranking requests, but NOT proximity searches."""
    if not text or _PROXIMITY_RE.search(text):
        return False
    grouped = bool(_GROUP_RE.search(text))
    # Unambiguous analytics signals.
    if _DISTRIB_RE.search(text) or _COMPARE_RE.search(text) or _TOPN_RE.search(text):
        return True
    # Aggregation or ranking combined with a group-by ("count per district").
    if grouped and (_AGG_RE.search(text) or _RANK_RE.search(text)):
        return True
    # "which district has the most parks" (ranking a group).
    if re.search(r"\bwhich\b.*\b(most|least|highest|lowest|fewest|top|bottom)\b", text, re.IGNORECASE):
        return True
    return False


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

    @property
    def needs_analytics(self) -> bool:
        return self.intent == ANALYTICS

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "entity_focus": self.entity_focus,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "clarifying_question": self.clarifying_question,
            "needs_map": self.needs_map,
            "needs_rag": self.needs_rag,
            "needs_analytics": self.needs_analytics,
        }


def heuristic_intent(query: str) -> IntentResult:
    """Fast, dependency-free intent guess from lexical cues.

    Used as a fallback and to sanity-check the LLM. Returns UNKNOWN with low
    confidence when the signal is too weak to be trusted.
    """
    text = query or ""
    # Analytics (aggregation) takes precedence over map/knowledge cues, but only
    # when it's not a proximity search.
    if _looks_analytics(text):
        return IntentResult(intent=ANALYTICS, confidence=0.7,
                            reasoning="aggregation / grouping / ranking cues")
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
    "- ANALYTICS: wants AGGREGATED statistics or analytical insight ACROSS groups rather than "
    "individual features on the map. Triggers: count / sum / average / min / max / percentage / "
    "distribution / ranking (top or bottom N) / group-by / comparisons between groups — especially "
    "phrased as 'per <group>', 'by <group>', 'most/least', 'top N', 'distribution of', "
    "'compare X and Y by <group>'. e.g. 'count schools per district', 'top 10 districts by "
    "pharmacies', 'which district has the most parks', 'compare schools and hospitals by district'.\n"
    "- UNKNOWN: too ambiguous to route; needs clarification.\n"
    "GUIDELINES:\n"
    "- 'which/what <places> have/has/with <attribute>' (e.g. 'which cafes have handcrafted "
    "beverages') → HYBRID: the user wants to know WHICH places (list/plot) AND the qualifying "
    "detail, and that detail may live in documents rather than map columns.\n"
    "- PROXIMITY/where-is searches are MAP, never ANALYTICS: 'show schools within 500m of X', "
    "'pharmacies near me', 'nearest hospital', 'parks around here'.\n"
    "- A single-entity fact ('how many students does Cairo University have') is KNOWLEDGE, NOT "
    "ANALYTICS — ANALYTICS aggregates OVER groups (per district, by category, top N, etc.).\n"
    "- Be consistent: near-identical phrasings must get the SAME intent. Ignore minor wording "
    "differences ('which cafes has X' == 'cafes with X').\n"
    "Also extract the main entity/place the query focuses on, if any.\n"
    'Return ONLY JSON: {"intent": "MAP|KNOWLEDGE|HYBRID|ANALYTICS|UNKNOWN", "entity_focus": "..."|null, '
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
