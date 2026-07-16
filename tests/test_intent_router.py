"""Unit tests for the heuristic intent classifier (LLM path not exercised)."""

from agents.intent_router import heuristic_intent, classify_intent, MAP, KNOWLEDGE, HYBRID, UNKNOWN


def test_map_query_detected():
    r = heuristic_intent("Show cafes on Tahrir Street")
    assert r.intent == MAP
    assert r.needs_map and not r.needs_rag


def test_knowledge_query_detected():
    r = heuristic_intent("When was this museum established?")
    assert r.intent == KNOWLEDGE
    assert r.needs_rag and not r.needs_map


def test_hybrid_query_detected():
    r = heuristic_intent("Show Cairo University on the map and tell me how many students are enrolled in 2025")
    assert r.intent == HYBRID
    assert r.needs_map and r.needs_rag


def test_ambiguous_query_is_unknown():
    r = heuristic_intent("hmm")
    assert r.intent == UNKNOWN


def test_classify_with_use_llm_false_uses_heuristic():
    r = classify_intent("Find hospitals near Cairo University", use_llm=False)
    assert r.intent in {MAP, HYBRID}
