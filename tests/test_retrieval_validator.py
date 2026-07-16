"""Unit tests for retrieval validation (context sufficiency)."""

from agents.retrieval_validator import validate, keyword_overlap


def test_keyword_overlap_ignores_stopwords():
    # "how many" are stopwords; content words: students, cairo, university
    ov = keyword_overlap("How many students at Cairo University", "Cairo University has many students")
    assert ov > 0.5


def test_empty_hits_are_insufficient():
    r = validate("anything", [])
    assert not r.is_sufficient
    assert r.num_chunks == 0
    assert r.context_score == 0.0


def test_relevant_high_score_hits_are_sufficient():
    hits = [
        {"text": "Cairo University enrolled 250000 students in 2025.", "dense_score": 0.82, "score": 0.05},
        {"text": "The university is located in Giza.", "dense_score": 0.6, "score": 0.03},
    ]
    r = validate("How many students at Cairo University in 2025", hits)
    assert r.is_sufficient
    assert r.top_score >= 0.8


def test_irrelevant_hits_are_insufficient():
    hits = [
        {"text": "The best pizza recipe uses fresh basil.", "dense_score": 0.2, "score": 0.01},
    ]
    r = validate("How many students at Cairo University in 2025", hits)
    assert not r.is_sufficient
