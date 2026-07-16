"""Unit tests for Reciprocal Rank Fusion and tokenization (pure logic)."""

import math

import pytest

from agents.hybrid_retriever import reciprocal_rank_fusion, tokenize


def test_rrf_prefers_items_ranked_high_in_both_lists():
    dense = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    ids = [i for i, _ in fused]
    # 'a' (ranks 1,2) and 'b' (ranks 2,1) should outrank single-list 'c'/'d'.
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-1] in {"c", "d"}


def test_rrf_scores_match_formula():
    fused = dict(reciprocal_rank_fusion([["x", "y"]], weights=[1.0], k=60))
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62)


def test_rrf_weights_bias_toward_a_retriever():
    dense = ["a", "b"]
    bm25 = ["b", "a"]
    # Heavily weight bm25 -> 'b' (top of bm25) should win.
    fused = reciprocal_rank_fusion([dense, bm25], weights=[0.1, 10.0], k=60)
    assert fused[0][0] == "b"


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []


def test_tokenize_handles_arabic_and_latin():
    toks = tokenize("Cairo University جامعة القاهرة 2025")
    assert "cairo" in toks
    assert "university" in toks
    assert "2025" in toks
    assert "جامعة" in toks
