"""Unit tests for official-domain resolution helpers in the web fallback.

These exercise the pure domain-scoring logic (no network). The module imports
httpx at load time, so the suite skips cleanly when httpx isn't installed.
"""

import pytest

pytest.importorskip("httpx", reason="httpx not installed")

from agents.web_fallback import _root_domain, _is_blocked_domain, _score_domain


def test_root_domain_normalizes():
    assert _root_domain("www.cu.edu.eg:443") == "cu.edu.eg"
    assert _root_domain("EG.LinkedIn.com") == "eg.linkedin.com"


def test_social_and_aggregator_domains_blocked():
    assert _is_blocked_domain("linkedin.com")
    assert _is_blocked_domain("eg.linkedin.com")     # subdomain
    assert _is_blocked_domain("en.wikipedia.org")
    assert not _is_blocked_domain("cu.edu.eg")


def test_official_edu_domain_outscores_generic():
    entity = "Cairo University"
    assert _score_domain("cu.edu.eg", entity) > _score_domain("topuniversities.com", entity)


def test_domain_containing_entity_name_scores_high():
    entity = "Cairo University"
    # a domain containing the entity token beats an unrelated one
    assert _score_domain("cairouniversity.eg", entity) > _score_domain("randomsite.com", entity)
