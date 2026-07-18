"""
agents/web_fallback.py — official-website-first web retrieval.

Invoked only when local RAG has no answer or its context fails validation. The
spec is explicit about ordering: never search arbitrary sites first. Every
GeoJSON feature already carries a ``website`` field, so:

  Tier 0  Use the feature's own ``website`` URL (highest trust).
  Tier A  If no URL is known, resolve the entity's official domain via a search
          engine (Tavily) and cache it in ``domain_cache`` for reuse.
  Tier B  Search restricted to that official domain and scrape the top pages.
  Tier C  Only as a last resort, an unrestricted search — clearly labelled as a
          lower-trust source.

Scraped pages are chunked and embedded *ephemerally* (never persisted to
Qdrant) and the most query-relevant chunks are returned as context, so the same
grounded-generation path can consume web context and local context identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from config import settings
from agents.guardrails import redact_secrets
from agents.logging_config import get_logger

log = get_logger("web_fallback")

_HEADERS = {"User-Agent": "GeoChat/1.0 (educational spatial project) python-httpx"}
_TAVILY_URL = "https://api.tavily.com/search"

# Social/aggregator domains are never treated as an entity's "official" site.
_BLOCKED_DOMAINS = {
    "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "tiktok.com", "pinterest.com", "reddit.com", "quora.com",
    "wikipedia.org", "wikiwand.com", "fandom.com", "tripadvisor.com", "yelp.com",
    "foursquare.com", "glassdoor.com", "indeed.com", "crunchbase.com", "zoominfo.com",
    "bloomberg.com", "mapquest.com", "yellowpages.com", "medium.com", "blogspot.com",
    "wordpress.com", "amazon.com", "booking.com",
}
# TLDs that strongly indicate an authoritative/official site.
_OFFICIAL_TLD_HINTS = (".edu", ".gov", ".ac.", ".edu.eg", ".gov.eg", ".org")


def _root_domain(netloc: str) -> str:
    """Normalize a netloc to a bare host (lowercase, no port, no leading www.)."""
    host = (netloc or "").lower().split(":")[0].strip()
    return host[4:] if host.startswith("www.") else host


def _is_blocked_domain(domain: str) -> bool:
    return any(domain == b or domain.endswith("." + b) for b in _BLOCKED_DOMAINS)


def _score_domain(domain: str, entity: str) -> float:
    """Rank a candidate domain as the entity's official site (higher = better)."""
    score = 0.0
    tokens = [t for t in re.split(r"\W+", entity.lower()) if len(t) > 2]
    if any(t in domain for t in tokens):
        score += 2.0
    if any(hint in domain for hint in _OFFICIAL_TLD_HINTS):
        score += 1.5
    # Prefer shorter, canonical domains over deep subdomains.
    score += max(0.0, 1.0 - 0.1 * domain.count("."))
    return score


class WebFallbackError(Exception):
    """Raised when web retrieval is requested but cannot proceed."""


@dataclass
class WebContext:
    """Context assembled from official web sources."""

    context: str
    sources: list[str] = field(default_factory=list)
    source_tier: str = "none"

    @property
    def has_content(self) -> bool:
        return bool(self.context.strip())


# --------------------------------------------------------------------------- #
# domain cache (Postgres) — resolve an entity to its official domain once
# --------------------------------------------------------------------------- #

def ensure_domain_cache_table() -> None:
    from sqlalchemy import text

    from db import engine

    with engine.begin() as conn:
        conn.execute(
            text(
                """CREATE TABLE IF NOT EXISTS domain_cache (
                       entity_name TEXT PRIMARY KEY,
                       domain TEXT NOT NULL,
                       discovered_at TIMESTAMPTZ NOT NULL DEFAULT now()
                   )"""
            )
        )


def _get_cached_domain(entity: str) -> str | None:
    from sqlalchemy import text

    from db import engine

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT domain FROM domain_cache WHERE entity_name = :n"),
            {"n": entity.lower().strip()},
        ).fetchone()
    return row[0] if row else None


def _cache_domain(entity: str, domain: str) -> None:
    from sqlalchemy import text

    from db import engine

    with engine.begin() as conn:
        conn.execute(
            text(
                """INSERT INTO domain_cache (entity_name, domain) VALUES (:n, :d)
                   ON CONFLICT (entity_name)
                   DO UPDATE SET domain = EXCLUDED.domain, discovered_at = now()"""
            ),
            {"n": entity.lower().strip(), "d": domain},
        )


# --------------------------------------------------------------------------- #
# search + scrape
# --------------------------------------------------------------------------- #

def _tavily_search(query: str, include_domains: list[str] | None = None, max_results: int = 5) -> list[dict]:
    if not settings.tavily_api_key:
        raise WebFallbackError("TAVILY_API_KEY is not configured — official-domain discovery is unavailable.")
    payload: dict = {"api_key": settings.tavily_api_key, "query": query, "max_results": max_results}
    if include_domains:
        payload["include_domains"] = include_domains
    try:
        r = httpx.post(_TAVILY_URL, json=payload, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:  # noqa: BLE001
        raise WebFallbackError(f"Web search failed: {e}") from e


def _use_firecrawl() -> bool:
    """Whether Firecrawl is the active scraper (JS-rendering + crawling)."""
    if settings.scraper == "firecrawl":
        return bool(settings.firecrawl_api_key)
    if settings.scraper == "auto":
        return bool(settings.firecrawl_api_key)
    return False


def _fc_headers() -> dict:
    return {"Authorization": f"Bearer {settings.firecrawl_api_key}", "Content-Type": "application/json"}


def _fc_markdown(obj: dict) -> str | None:
    """Pull markdown out of a Firecrawl response item, tolerant of shape."""
    if not isinstance(obj, dict):
        return None
    return obj.get("markdown") or (obj.get("data") or {}).get("markdown")


def _firecrawl_scrape(url: str) -> str | None:
    """Scrape a single URL with Firecrawl v2 -> clean markdown (renders JS)."""
    try:
        r = httpx.post(f"{settings.firecrawl_api_base}/scrape", headers=_fc_headers(),
                       json={"url": url, "formats": ["markdown"], "onlyMainContent": True}, timeout=45)
        if r.status_code != 200:
            log.warning("firecrawl scrape %s -> HTTP %s: %s", url, r.status_code, r.text[:200])
            return None
        md = _fc_markdown(r.json())
        if not md:
            log.warning("firecrawl scrape %s -> no markdown in response: %s", url, r.text[:200])
            return None
        return redact_secrets(md)
    except Exception as e:  # noqa: BLE001
        log.warning("firecrawl scrape error for %s: %s", url, e)
        return None


def _firecrawl_crawl(url: str, limit: int) -> list[str]:
    """Crawl a site with Firecrawl v2 (bounded by ``limit`` pages) -> page markdowns.

    Submits a crawl job then polls until it completes or ``FIRECRAWL_TIMEOUT`` elapses.
    """
    import time

    try:
        r = httpx.post(f"{settings.firecrawl_api_base}/crawl", headers=_fc_headers(),
                       json={"url": url, "limit": limit,
                             "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True}},
                       timeout=45)
        if r.status_code not in (200, 201):
            log.warning("firecrawl crawl submit %s -> HTTP %s: %s", url, r.status_code, r.text[:200])
            return []
        job = r.json()
        if job.get("data"):                       # some responses return pages synchronously
            return _extract_crawl_pages(job)
        crawl_id = job.get("id")
        if not crawl_id:
            log.warning("firecrawl crawl submit returned no job id: %s", str(job)[:200])
            return []
        deadline = time.time() + settings.firecrawl_timeout
        while time.time() < deadline:
            time.sleep(2)
            pr = httpx.get(f"{settings.firecrawl_api_base}/crawl/{crawl_id}", headers=_fc_headers(), timeout=30)
            if pr.status_code != 200:
                log.warning("firecrawl crawl poll -> HTTP %s: %s", pr.status_code, pr.text[:200])
                break
            pj = pr.json()
            status = pj.get("status")
            if status in ("completed", "complete"):
                pages = _extract_crawl_pages(pj)
                log.info("firecrawl crawl of %s completed: %d page(s)", url, len(pages))
                return pages
            if status == "failed":
                log.warning("firecrawl crawl of %s failed: %s", url, str(pj)[:200])
                return []
        log.warning("firecrawl crawl of %s timed out after %ds", url, settings.firecrawl_timeout)
        return []
    except Exception as e:  # noqa: BLE001
        log.warning("firecrawl crawl error for %s: %s", url, e)
        return []


def _extract_crawl_pages(payload: dict) -> list[str]:
    pages: list[str] = []
    for item in (payload.get("data") or []):
        md = _fc_markdown(item)
        if md:
            pages.append(redact_secrets(md))
    return pages


def _scrape_httpx(url: str) -> str | None:
    """Fetch + extract readable text with httpx + BeautifulSoup (no JS rendering)."""
    try:
        from bs4 import BeautifulSoup

        r = httpx.get(url, headers=_HEADERS, timeout=10, follow_redirects=True)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        return redact_secrets(" ".join(soup.get_text(separator=" ").split()))
    except Exception:  # noqa: BLE001
        return None


def _scrape(url: str) -> str | None:
    """Scrape a single page — Firecrawl when enabled (renders JS), else httpx."""
    if _use_firecrawl():
        return _firecrawl_scrape(url)
    return _scrape_httpx(url)


def _extract_entity(query: str) -> str | None:
    """Extract the main place/organization the question is about (LLM, best-effort).

    Used when no entity was resolved upstream, so domain discovery searches for
    e.g. "Cairo University" rather than the whole question — which otherwise
    surfaces social/aggregator pages instead of the official site.
    """
    from agents.llm_client import get_llm, LLMError

    try:
        out = get_llm().complete_json(
            [{"role": "user", "content": (
                "Extract the single main place or organization this question is about. "
                'Return ONLY JSON: {"entity": "<name>"} or {"entity": null} if none.\n\n'
                f"Question: {query}"
            )}]
        )
        entity = out.get("entity")
        return entity.strip() if isinstance(entity, str) and entity.strip() else None
    except (LLMError, Exception):  # noqa: BLE001 — extraction is optional
        return None


def _discover_domain(entity: str) -> str | None:
    """Resolve an entity's official domain, ignoring social/aggregator sites.

    Searches candidates, drops blocked domains (LinkedIn, Facebook, Wikipedia,
    …), and picks the highest-scoring remaining domain (favouring .edu/.gov and
    domains containing the entity name). Cached for reuse.
    """
    cached = _get_cached_domain(entity)
    if cached and not _is_blocked_domain(cached):
        return cached
    if cached:
        # A previously-cached social/aggregator domain — ignore and re-resolve.
        log.info("ignoring stale blocked domain '%s' cached for '%s'", cached, entity)

    results = _tavily_search(f"{entity} official website")
    candidates: list[str] = []
    for r in results:
        dom = _root_domain(urlparse(r.get("url", "")).netloc)
        if dom and not _is_blocked_domain(dom) and dom not in candidates:
            candidates.append(dom)
    if not candidates:
        log.info("no official (non-social) domain found for '%s'", entity)
        return None

    best = max(candidates, key=lambda d: _score_domain(d, entity))
    log.info("resolved official domain for '%s' -> %s (from %s)", entity, best, candidates)
    _cache_domain(entity, best)
    return best


def _relevant_context_from_pages(query: str, texts: list[str], top_k: int = 5) -> str:
    """Chunk + ephemerally-embed scraped pages and keep the most relevant chunks.

    Falls back to a simple truncation if embedding is unavailable, so web
    fallback still works without the sentence-transformer model present.
    """
    from agents.chunker import _split_text, DEFAULT_SEPARATORS

    pieces: list[str] = []
    for t in texts:
        pieces.extend(_split_text(t, settings.chunk_size, settings.chunk_overlap, DEFAULT_SEPARATORS))
    if not pieces:
        return ""
    try:
        from agents.embeddings import embed_query, embed_passages

        qv = embed_query(query)
        pvs = embed_passages(pieces)
        scored = sorted(
            zip(pieces, pvs),
            key=lambda pp: sum(a * b for a, b in zip(qv, pp[1])),
            reverse=True,
        )
        return "\n\n".join(p for p, _ in scored[:top_k])
    except Exception:  # noqa: BLE001 — embedding optional for web tier
        return "\n\n---\n\n".join(p[:2000] for p in pieces[:top_k])


def _firecrawl_site_context(query: str, website: str) -> str:
    """Scrape the official site with Firecrawl, then CRAWL deeper until relevant
    content for ``query`` is retrieved.

    Strategy: scrape the landing page first (cheap). If it already contains
    query-relevant content, stop there. Otherwise crawl the site (bounded by
    ``FIRECRAWL_CRAWL_LIMIT``) and pick the most relevant chunks across all pages.
    """
    from agents.retrieval_validator import keyword_overlap

    texts: list[str] = []
    landing = _firecrawl_scrape(website)
    if landing:
        texts.append(landing)

    # Enough already? (landing page mentions what the question asks about)
    enough = bool(landing) and keyword_overlap(query, landing) >= 0.15
    if not enough:
        log.info("firecrawl: landing page insufficient -> crawling %s (limit %d)",
                 website, settings.firecrawl_crawl_limit)
        texts.extend(_firecrawl_crawl(website, settings.firecrawl_crawl_limit))

    if not texts:
        return ""
    return _relevant_context_from_pages(query, texts)


def fetch_web_context(query: str, *, entity: str | None = None, website: str | None = None) -> WebContext:
    """Assemble official-source context for ``query``.

    Args:
        query: the user's question.
        entity: the place/entity name (used for domain discovery + caching).
        website: the feature's own ``website`` URL from GeoJSON, if any — tried
            first as the highest-trust source.
    """
    # Prefer an explicit entity; otherwise extract one so domain discovery
    # searches for the entity name, not the whole question.
    subject = entity or _extract_entity(query) or query
    if subject != (entity or query):
        log.info("web fallback entity resolved to '%s'", subject)

    # Tier 0 — the feature's own official website (from GeoJSON).
    if website:
        if _use_firecrawl():
            # Firecrawl: scrape the site, crawling deeper until relevant content is found.
            log.info("web fallback tier 0: Firecrawl crawl of feature website %s", website)
            context = _firecrawl_site_context(query, website)
            if context:
                return WebContext(context=context, sources=[website], source_tier="official_website")
        else:
            log.info("web fallback tier 0: scraping feature website %s", website)
            page = _scrape(website)
            extra: list[str] = []
            if page:
                # Also try an "about"/"stats" page under the same domain via search.
                if settings.tavily_api_key:
                    domain = urlparse(website).netloc
                    try:
                        for r in _tavily_search(query, include_domains=[domain], max_results=settings.web_max_pages):
                            t = _scrape(r["url"])
                            if t:
                                extra.append(t)
                    except WebFallbackError:
                        pass
                context = _relevant_context_from_pages(query, [page, *extra])
                if context:
                    return WebContext(context=context, sources=[website], source_tier="official_website")

    if not settings.web_fallback_enabled:
        return WebContext(context="", sources=[], source_tier="disabled")

    # Tier A/B — discover the official domain, then search restricted to it.
    try:
        domain = _discover_domain(subject)
        if domain:
            log.info("web fallback tier A/B: official domain '%s' for '%s'", domain, subject)
            results = _tavily_search(query, include_domains=[domain], max_results=settings.web_max_pages)
            texts = [t for r in results if (t := _scrape(r.get("url", "")))]
            if texts:
                context = _relevant_context_from_pages(query, texts)
                if context:
                    return WebContext(
                        context=context,
                        sources=[r["url"] for r in results[: settings.web_max_pages]],
                        source_tier="official_domain",
                    )

        # Tier C — last-resort unrestricted search, clearly lower trust.
        log.info("web fallback tier C: unrestricted search (lower trust)")
        results = _tavily_search(query, max_results=settings.web_max_pages)
        texts = [t for r in results if (t := _scrape(r.get("url", "")))]
        context = _relevant_context_from_pages(query, texts) if texts else ""
        return WebContext(
            context=context,
            sources=[r["url"] for r in results[: settings.web_max_pages]] if results else [],
            source_tier="web_search",
        )
    except WebFallbackError as e:
        log.warning("web fallback unavailable: %s", e)
        return WebContext(context="", sources=[], source_tier="unavailable")
