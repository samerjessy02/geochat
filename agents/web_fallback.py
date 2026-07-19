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


def _firecrawl_scrape_full(url: str) -> tuple[str | None, list[str]]:
    """Scrape a URL with Firecrawl v2 -> (markdown, links on the page).

    The ``links`` format lets us follow the page's own links one level deeper
    (e.g. /menu -> /drinks) using the reliable synchronous scrape.
    """
    try:
        r = httpx.post(f"{settings.firecrawl_api_base}/scrape", headers=_fc_headers(),
                       json={"url": url, "formats": ["markdown", "links"], "onlyMainContent": True},
                       timeout=45)
        if r.status_code != 200:
            log.warning("firecrawl scrape %s -> HTTP %s: %s", url, r.status_code, r.text[:200])
            return None, []
        data = r.json().get("data") or r.json()
        md = data.get("markdown")
        raw_links = data.get("links") or []
        links = [lk if isinstance(lk, str) else lk.get("url")
                 for lk in raw_links if (isinstance(lk, str) or isinstance(lk, dict))]
        links = [lk for lk in links if lk]
        return (redact_secrets(md) if md else None), links
    except Exception as e:  # noqa: BLE001
        log.warning("firecrawl scrape error for %s: %s", url, e)
        return None, []


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


def _firecrawl_map(url: str, limit: int = 60) -> list[str]:
    """Discover a site's URLs with Firecrawl v2 Map (fast, synchronous)."""
    try:
        r = httpx.post(f"{settings.firecrawl_api_base}/map", headers=_fc_headers(),
                       json={"url": url, "limit": limit}, timeout=30)
        if r.status_code != 200:
            log.warning("firecrawl map %s -> HTTP %s: %s", url, r.status_code, r.text[:200])
            return []
        data = r.json()
        links = data.get("links") or (data.get("data") or {}).get("links") or []
        urls: list[str] = []
        for lk in links:
            u = lk if isinstance(lk, str) else (lk.get("url") if isinstance(lk, dict) else None)
            if u:
                urls.append(u)
        return urls
    except Exception as e:  # noqa: BLE001
        log.warning("firecrawl map error for %s: %s", url, e)
        return []


# URL-path keywords that commonly hold the kind of info users ask about.
_INFO_PATH_KEYWORDS = (
    "menu", "price", "pricing", "product", "about", "contact", "faq", "service",
    "location", "branch", "event", "news", "program", "academic", "admission", "hour",
    "drink", "beverage", "coffee", "tea", "food", "cafe", "shop", "store", "catalog",
)


def _rank_urls(query: str, website: str, urls: list[str]) -> list[str]:
    """Rank same-domain URLs by how relevant their path looks to the query."""
    q_tokens = {t for t in re.findall(r"[^\W\d_]+", query.lower()) if len(t) > 2}
    base_host = urlparse(website).netloc.lower().lstrip("www.")

    def score(u: str) -> float:
        p = urlparse(u)
        host = p.netloc.lower().lstrip("www.")
        if base_host and host and base_host not in host and host not in base_host:
            return -1.0  # off-domain
        path = (p.path + " " + p.query).lower()
        s = float(sum(1 for t in q_tokens if t in path))
        s += sum(1 for kw in _INFO_PATH_KEYWORDS if kw in path)
        s -= 0.1 * path.count("/")  # prefer shallower pages on ties
        return s

    scored = [(u, score(u)) for u in urls]
    scored = [x for x in scored if x[1] >= 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in scored]


def _distinctive_overlap(query: str, text: str, website: str) -> float:
    """Keyword overlap of the query's *distinctive* words against ``text``.

    Excludes the entity/site-name tokens (e.g. "starbucks", "eg") from the query,
    because they're trivially present on the entity's own site and would make a
    marketing homepage look like it answers a specific question when it doesn't.
    """
    from agents.retrieval_validator import keyword_overlap

    name_tokens = {t for t in re.findall(r"[^\W\d_]+", urlparse(website).netloc.lower())}
    stripped = " ".join(w for w in re.findall(r"[^\W\d_]+", query) if w.lower() not in name_tokens)
    return keyword_overlap(stripped, text)


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
    """Scrape the official site, then go deeper until relevant content is found.

    Strategy (all built on the synchronous scrape call, which is reliable):
      1. Scrape the landing page. If it already answers, stop.
      2. Otherwise MAP the site to discover its URLs, rank them by relevance to
         the question, and scrape the top few (bounded by FIRECRAWL_CRAWL_LIMIT),
         stopping early once a page clearly contains what the question asks about.

    Map + targeted scrape avoids the async /crawl job (submit + poll), which is
    slow and often doesn't finish within the timeout.
    """
    texts: list[str] = []
    scraped: set[str] = set()
    budget = [max(1, settings.firecrawl_crawl_limit)]   # overall scrape budget (mutable)

    def visit(u: str) -> tuple[list[str], bool]:
        """Scrape ``u`` once; append its text; return (its links, strong_match)."""
        key = u.rstrip("/")
        if key in scraped or budget[0] <= 0:
            return [], False
        scraped.add(key)
        budget[0] -= 1
        md, links = _firecrawl_scrape_full(u)
        if md:
            texts.append(md)
            if _distinctive_overlap(query, md, website) >= 0.6:
                log.info("firecrawl: relevant content found on %s", u)
                return links, True
        return links, False

    # 1) landing page
    _, strong = visit(website)
    if strong:
        return _relevant_context_from_pages(query, texts)

    # 2) map the site -> take only the TOP-N ranked entry pages
    entry = _rank_urls(query, website, _firecrawl_map(website))
    entry = [u for u in entry if u.rstrip("/") not in scraped][: settings.firecrawl_top_pages]
    log.info("firecrawl: exploring top %d page(s) of %s (deep %d, budget %d)",
             len(entry), website, settings.firecrawl_deep_links, budget[0])

    # 3) for each entry page, scrape it AND follow its most relevant links one level deeper
    for u in entry:
        links, strong = visit(u)
        if strong:
            return _relevant_context_from_pages(query, texts)
        children = _rank_urls(query, website, links)
        children = [c for c in children if c.rstrip("/") not in scraped][: settings.firecrawl_deep_links]
        for c in children:
            _, cstrong = visit(c)
            if cstrong:
                return _relevant_context_from_pages(query, texts)
        if budget[0] <= 0:
            break

    if not texts:
        return ""
    return _relevant_context_from_pages(query, texts)


def fetch_web_context(query: str, *, entity: str | None = None, website: str | None = None) -> WebContext:
    """Assemble context by scraping the feature's OFFICIAL website (from GeoJSON).

    The only web source is the ``website`` URL provided on the matched dataset
    feature — there is NO general web search or domain discovery. If no official
    website is available, the web tier returns nothing (the pipeline then refuses).

    Args:
        query: the user's question.
        entity: accepted for signature compatibility; not used here.
        website: the feature's own ``website`` URL from GeoJSON.
    """
    if not settings.web_fallback_enabled:
        return WebContext(context="", sources=[], source_tier="disabled")
    if not website:
        log.info("web fallback: no official website on the feature -> skipping web tier")
        return WebContext(context="", sources=[], source_tier="none")

    if _use_firecrawl():
        log.info("web fallback: Firecrawl scrape of official website %s", website)
        context = _firecrawl_site_context(query, website)
    else:
        log.info("web fallback: httpx scrape of official website %s", website)
        page = _scrape(website)
        context = _relevant_context_from_pages(query, [page]) if page else ""

    if context:
        return WebContext(context=context, sources=[website], source_tier="official_website")
    return WebContext(context="", sources=[website], source_tier="none")
