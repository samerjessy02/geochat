"""
agents/rag_agent.py — backs the RAG and LOOKUP intents (and their combination
inside HYBRID).

RAG intent  -> use_local=True,  use_web=False  -> only searches your own
               uploaded documents (agents/vector_store.py, Qdrant).
LOOKUP intent -> use_local=False, use_web=True  -> only does the live web
               lookup below (domain cache -> scrape -> generic fallback).
HYBRID with both -> use_local=True, use_web=True -> tries local first,
               falls back to web only if the local answer wasn't sufficient.

Local tier:
  Retrieves top-k chunks from Qdrant by cosine similarity, then runs an LLM
  relevance grading pass over them (CRAG-style corrective retrieval) rather
  than trusting raw vector distance alone.

Web tiers:
  Tier A (domain cache): resolve an entity's official website once (e.g.
    "Cairo University" -> "cu.edu.eg") via Tavily search, cache it in
    `domain_cache`, and reuse it on every future query about that entity.
  Tier B (site-restricted scrape): Tavily search restricted to that domain,
    scrape the top pages with BeautifulSoup, synthesize an answer.
  Tier C (generic fallback): if the official site turned up nothing usable,
    fall back to an unrestricted Tavily search + scrape.

Requires TAVILY_API_KEY in .env for the web tiers (https://tavily.com).
"""

import json
import os
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import text

from db import engine
from agents import vector_store

load_dotenv()

groq = Groq(api_key=os.getenv("API_KEY"))
MODEL = "openai/gpt-oss-20b"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"

HEADERS = {"User-Agent": "GeoChat/1.0 (educational spatial project) python-httpx"}


class RagAgentError(Exception):
    pass


# ---------------------------------------------------------------------------
# local tier (Qdrant)
# ---------------------------------------------------------------------------

def _grade_relevance(query: str, chunks: list[str]) -> list[bool]:
    """CRAG-style corrective grading: an LLM pass that flags which retrieved
    chunks are actually relevant, rather than trusting raw vector distance."""
    if not chunks:
        return []
    numbered = "\n\n".join(f"[{i}] {c[:500]}" for i, c in enumerate(chunks))
    prompt = (
        f'Query: "{query}"\n\nCandidate passages:\n{numbered}\n\n'
        "For each numbered passage, is it actually relevant and useful for answering the query? "
        'Return ONLY a JSON object: {"relevant": [true|false, ...]} in the same order as the passages.'
    )
    try:
        resp = groq.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        flags = json.loads(resp.choices[0].message.content).get("relevant", [])
        if len(flags) != len(chunks):
            return [True] * len(chunks)  # fail open rather than discard everything
        return [bool(f) for f in flags]
    except Exception:
        return [True] * len(chunks)


def _query_local_index(query: str, top_k: int = 5) -> list[dict]:
    hits = vector_store.similarity_search(query, top_k=top_k)
    if not hits:
        return []
    flags = _grade_relevance(query, [h["text"] for h in hits])
    return [h for h, keep in zip(hits, flags) if keep]


# ---------------------------------------------------------------------------
# web tier — domain cache
# ---------------------------------------------------------------------------

def ensure_domain_cache_table():
    """Call once at startup, same pattern as registry.init_registry()."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS domain_cache (
                entity_name TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                discovered_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))


def _get_cached_domain(entity_name: str) -> str | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT domain FROM domain_cache WHERE entity_name = :name"),
            {"name": entity_name.lower().strip()},
        ).fetchone()
    return row[0] if row else None


def _cache_domain(entity_name: str, domain: str):
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO domain_cache (entity_name, domain) VALUES (:name, :domain)
                     ON CONFLICT (entity_name)
                     DO UPDATE SET domain = EXCLUDED.domain, discovered_at = now()"""),
            {"name": entity_name.lower().strip(), "domain": domain},
        )


def _tavily_search(query: str, include_domains: list[str] | None = None, max_results: int = 5) -> list[dict]:
    if not TAVILY_API_KEY:
        raise RagAgentError("TAVILY_API_KEY is not configured — the LOOKUP intent is unavailable.")
    payload = {"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results}
    if include_domains:
        payload["include_domains"] = include_domains
    try:
        r = httpx.post(TAVILY_URL, json=payload, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        raise RagAgentError(f"Web search failed: {e}")


def _discover_domain(entity_name: str) -> str | None:
    cached = _get_cached_domain(entity_name)
    if cached:
        return cached

    results = _tavily_search(f"{entity_name} official website")
    if not results:
        return None

    domain = urlparse(results[0]["url"]).netloc
    if domain:
        _cache_domain(entity_name, domain)
    return domain or None


# ---------------------------------------------------------------------------
# web tier — scraping
# ---------------------------------------------------------------------------

_scrape_cache: dict[str, str] = {}


def _scrape_page(url: str) -> str | None:
    if url in _scrape_cache:
        return _scrape_cache[url]
    try:
        r = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        cleaned = " ".join(soup.get_text(separator=" ").split())
        _scrape_cache[url] = cleaned
        return cleaned
    except Exception:
        return None


def _scrape_search_results(results: list[dict], max_pages: int = 3) -> str:
    texts = []
    for hit in results[:max_pages]:
        page_text = _scrape_page(hit.get("url", ""))
        if page_text:
            texts.append(page_text[:4000])
    return "\n\n---\n\n".join(texts)


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------

def _synthesize(query: str, entity_focus: str | None, context: str, source_tier: str) -> dict:
    prompt = f"""Answer the user's query using ONLY the context below. If the context doesn't
actually answer it, say so honestly (found: false) rather than guessing.

User query: "{query}"
Entity focus: {entity_focus or "n/a"}

Context:
\"\"\"
{context[:6000] if context else "No context available."}
\"\"\"

Return ONLY a JSON object: {{"answer": "2-4 sentence answer", "found": true or false}}"""

    resp = groq.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content)
    return {
        "answer": parsed.get("answer", ""),
        "found": bool(parsed.get("found", False)),
        "source_tier": source_tier,
    }


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    entity_focus: str | None = None,
    use_local: bool = True,
    use_web: bool = True,
) -> dict:
    """Returns {"answer": str, "found": bool, "source_tier": str, "sources": [...]}.

    use_local controls the RAG (Qdrant) tier, use_web controls the LOOKUP
    tiers (domain cache -> scrape -> generic fallback). main.py sets these
    based on the router's resolved intent(s)."""
    subject = entity_focus or query

    if use_local:
        hits = _query_local_index(query)
        if hits:
            context = "\n\n".join(h["text"] for h in hits)
            result = _synthesize(query, entity_focus, context, "local_index")
            result["sources"] = sorted({h["source"] for h in hits if h.get("source")})
            if result["found"] or not use_web:
                return result

    if not use_web:
        return {
            "answer": "I couldn't find this in your uploaded documents.",
            "found": False,
            "source_tier": "local_index",
            "sources": [],
        }

    if not TAVILY_API_KEY:
        return {
            "answer": "I couldn't find this in the uploaded documents, and web lookup isn't "
                      "configured (missing TAVILY_API_KEY), so I can't look further.",
            "found": False,
            "source_tier": "none",
            "sources": [],
        }

    # Tier A/B — resolve the entity's official domain, then search restricted to it
    domain = _discover_domain(subject)
    if domain:
        results = _tavily_search(query, include_domains=[domain])
        if results:
            context = _scrape_search_results(results)
            if context:
                result = _synthesize(query, entity_focus, context, "domain_scrape")
                result["sources"] = [r["url"] for r in results[:3]]
                if result["found"]:
                    return result

    # Tier C — generic web fallback
    results = _tavily_search(query)
    context = _scrape_search_results(results) if results else ""
    result = _synthesize(query, entity_focus, context, "web_fallback")
    result["sources"] = [r["url"] for r in results[:3]] if results else []
    return result