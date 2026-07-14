"""
rag.py — Wikipedia + Local Places RAG for place enrichment.

Search order:
  1. OSM wikipedia tag (most reliable)
  2. Arabic Wikipedia search with Egypt context
  3. English name + "Egypt" context search
  4. Google Places API (if key exists) OR OSM Nominatim API fallback
"""

import httpx
import json
import os
import asyncio
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

groq = AsyncGroq(api_key=os.getenv("API_KEY"))
GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_API_KEY") # Add this to your .env file
MODEL = "llama-3.3-70b-versatile"

_cache: dict[str, dict] = {}

HEADERS = {"User-Agent": "GeoChat/1.0 (educational spatial project) python-httpx"}

EN_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
AR_SUMMARY = "https://ar.wikipedia.org/api/rest_v1/page/summary/{title}"
EN_SEARCH  = "https://en.wikipedia.org/w/api.php"
AR_SEARCH  = "https://ar.wikipedia.org/w/api.php"

MIN_LEN = 150
BAD_SIGNALS = ["may refer to", "disambiguation", "versioning", "can refer to"]


def _is_good(text: str | None) -> bool:
    if not text or len(text) < MIN_LEN:
        return False
    tl = text.lower()
    return not any(b in tl for b in BAD_SIGNALS)


async def _fetch_en(title: str, client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get(EN_SUMMARY.format(title=title.replace(" ", "_")))
        if r.status_code == 200:
            text = r.json().get("extract", "")
            return text if _is_good(text) else None
    except Exception:
        pass
    return None


async def _fetch_ar(title: str, client: httpx.AsyncClient) -> tuple[str | None, str | None]:
    try:
        r = await client.get(AR_SUMMARY.format(title=title.replace(" ", "_")))
        if r.status_code == 200:
            data = r.json()
            return data.get("extract", "") or None, data.get("wikibase_item")
    except Exception:
        pass
    return None, None


async def _wikidata_en_title(qid: str, client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json", timeout=8)
        if r.status_code == 200:
            sitelinks = r.json().get("entities", {}).get(qid, {}).get("sitelinks", {})
            return sitelinks.get("enwiki", {}).get("title")
    except Exception:
        pass
    return None


async def _search_ar(query: str, client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get(AR_SEARCH, params={
            "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3,
        })
        if r.status_code == 200:
            results = r.json().get("query", {}).get("search", [])
            for hit in results:
                ar_title = hit["title"]
                _ar_text, qid = await _fetch_ar(ar_title, client)
                if qid:
                    en_title = await _wikidata_en_title(qid, client)
                    if en_title:
                        text = await _fetch_en(en_title, client)
                        if text: return text
                if _ar_text and _is_good(_ar_text): return _ar_text
    except Exception:
        pass
    return None


async def _search_en(query: str, client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get(EN_SEARCH, params={
            "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3,
        })
        if r.status_code == 200:
            results = r.json().get("query", {}).get("search", [])
            for hit in results:
                text = await _fetch_en(hit["title"], client)
                if text: return text
    except Exception:
        pass
    return None


async def _get_wikipedia_text(name: str, name_en: str | None, wikipedia_tag: str | None, client: httpx.AsyncClient) -> str | None:
    if wikipedia_tag:
        lang, _, title = wikipedia_tag.partition(":")
        if lang == "en" and title:
            text = await _fetch_en(title, client)
            if text: return text
        elif lang == "ar" and title:
            _ar, qid = await _fetch_ar(title, client)
            if qid:
                en_title = await _wikidata_en_title(qid, client)
                if en_title:
                    text = await _fetch_en(en_title, client)
                    if text: return text

    ar_query = f"{name} مصر"
    text = await _search_ar(ar_query, client)
    if text: return text
    
    text = await _search_ar(name, client)
    if text: return text

    if name_en:
        text = await _search_en(f"{name_en} Egypt", client)
        if text: return text
        text = await _search_en(name_en, client)
        if text: return text
    return None

# --- NEW: LOCAL PLACES APIs ---

async def _fetch_google_places(query: str, client: httpx.AsyncClient) -> dict | None:
    """Fetches ratings, formatted address, and hours from Google Places API."""
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.nationalPhoneNumber,places.regularOpeningHours.weekdayDescriptions"
    }
    try:
        r = await client.post(url, headers=headers, json={"textQuery": f"{query} Egypt"}, timeout=8)
        if r.status_code == 200:
            places = r.json().get("places", [])
            if places: return places[0]
    except Exception as e:
        print(f"Google Places Error: {e}")
    return None

async def _fetch_osm_nominatim(query: str, client: httpx.AsyncClient) -> dict | None:
    """Fallback: Fetches OSM metadata (opening hours, website) via Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{query} Egypt", "format": "json", "addressdetails": 1, "extratags": 1, "limit": 1}
    try:
        r = await client.get(url, params=params, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            data = r.json()
            if data: return data[0]
    except Exception as e:
        print(f"Nominatim Error: {e}")
    return None


async def _synthesize(place_name: str, place_type: str, wiki_text: str | None, local_data: dict | None) -> dict:
    prompt = f"""You are a helpful geographic assistant about Egypt.
Below is information gathered about "{place_name}" (a {place_type} in Egypt).

Wikipedia Extract:
\"\"\"
{wiki_text[:2500] if wiki_text else "No Wikipedia data available."}
\"\"\"

Local Business/Map Data (Ratings, Address, Contact):
\"\"\"
{json.dumps(local_data, ensure_ascii=False) if local_data else "No local map data available."}
\"\"\"

Based ONLY on the above provided information, produce a JSON object:
- "summary": 2-3 sentence human-friendly summary in English.
- "highlights": list of 3-5 short interesting fact bullets. If rating, phone number, or opening hours are available in the Local Data, INCLUDE THEM as visually appealing bullets (e.g. '⭐ 4.5/5 User Rating', '📍 Address', etc.).
- "category": single word (University / Hospital / Mosque / School / Park / Pharmacy / etc.)
- "source": "Wikipedia & Google Places", "Wikipedia", "OpenStreetMap", or "Local AI" depending on what data was actually provided above.

Return ONLY valid JSON. No markdown, no extra text."""

    resp = await groq.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=600,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


async def enrich_place(
    name: str,
    name_en: str | None = None,
    place_type: str = "place",
    wikipedia_tag: str | None = None,
    wikidata: str | None = None,
) -> dict:
    cache_key = (name_en or name).lower().strip()
    if cache_key in _cache:
        return _cache[cache_key]

    display_name = name_en or name
    
    async with httpx.AsyncClient(timeout=12, headers=HEADERS) as client:
        # Fetch Wikipedia and Local Data concurrently
        wiki_task = _get_wikipedia_text(name, name_en, wikipedia_tag, client)
        
        if GOOGLE_PLACES_KEY:
            local_task = _fetch_google_places(display_name, client)
        else:
            local_task = _fetch_osm_nominatim(display_name, client)
            
        wiki_text, local_data = await asyncio.gather(wiki_task, local_task)

    # If absolutely nothing was found across both APIs
    if not wiki_text and not local_data:
        result = {
            "summary": f"No detailed information or local map data found for {display_name}.",
            "highlights": [],
            "category": place_type.capitalize(),
            "source": "None",
            "found": False,
        }
        _cache[cache_key] = result
        return result

    try:
        card = await _synthesize(display_name, place_type, wiki_text, local_data)
        card["found"] = True
        _cache[cache_key] = card
        return card
    except Exception as e:
        print(f"LLM Synthesis Error: {e}")
        # Fallback if the LLM JSON parsing fails
        result = {
            "summary": (wiki_text[:300] + "…") if wiki_text else "Local map data found but failed to synthesize.",
            "highlights": [],
            "category": place_type.capitalize(),
            "source": "Fallback Data",
            "found": True,
        }
        _cache[cache_key] = result
        return result