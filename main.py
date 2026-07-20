"""
main.py — FastAPI application wiring the spatial + RAG pipelines together.

Endpoints
---------
Datasets (spatial):
    POST   /datasets/upload      ingest a GeoJSON/CSV into PostGIS
    POST   /datasets/describe    attach per-column descriptions
    GET    /datasets             list datasets
    DELETE /datasets/{id}        drop a dataset
Documents (RAG):
    POST   /documents/upload     ingest a PDF/DOCX/TXT/MD/HTML/CSV into Qdrant
    GET    /documents            list ingested documents
    DELETE /documents/{id}       remove a document from index + metadata
Query:
    POST   /chat                 legacy spatial-only NL->SQL (kept as-is)
    POST   /query                intent-routed: MAP / KNOWLEDGE / HYBRID / UNKNOWN
Enrichment / misc:
    POST   /enrich               place enrichment card (Wikipedia + Places)
    GET    /layers               layer metadata

The /query endpoint is the new front door: it classifies intent, runs the
spatial pipeline and/or the grounded RAG pipeline accordingly, and returns map
features, a natural-language answer, citations, evaluation scores and guardrail
results in one response.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import registry
import ingest
from llm import generate_sql
from validator import validate_sql
from db import run_query, ensure_postgis
from rag import enrich_place

from agents import guardrails
from agents import doc_ingest
from agents import vector_store
from agents import column_describer
from agents.hybrid_retriever import invalidate_bm25_cache
from agents.intent_router import classify_intent, MAP, KNOWLEDGE, HYBRID, UNKNOWN
from agents.knowledge_pipeline import answer_knowledge_query
from agents.memory import get_memory, reset_memory, condense_query
from agents import semantic_cache
from agents import embeddings
from agents.logging_config import get_logger, setup_logging, snippet
from config import settings

setup_logging()
logger = get_logger("api")

app = FastAPI(title="GeoChat — Hybrid Geospatial RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Legacy keyword screen kept for /chat; /query uses the fuller guardrail layer.
BLOCKED_INTENT = ["drop", "delete", "truncate", "alter", "insert", "update", "remove", "destroy"]


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return unhandled errors as JSON *with CORS headers*.

    Two problems this solves:
      1. Starlette's default 500 is plain text, which the frontend can't parse
         (it surfaced as a misleading "could not reach the server").
      2. The server-error middleware sits OUTSIDE CORSMiddleware, so a 500 would
         normally ship without `Access-Control-Allow-Origin`. The browser then
         blocks the response and reports "Failed to fetch", hiding the real
         cause. Setting the header here lets the browser read the error.
    """
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin",
        },
    )


@app.on_event("startup")
def startup() -> None:
    """Idempotently provision every backing store the app needs."""
    logger.info("startup: provisioning backing stores")
    ensure_postgis()
    registry.init_registry()
    try:
        doc_ingest.init_documents_table()
    except Exception as e:  # noqa: BLE001 — never block spatial startup on RAG stores
        logger.warning("RAG metadata tables unavailable at startup: %s", e)
    try:
        vector_store.ensure_collection()
    except Exception as e:  # noqa: BLE001 — Qdrant may not be running yet
        logger.warning("Qdrant collection not ready at startup: %s", e)
    logger.info("startup complete — %d document chunk(s) indexed", vector_store.count())


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    message: str
    dataset_ids: list[str]


class QueryRequest(BaseModel):
    message: str
    dataset_ids: list[str] = []
    collection: str | None = None          # metadata filter for RAG retrieval
    website: str | None = None             # focused feature's official website
    entity_focus: str | None = None        # override entity for web fallback
    allow_web: bool = True
    web_confirmed: bool = False            # user approved scraping the official site
    session_id: str | None = None          # conversation window key (memory)


class ResetMemoryRequest(BaseModel):
    session_id: str | None = None


class EnrichRequest(BaseModel):
    name: str
    name_en: str | None = None
    place_type: str = "place"
    wikipedia: str | None = None
    wikidata: str | None = None


class ColumnDescription(BaseModel):
    column_name: str
    data_type: str
    description: str = ""


class DescribeColumnsRequest(BaseModel):
    dataset_id: str
    columns: list[ColumnDescription]


# --------------------------------------------------------------------------- #
# dataset (spatial) endpoints — unchanged behavior
# --------------------------------------------------------------------------- #

@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), display_name: str = Form(...)):
    raw = await file.read()
    try:
        result = ingest.ingest_file(file.filename, raw, display_name)
    except ingest.IngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error during ingestion: {e}")
    semantic_cache.invalidate("dataset uploaded")  # map answers may change
    return result


@app.post("/datasets/describe")
def describe_columns(req: DescribeColumnsRequest):
    registry.set_column_descriptions(req.dataset_id, [c.model_dump() for c in req.columns])
    return {"status": "ok"}


@app.post("/datasets/describe/auto")
def auto_describe_columns(req: DescribeColumnsRequest):
    """Draft LLM descriptions for a dataset's columns (not saved — for review)."""
    try:
        columns = column_describer.generate_descriptions(
            req.dataset_id, [c.model_dump() for c in req.columns]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not auto-generate descriptions: {e}")
    return {"columns": columns}


@app.get("/datasets")
def list_datasets():
    return registry.list_datasets()


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    if not registry.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    semantic_cache.invalidate("dataset deleted")
    return {"status": "deleted"}


# --------------------------------------------------------------------------- #
# document (RAG) endpoints
# --------------------------------------------------------------------------- #

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    collection: str | None = Form(None),
):
    """Ingest a document into the RAG index (parse -> chunk -> embed -> Qdrant)."""
    raw = await file.read()
    logger.info("POST /documents/upload — '%s' (%d bytes)", file.filename, len(raw))
    try:
        result = doc_ingest.ingest_document(file.filename, raw, title=title, collection=collection)
    except doc_ingest.IngestError as e:
        logger.warning("document ingestion rejected '%s': %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("document ingestion failed for '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {e}")
    invalidate_bm25_cache()  # keyword index must see the new chunks
    semantic_cache.invalidate("document uploaded")  # answers may change
    return {
        "document_id": result.document_id,
        "title": result.title,
        "source": result.source,
        "filetype": result.filetype,
        "page_count": result.page_count,
        "chunk_count": result.chunk_count,
        "upload_date": result.upload_date,
        "injection_flags": result.injection_flags,
    }


@app.get("/documents")
def list_documents():
    return doc_ingest.list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if not doc_ingest.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    invalidate_bm25_cache()
    semantic_cache.invalidate("document deleted")
    return {"status": "deleted"}


@app.post("/documents/purge-orphans")
def purge_orphans():
    """Delete Qdrant chunks whose document is no longer registered (cleanup)."""
    try:
        result = doc_ingest.purge_orphan_chunks()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Purge failed: {e}")
    invalidate_bm25_cache()
    semantic_cache.invalidate("orphan chunks purged")
    return result


# --------------------------------------------------------------------------- #
# spatial helper (shared by /chat and /query)
# --------------------------------------------------------------------------- #

def _run_spatial(message: str, dataset_ids: list[str]) -> dict:
    """Generate, validate and execute a spatial SQL query. Returns {sql, results}."""
    if not dataset_ids:
        raise HTTPException(status_code=400, detail="Select at least one dataset to query on the map.")
    try:
        sql = generate_sql(message, dataset_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 — LLM/provider failure (e.g. bad API key)
        logger.exception("spatial: SQL generation failed")
        raise HTTPException(status_code=502, detail=f"SQL generation failed (LLM error): {e}")
    logger.info("spatial: generated SQL -> %s", snippet(sql, 200))

    allowed_tables = registry.get_table_names(dataset_ids)
    valid, reason = validate_sql(sql, allowed_tables=allowed_tables)
    if not valid:
        logger.warning("spatial: SQL validator rejected query — %s", reason)
        raise HTTPException(status_code=400, detail=reason)
    try:
        rows = run_query(sql)
    except Exception as e:  # noqa: BLE001
        logger.exception("spatial: query execution failed")
        raise HTTPException(status_code=500, detail=str(e))
    logger.info("spatial: %d row(s) returned", len(rows))
    return {"sql": sql, "results": rows}


@app.post("/chat")
def chat(req: ChatRequest):
    """Legacy spatial-only endpoint (kept for backwards compatibility)."""
    lower = req.message.lower()
    for word in BLOCKED_INTENT:
        if word in lower:
            raise HTTPException(status_code=400, detail=f"Query intent not allowed: '{word}'")
    return _run_spatial(req.message, req.dataset_ids)


# --------------------------------------------------------------------------- #
# routed query endpoint — the new front door
# --------------------------------------------------------------------------- #

@app.post("/query")
def query(req: QueryRequest):
    """Classify intent and route to the map and/or RAG pipelines.

    Returns a unified payload: ``intent`` metadata, optional ``map`` (SQL +
    GeoJSON-ready rows), optional ``knowledge`` (grounded answer + citations +
    evaluation + guardrail results).
    """
    logger.info("POST /query — '%s' (datasets=%d, session=%s)",
                snippet(req.message), len(req.dataset_ids), req.session_id or "default")

    # --- input guardrail -------------------------------------------------
    guard = guardrails.check_input(req.message)
    if not guard.allowed:
        raise HTTPException(status_code=400, detail={"error": "blocked_by_guardrail", **guard.as_dict()})

    # --- conversational memory (ConversationBufferWindowMemory) ----------
    # Rewrite a follow-up into a standalone question using the last k turns, so
    # intent classification, spatial SQL and RAG retrieval all see a
    # self-contained query with pronouns resolved ("does it deliver?" ->
    # "does Cilantro deliver?"). The turn is saved at the end.
    memory = get_memory(req.session_id)
    effective_query = condense_query(memory, req.message)

    # --- semantic cache lookup (before any LLM / retrieval work) ----------
    # Embed the (condensed) query and look for a semantically-equivalent past
    # question in the same dataset/collection scope. A hit returns the stored
    # response and skips intent classification, SQL, retrieval and generation.
    cache = semantic_cache.get_cache()
    scope = semantic_cache.scope_key(req.dataset_ids, req.collection)
    query_vec = _embed_for_cache(effective_query) if settings.cache_enabled else None
    if query_vec is not None:
        hit = cache.get(query_vec, scope)
        if hit is not None:
            cached, sim, matched = hit
            logger.info("/query CACHE HIT (sim=%.3f, matched=%s) -> skipping pipeline",
                        sim, snippet(matched))
            response = dict(cached)
            response["cache_hit"] = True
            response["cache_similarity"] = round(sim, 3)
            memory.add(req.message, _reply_for_memory(response))
            return response

    intent = classify_intent(effective_query)
    response: dict = {"intent": intent.as_dict()}
    if effective_query != req.message:
        response["resolved_query"] = effective_query

    if intent.intent == UNKNOWN:
        logger.info("/query -> UNKNOWN, asking for clarification")
        response["clarification"] = intent.clarifying_question
        memory.add(req.message, intent.clarifying_question or "")
        return response

    def _run_knowledge():
        return answer_knowledge_query(
            effective_query,
            entity_focus=req.entity_focus or intent.entity_focus,
            website=req.website,
            filters={"collection": req.collection} if req.collection else None,
            allow_web=req.allow_web,
            dataset_ids=req.dataset_ids,
            web_confirmed=req.web_confirmed,
        )

    map_empty = False
    if intent.needs_map:
        try:
            map_result = _run_spatial(effective_query, req.dataset_ids)
            response["map"] = map_result
            map_empty = not map_result.get("results")
        except HTTPException as e:
            # Don't hard-fail: record the error and let the knowledge fallback try.
            response["map"] = {"error": e.detail}
            map_empty = True

    ran_knowledge = False
    if intent.needs_rag:
        response["knowledge"] = _run_knowledge().as_dict()
        ran_knowledge = True

    # Fallback: a MAP query that plotted nothing (or failed) often means the
    # asked-for attribute lives in documents, not the dataset columns. Try the
    # knowledge route so the user still gets an answer regardless of routing.
    if intent.intent == MAP and map_empty and not ran_knowledge:
        logger.info("/query MAP returned no results -> knowledge fallback")
        kanswer = _run_knowledge()
        if kanswer.found:
            response["knowledge"] = kanswer.as_dict()

    # --- record the interaction in the window ----------------------------
    # Skip the intermediate human-in-the-loop confirmation prompt so it doesn't
    # pollute the window (the query is re-sent with web_confirmed=true anyway).
    kn = response.get("knowledge")
    if not (isinstance(kn, dict) and kn.get("needs_web_confirmation")):
        memory.add(req.message, _reply_for_memory(response))

    # --- store a useful answer in the semantic cache ---------------------
    if query_vec is not None and _is_cacheable(response):
        cache.put(query_vec, scope, dict(response), effective_query)

    logger.info("/query done — intent=%s, map=%s, knowledge=%s",
                intent.intent, "map" in response, "knowledge" in response)
    return response


def _embed_for_cache(text: str) -> list[float] | None:
    """Embed a query for cache keying; returns None (cache-skipped) on failure."""
    try:
        return embeddings.embed_query(text)
    except Exception as e:  # noqa: BLE001 — never fail a request because caching couldn't embed
        logger.warning("cache: could not embed query (%s) — proceeding without cache", e)
        return None


def _is_cacheable(response: dict) -> bool:
    """Only cache responses that carry a real answer (not clarifications, HITL
    prompts, errors, or empty results)."""
    if response.get("clarification"):
        return False
    kn = response.get("knowledge")
    if isinstance(kn, dict):
        if kn.get("needs_web_confirmation"):
            return False
        if kn.get("source_tier") in ("error", "awaiting_confirmation"):
            return False
    has_map = isinstance(response.get("map"), dict) and bool(response["map"].get("results"))
    has_knowledge = isinstance(kn, dict) and bool(kn.get("found"))
    return bool(has_map or has_knowledge)


def _reply_for_memory(response: dict) -> str:
    """Best-effort assistant text to store for a turn (for follow-up context)."""
    kn = response.get("knowledge")
    if isinstance(kn, dict) and kn.get("answer"):
        return str(kn["answer"])
    if response.get("clarification"):
        return str(response["clarification"])
    mp = response.get("map")
    if isinstance(mp, dict) and mp.get("results") is not None:
        return f"(mapped {len(mp.get('results') or [])} feature(s))"
    return ""


@app.post("/memory/reset")
def memory_reset(req: ResetMemoryRequest):
    """Clear a session's conversation window (start a fresh conversation)."""
    reset_memory(req.session_id)
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# enrichment / misc
# --------------------------------------------------------------------------- #

@app.post("/enrich")
async def enrich(req: EnrichRequest):
    try:
        return await enrich_place(
            name=req.name,
            name_en=req.name_en,
            place_type=req.place_type,
            wikipedia_tag=req.wikipedia,
            wikidata=req.wikidata,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/layers")
def get_layers():
    return run_query("SELECT * FROM layer_metadata")


@app.get("/health")
def health():
    """Liveness + backing-store readiness snapshot."""
    return {
        "status": "ok",
        "documents_indexed": vector_store.count(),
        "cache": semantic_cache.get_cache().stats() if settings.cache_enabled else {"enabled": False},
    }
