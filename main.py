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

import json
import re

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
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
from agents import spatial_search
from agents import routing
from agents import geojson_validator as gjv
from agents import log_store
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
        n = registry.ensure_spatial_indexes()
        logger.info("spatial indexes ensured on %d dataset table(s)", n)
    except Exception as e:  # noqa: BLE001 — never block startup on index backfill
        logger.warning("could not backfill spatial indexes: %s", e)
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
    filter_polygon: dict | None = None     # active spatial filter — restrict all map SQL to it


class ResetMemoryRequest(BaseModel):
    session_id: str | None = None


class RouteRequest(BaseModel):
    # Either place names (geocoded against datasets) or explicit [lon, lat] pairs.
    from_place: str | None = None
    to_place: str | None = None
    from_lonlat: list[float] | None = None
    to_lonlat: list[float] | None = None
    mode: str = "auto"                      # auto | pedestrian | bicycle
    dataset_ids: list[str] = []


class PolygonSearchRequest(BaseModel):
    geometry: dict                          # GeoJSON Polygon / MultiPolygon
    feature_type: str | None = None         # e.g. "schools" (matched to a dataset)
    dataset_id: str | None = None           # or an explicit dataset id
    dataset_ids: list[str] = []             # scope of layers to resolve within
    mode: str = "intersects"                # "intersects" | "within"
    limit: int = 2000
    offset: int = 0


class ParsePolygonRequest(BaseModel):
    text: str                               # pasted GeoJSON / coords / WKT


class SearchAreaRequest(BaseModel):
    geometry: dict                          # active GeoJSON Polygon / MultiPolygon
    dataset_ids: list[str] = []             # layers to search (or all if empty)
    feature_types: list[str] | None = None  # optional: only these named layers
    mode: str = "intersects"                # "intersects" | "within"
    limit_per_layer: int = 2000


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
    is_primary_key: bool = False
    foreign_key: str | None = None


class DescribeColumnsRequest(BaseModel):
    dataset_id: str
    columns: list[ColumnDescription]


class ValidateAttributesRequest(BaseModel):
    validation_id: str
    required_fields: list[str] = []
    numeric_fields: list[str] = []


class ResolveRequest(BaseModel):
    validation_id: str
    kind: str
    feature_index: int | None = None
    field: str | None = None
    resolution: dict = {}          # {action, value?, geometry?}


class CommitRequest(BaseModel):
    validation_id: str
    display_name: str


class ValidateDescribeAutoRequest(BaseModel):
    validation_id: str
    display_name: str | None = None


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


# --------------------------------------------------------------------------- #
# validated upload pipeline (Stage 1–4 + HITL, commit gated on resolution)
# --------------------------------------------------------------------------- #

# In-memory validation sessions: validation_id -> {result, filename, required, numeric}
_validation_sessions: dict[str, dict] = {}


@app.post("/datasets/validate")
async def validate_dataset(file: UploadFile = File(...), schema: UploadFile | None = File(None)):
    """Stage 1–2 + 4 validation of an uploaded GeoJSON. Returns a report,
    inferred columns, and any HITL items to resolve before commit.

    If an optional ``schema`` JSON is provided, its column metadata (description,
    type, required, keys) is the source of truth — the LLM is never called to
    describe columns (see /datasets/validate/describe-auto)."""
    import uuid as _uuid
    raw = await file.read()
    logger.info("POST /datasets/validate — '%s' (%d bytes, schema=%s)",
                file.filename, len(raw), bool(schema))
    try:
        result = gjv.validate_upload(file.filename, raw)
    except gjv.FileReject as e:
        raise HTTPException(status_code=400, detail={"stage": 1, "error": str(e)})
    except Exception as e:  # noqa: BLE001
        logger.exception("validation crashed")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

    has_schema = False
    if schema is not None:
        sraw = await schema.read()
        if sraw and sraw.strip():
            sname = (schema.filename or "").lower()
            if sname and not sname.endswith(".json"):
                raise HTTPException(status_code=400, detail={"stage": 0,
                    "error": f"Schema must be a JSON file (.json) — got '{schema.filename}'."})
            try:
                schema_map = gjv.parse_schema(sraw)
            except gjv.FileReject as e:
                raise HTTPException(status_code=400, detail={"stage": 0, "error": str(e)})
            # Surface any coordinate columns the schema documents that live in the
            # Point geometry rather than in properties (e.g. longitude/latitude).
            existing = {c["name"].strip().lower() for c in result.columns}
            for extra in gjv.coordinate_columns_from_schema(schema_map, result.accepted):
                if extra["name"].strip().lower() not in existing:
                    result.columns.append(extra)
            gjv.enrich_columns_with_schema(result.columns, schema_map)
            has_schema = True
            logger.info("schema applied: %d field(s) — LLM description skipped", len(schema_map))

    vid = _uuid.uuid4().hex[:12]
    _validation_sessions[vid] = {"result": result, "filename": file.filename,
                                 "required": [], "numeric": [], "has_schema": has_schema}
    return {"validation_id": vid, "has_schema": has_schema, "report": result.report()}


def _get_session(vid: str) -> dict:
    s = _validation_sessions.get(vid)
    if not s:
        raise HTTPException(status_code=404, detail="Validation session not found (re-upload the file).")
    return s


@app.post("/datasets/validate/describe-auto")
def validate_describe_auto(req: ValidateDescribeAutoRequest):
    """Draft column descriptions from the uploaded features (no table exists yet).

    Unified metadata rule: a column that the schema already describes keeps that
    description (``source="schema"``) and the LLM is NOT invoked for it. Columns
    the schema does not describe — whether no schema was uploaded, or the schema
    only carried types/keys — are auto-generated by the LLM (``source="llm"``).
    So the LLM is called only for the columns that actually need it, and skipped
    entirely when the schema documents every column."""
    s = _get_session(req.validation_id)
    result = s["result"]
    cols = result.columns
    if not cols:
        return {"source": "schema" if s.get("has_schema") else "llm", "columns": []}

    # Columns the schema already documents — kept verbatim, never sent to the LLM.
    from_schema = {c["name"] for c in cols
                   if c.get("source") == "schema" and (c.get("description") or "").strip()}
    todo = [c for c in cols if c["name"] not in from_schema]

    by_name: dict[str, dict] = {}
    for c in cols:
        if c["name"] in from_schema:
            by_name[c["name"]] = {"column_name": c["name"], "data_type": c["dtype"],
                                  "description": c.get("description", ""), "source": "schema"}

    if todo:
        payload = [{"column_name": c["name"], "data_type": c["dtype"]} for c in todo]
        try:
            gen = column_describer.describe_from_features(
                req.display_name or "dataset", result.accepted, payload)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not auto-generate descriptions: {e}")
        for g in gen:
            g["source"] = "llm"
            by_name[g["column_name"]] = g
        logger.info("describe-auto: %d column(s) from schema, %d generated by LLM",
                    len(from_schema), len(todo))
    else:
        logger.info("describe-auto skipped LLM — schema documents all %d column(s)", len(cols))

    # Preserve original column order.
    columns = [by_name[c["name"]] for c in cols if c["name"] in by_name]
    if not todo:
        src = "schema"
    elif from_schema:
        src = "mixed"
    else:
        src = "llm"
    return {"source": src, "columns": columns}


@app.post("/datasets/validate/attributes")
def validate_attributes(req: ValidateAttributesRequest):
    """Run Stage 3 attribute validation once the user picks required/numeric fields."""
    s = _get_session(req.validation_id)
    result = s["result"]
    # drop previous attribute-stage issues/HITL so re-submitting is idempotent
    result.issues = [i for i in result.issues if i.stage != 3]
    result.hitl = [h for h in result.hitl if h.kind not in ("missing_attribute", "outlier")]
    s["required"], s["numeric"] = req.required_fields, req.numeric_fields
    gjv.stage3_attributes(result.accepted, req.required_fields, req.numeric_fields,
                          result.issues, result.hitl)
    return {"validation_id": req.validation_id, "report": result.report()}


@app.post("/datasets/validate/resolve")
def validate_resolve(req: ResolveRequest):
    """Record a user's decision on one HITL item."""
    s = _get_session(req.validation_id)
    result = s["result"]
    matched = None
    for h in result.hitl:
        if h.kind == req.kind and h.feature_index == req.feature_index and h.field == req.field:
            matched = h
            break
    if matched is None:
        raise HTTPException(status_code=404, detail="HITL item not found.")
    matched.resolved = True
    matched.resolution = req.resolution
    # A CRS decision unblocks reprojection.
    if matched.kind == "crs" and req.resolution.get("action") == "set_epsg":
        epsg = int(req.resolution.get("epsg") or 4326)
        if epsg != 4326:
            try:
                gjv._reproject([f for f in result.accepted], epsg)
                result.reprojected = True
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Reprojection from EPSG:{epsg} failed: {e}")
        result.crs_epsg = epsg
    return {"validation_id": req.validation_id, "report": result.report(),
            "all_resolved": gjv.all_hitl_resolved(result)}


@app.post("/datasets/validate/commit")
def validate_commit(req: CommitRequest):
    """Commit the validated dataset — only when every HITL item is resolved."""
    s = _get_session(req.validation_id)
    result = s["result"]
    if not gjv.all_hitl_resolved(result):
        pending = [h.as_dict() for h in result.hitl if not h.resolved]
        raise HTTPException(status_code=409,
                            detail={"error": "hitl_pending", "pending": pending})
    features = gjv.finalize(result)
    if not features:
        raise HTTPException(status_code=400, detail="No features left to import after your decisions.")
    fc = {"type": "FeatureCollection", "features": features}
    import json as _json
    raw = _json.dumps(fc).encode("utf-8")
    fname = s["filename"] if s["filename"].lower().endswith((".geojson", ".json")) else "validated.geojson"
    try:
        out = ingest.ingest_file(fname, raw, req.display_name)
    except ingest.IngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Commit failed during ingestion: {e}")
    invalidate_bm25_cache()
    semantic_cache.invalidate("dataset committed")
    _validation_sessions.pop(req.validation_id, None)
    out["committed_features"] = len(features)
    out["validation_summary"] = result.summary()
    return out


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

def _clip_sql_to_polygon(sql: str, geometry: dict) -> str:
    """Wrap a validated SELECT so only rows whose ``geometry`` (the GeoJSON text
    every generated query emits as ``AS geometry``) intersect the active polygon
    are returned. The polygon is embedded as a JSON literal — safe because it is
    validated to contain only numbers/brackets (no quotes to break out of).
    """
    geom_json = json.dumps(geometry).replace("'", "")  # validated numeric GeoJSON
    inner = sql.strip().rstrip(";")
    return (
        "WITH __clip AS (SELECT ST_MakeValid(ST_SetSRID("
        f"ST_GeomFromGeoJSON('{geom_json}'), 4326)) AS g)\n"
        "SELECT __base.* FROM (\n" + inner + "\n) __base, __clip\n"
        "WHERE __base.geometry IS NOT NULL\n"
        "  AND ST_Intersects(ST_SetSRID(ST_GeomFromGeoJSON(__base.geometry), 4326), __clip.g)"
    )


def _run_spatial(message: str, dataset_ids: list[str],
                 filter_polygon: dict | None = None) -> dict:
    """Generate, validate and execute a spatial SQL query. Returns {sql, results}.

    When ``filter_polygon`` is set (an active spatial filter), the validated query
    is clipped so every returned feature falls inside that polygon."""
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

    exec_sql = sql
    clipped = False
    if filter_polygon:
        try:
            spatial_search._validate_geometry(filter_polygon)
            exec_sql = _clip_sql_to_polygon(sql, filter_polygon)
            clipped = True
            logger.info("spatial: clipping results to active filter polygon")
        except spatial_search.SpatialSearchError as e:
            raise HTTPException(status_code=400, detail=f"Invalid filter polygon: {e}")

    try:
        rows = run_query(exec_sql)
    except Exception as e:  # noqa: BLE001
        # If the clip wrapper failed (e.g. the query didn't expose a geometry
        # column), retry unclipped so the user still gets an answer.
        if clipped:
            logger.warning("spatial: clipped query failed (%s) — retrying unclipped", e)
            try:
                rows = run_query(sql)
                return {"sql": sql, "results": rows, "clipped": False,
                        "clip_error": "query could not be restricted to the area"}
            except Exception:  # noqa: BLE001
                pass
        logger.exception("spatial: query execution failed")
        raise HTTPException(status_code=500, detail=str(e))
    logger.info("spatial: %d row(s) returned%s", len(rows), " (clipped)" if clipped else "")
    return {"sql": sql, "results": rows, "clipped": clipped}


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

    # Routing/directions can't be answered with SQL — hand off to the routing
    # engine (Valhalla) instead of the NL->SQL path.
    if _looks_like_routing(effective_query):
        return _handle_route(effective_query, req.dataset_ids)

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
            map_result = _run_spatial(effective_query, req.dataset_ids, req.filter_polygon)
            response["map"] = map_result
            map_empty = not map_result.get("results")
        except HTTPException as e:
            # Don't hard-fail: record the error and let the knowledge fallback try.
            response["map"] = {"error": e.detail}
            map_empty = True

    ran_knowledge = False
    if intent.needs_rag:
        kn = _run_knowledge()
        response["knowledge"] = kn.as_dict()
        ran_knowledge = True
        # HYBRID: the spatial query couldn't apply a document-only filter (e.g.
        # ">40,000 students", "teaching hospital") so it plotted nothing. Show the
        # places the answer names by matching them against the selected map layers.
        if intent.needs_map and map_empty and kn.found:
            feats = _entities_from_answer(kn.answer, req.dataset_ids)
            if feats:
                logger.info("/query HYBRID: plotting %d entity(ies) named in the answer", len(feats))
                response["map"] = {"sql": "-- features matched from the knowledge answer",
                                   "results": feats, "from_answer": True}
                map_empty = False

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


def _looks_like_routing(q: str) -> bool:
    """Heuristic: is this a routing/directions request (needs the routing engine)?"""
    ql = q or ""
    if re.search(r"\b(route|directions|fastest route|shortest route|navigate)\b", ql, re.I):
        return True
    # "walk/drive from X to Y" style
    return bool(re.search(r"\b(walk|drive|driving|walking|cycle|bike)\b.*\bto\b", ql, re.I)
                and re.search(r"\bfrom\b", ql, re.I))


_ROUTE_INTENT = {"intent": "ROUTE", "needs_map": True, "needs_rag": False}


def _route_message(msg: str) -> dict:
    """A ROUTE response that just carries a chat message (no map)."""
    return {"intent": {**_ROUTE_INTENT, "needs_map": False}, "clarification": msg}


def _handle_route(query: str, dataset_ids: list[str]) -> dict:
    """Resolve a natural-language routing request and return a drawn route."""
    if not settings.routing_enabled:
        return _route_message("Routing is turned off (set ROUTING_ENABLED=true).")

    frm, to, mode = routing.parse_route_request(query)
    if not frm or not to:
        return _route_message(
            "Tell me the start and end, e.g. \"drive from Cairo University to City Mall\"."
        )
    if not routing.is_available():
        return _route_message(
            f"The routing engine (Valhalla) isn't reachable at {settings.valhalla_url}. "
            "Check the container is running and the port is mapped, or set VALHALLA_URL."
        )
    origin = routing.geocode_place(frm, dataset_ids)
    dest = routing.geocode_place(to, dataset_ids)
    missing = [n for n, r in ((frm, origin), (to, dest)) if not r]
    if missing:
        return _route_message(
            f"I couldn't find {' or '.join(repr(m) for m in missing)} in your selected "
            "datasets. Make sure the layer containing those places is added and selected."
        )
    try:
        rt = routing.route([(origin["lat"], origin["lon"]), (dest["lat"], dest["lon"])], mode)
    except routing.RoutingError as e:
        return _route_message(f"Couldn't plan that route: {e}")

    logger.info("/query route %s -> %s (%s): %.0f m, %.0f s",
                origin["name"], dest["name"], mode, rt["distance_m"], rt["duration_s"])
    return {"intent": dict(_ROUTE_INTENT), "route": {**rt, "from": origin, "to": dest}}


def _entities_from_answer(answer: str, dataset_ids: list[str]) -> list[dict]:
    """Return features of the selected layers whose name appears in ``answer``.

    Used for HYBRID queries where the qualifying attribute lives in a document
    (not a map column): the RAG answer names the matching places, and we plot
    exactly those by matching their name value inside the answer text.
    """
    text = (answer or "").lower()
    if not text:
        return []
    results: list[dict] = []
    seen: set = set()
    for ds in registry.get_datasets_by_ids(dataset_ids):
        table = ds["table_name"]  # machine name -> safe to interpolate
        cols = [c["column_name"] for c in (ds.get("columns") or [])]
        name_cols = [c for c in cols if c == "name" or "name" in c.lower()
                     or c in ("title", "label", "display_name")]
        if not name_cols:
            continue
        sel = ", ".join(f'"{c}"' for c in name_cols)
        try:
            rows = run_query(f'SELECT {sel}, ST_AsGeoJSON(wkb_geometry) AS geometry '
                             f'FROM "{table}" LIMIT 5000')
        except Exception:  # noqa: BLE001
            continue
        for r in rows:
            for nc in name_cols:
                val = r.get(nc)
                if val and len(str(val)) >= 5 and str(val).lower() in text:
                    key = (table, str(val))
                    if key not in seen:
                        seen.add(key)
                        results.append(r)
                    break
    return results


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
# draw-a-search-area: spatial search within a user-drawn polygon
# --------------------------------------------------------------------------- #

@app.post("/route")
def route_endpoint(req: RouteRequest):
    """Plan a route between two places (by name) or two [lon, lat] points."""
    if not settings.routing_enabled:
        raise HTTPException(status_code=503, detail="Routing is disabled (ROUTING_ENABLED=false).")
    if not routing.is_available():
        raise HTTPException(status_code=503,
                            detail=f"Routing engine not reachable at {settings.valhalla_url}.")

    def _resolve(place, lonlat, which):
        if lonlat and len(lonlat) == 2:
            return {"name": which, "lon": float(lonlat[0]), "lat": float(lonlat[1])}
        r = routing.geocode_place(place, req.dataset_ids) if place else None
        if not r:
            raise HTTPException(status_code=404, detail=f"Could not resolve {which} location {place!r}.")
        return r

    origin = _resolve(req.from_place, req.from_lonlat, "start")
    dest = _resolve(req.to_place, req.to_lonlat, "end")
    try:
        rt = routing.route([(origin["lat"], origin["lon"]), (dest["lat"], dest["lon"])], req.mode)
    except routing.RoutingError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {**rt, "from": origin, "to": dest}


@app.post("/spatial/search-by-polygon")
def search_by_polygon(req: PolygonSearchRequest):
    """Find features of ``feature_type`` that fall inside the drawn ``geometry``.

    Returns ``{count, bbox, features, ...}``. The polygon is passed to PostGIS as
    a bound parameter (never string-interpolated) and validated/repaired there.
    """
    logger.info("POST /spatial/search-by-polygon — type=%s, mode=%s, datasets=%d",
                req.feature_type, req.mode, len(req.dataset_ids))
    try:
        return spatial_search.search_by_polygon(
            req.geometry,
            feature_type=req.feature_type,
            dataset_id=req.dataset_id,
            dataset_ids=req.dataset_ids,
            mode=req.mode,
            limit=req.limit,
            offset=req.offset,
        )
    except spatial_search.SpatialSearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("polygon search failed")
        raise HTTPException(status_code=500, detail=f"Polygon search failed: {e}")


@app.post("/spatial/parse-polygon")
def parse_polygon(req: ParsePolygonRequest):
    """Validate pasted polygon coordinates (GeoJSON, coordinate list, or WKT) and
    return a normalized WGS84 GeoJSON polygon + bbox for the frontend to draw and
    set as the active spatial filter."""
    logger.info("POST /spatial/parse-polygon — %d chars", len(req.text or ""))
    try:
        return spatial_search.parse_polygon_text(req.text)
    except spatial_search.SpatialSearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("polygon parse failed")
        raise HTTPException(status_code=500, detail=f"Could not parse polygon: {e}")


@app.post("/spatial/search-area")
def search_area(req: SearchAreaRequest):
    """Search several layers inside the active polygon at once. Returns one entry
    per layer (count + features) so the frontend can color each distinctly and
    build a legend. "pharmacies and schools in this area" returns BOTH layers."""
    logger.info("POST /spatial/search-area — datasets=%d, types=%s, mode=%s",
                len(req.dataset_ids), req.feature_types, req.mode)
    try:
        return spatial_search.search_area(
            req.geometry,
            dataset_ids=req.dataset_ids,
            feature_types=req.feature_types,
            mode=req.mode,
            limit_per_layer=req.limit_per_layer,
        )
    except spatial_search.SpatialSearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("area search failed")
        raise HTTPException(status_code=500, detail=f"Area search failed: {e}")


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


# --------------------------------------------------------------------------- #
# activity logs (queryable structured log store)
# --------------------------------------------------------------------------- #

def _csv_list(v: str | None) -> list[str] | None:
    return [x for x in v.split(",") if x] if v else None


@app.get("/logs")
def get_logs(search: str | None = None, modules: str | None = None,
             levels: str | None = None, start: float | None = None,
             end: float | None = None, limit: int = 200, offset: int = 0):
    """Filtered activity logs (filters AND together; newest first)."""
    return log_store.query_logs(
        search=search, modules=_csv_list(modules), levels=_csv_list(levels),
        start=start, end=end, limit=limit, offset=offset,
    )


@app.get("/logs/modules")
def get_log_modules():
    return {"modules": log_store.distinct_modules()}


@app.get("/logs/export")
def export_logs(format: str = "json", search: str | None = None, modules: str | None = None,
                levels: str | None = None, start: float | None = None, end: float | None = None):
    """Download the filtered logs as JSON or CSV."""
    result = log_store.query_logs(
        search=search, modules=_csv_list(modules), levels=_csv_list(levels),
        start=start, end=end, limit=100000,
    )
    rows = result["logs"]
    if format == "csv":
        import csv
        import io
        buf = io.StringIO()
        cols = ["iso", "module", "level", "message", "target", "duration_ms", "error_type", "logger"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return Response(content=buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=geochat_logs.csv"})
    return JSONResponse(content=rows,
                        headers={"Content-Disposition": "attachment; filename=geochat_logs.json"})


@app.delete("/logs")
def clear_logs():
    return {"cleared": log_store.clear_logs()}


@app.get("/health")
def health():
    """Liveness + backing-store readiness snapshot."""
    return {
        "status": "ok",
        "documents_indexed": vector_store.count(),
        "cache": semantic_cache.get_cache().stats() if settings.cache_enabled else {"enabled": False},
    }
