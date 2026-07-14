from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm import generate_sql
from validator import validate_sql
from db import run_query, ensure_postgis
from rag import enrich_place
import registry
import ingest

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BLOCKED_INTENT = ["drop", "delete", "truncate", "alter", "insert", "update", "remove", "destroy"]


@app.on_event("startup")
def startup():
    ensure_postgis()
    registry.init_registry()


class ChatRequest(BaseModel):
    message: str
    dataset_ids: list[str]   # which uploaded datasets this chat can query


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


@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    display_name: str = Form(...),
):
    """
    Upload a .geojson or .csv (with lat/lon columns) to create a new dataset.
    Returns the inferred columns so the frontend can show a form for the
    user to fill in per-column descriptions via /datasets/describe.
    """
    raw = await file.read()
    try:
        result = ingest.ingest_file(file.filename, raw, display_name)
    except ingest.IngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during ingestion: {e}")
    return result


@app.post("/datasets/describe")
def describe_columns(req: DescribeColumnsRequest):
    """User submits descriptions for the columns of a dataset they just uploaded."""
    registry.set_column_descriptions(
        req.dataset_id,
        [c.model_dump() for c in req.columns],
    )
    return {"status": "ok"}


@app.get("/datasets")
def list_datasets():
    return registry.list_datasets()


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    ok = registry.delete_dataset(dataset_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"status": "deleted"}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.dataset_ids:
        raise HTTPException(status_code=400, detail="Select at least one dataset to query.")

    lower = req.message.lower()
    for word in BLOCKED_INTENT:
        if word in lower:
            raise HTTPException(status_code=400, detail=f"Query intent not allowed: '{word}'")

    try:
        sql = generate_sql(req.message, req.dataset_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    allowed_tables = registry.get_table_names(req.dataset_ids)
    valid, reason = validate_sql(sql, allowed_tables=allowed_tables)
    if not valid:
        raise HTTPException(status_code=400, detail=reason)

    try:
        rows = run_query(sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"sql": sql, "results": rows}


@app.post("/enrich")
async def enrich(req: EnrichRequest):
    try:
        card = await enrich_place(
            name=req.name,
            name_en=req.name_en,
            place_type=req.place_type,
            wikipedia_tag=req.wikipedia,
            wikidata=req.wikidata,
        )
        return card
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/layers")
def get_layers():
    rows = run_query("SELECT * FROM layer_metadata")
    return rows
