# GeoChat — Hybrid Geospatial RAG

A Retrieval-Augmented Generation system that answers questions over **your own
documents** and **your GeoJSON datasets**, decides for itself whether a question
is spatial, textual or both, renders results on an interactive map, and falls
back to **official websites** when local knowledge runs out — with hybrid
retrieval, automatic RAG evaluation, and layered safety guardrails throughout.

---

## What it does

Given a user query, GeoChat classifies intent and routes accordingly:

| Intent | Example | Pipeline |
| --- | --- | --- |
| **MAP** | "Show cafes on Tahrir Street" | NL → PostGIS SQL → GeoJSON features on the map |
| **KNOWLEDGE** | "How many students at Cairo University in 2025?" | Hybrid RAG (dense + BM25 + RRF) → grounded answer |
| **HYBRID** | "Show Cairo University on the map and tell me its enrollment" | Both pipelines, combined response |
| **UNKNOWN** | ambiguous | Ask a clarifying question |

The KNOWLEDGE / HYBRID path retrieves from uploaded documents; if the retrieved
context fails a relevance check, it falls back to the **feature's own official
website** (from the GeoJSON `website` field), then to official-domain search,
and only as a last resort to general web search. Every answer is generated
**only from retrieved context**, evaluated for faithfulness, and screened by
output guardrails before it is returned.

---

## Architecture

```
                                   POST /query
                                        │
                              ┌─────────▼──────────┐
                              │  input guardrails  │  prompt-injection / SQLi /
                              └─────────┬──────────┘  malicious-URL / length
                                        │
                              ┌─────────▼──────────┐
                              │  intent classifier │  LLM + heuristic fallback
                              └─────────┬──────────┘
                    ┌───────────────────┼────────────────────┐
                 MAP│                HYBRID                   │KNOWLEDGE
                    ▼                   ▼                     ▼
          ┌──────────────────┐  (both run)        ┌────────────────────────┐
          │  NL → SQL (llm)  │                    │   hybrid retrieval      │
          │  AST validator   │                    │  dense (Qdrant) + BM25  │
          │  PostGIS query   │                    │  fused with RRF         │
          └────────┬─────────┘                    └───────────┬────────────┘
                   │ GeoJSON features                          │
                   │                              ┌────────────▼───────────┐
                   │                              │ retrieval validation   │
                   │                              │ (context sufficient?)  │
                   │                              └──────┬──────────┬──────┘
                   │                            sufficient│         │insufficient
                   │                                      ▼         ▼
                   │                         grounded generation   web fallback
                   │                          (context only)       (official site first)
                   │                                      │         │
                   │                                      └────┬────┘
                   │                                           ▼
                   │                                  RAG evaluation
                   │                              (answer/context relevancy,
                   │                                    faithfulness)
                   │                                           ▼
                   │                                  output guardrails
                   │                             (groundedness, citations, leakage)
                   └───────────────────┬───────────────────────┘
                                       ▼
                              unified JSON response
```

### Module map (`agents/`)

| Module | Responsibility |
| --- | --- |
| `llm_client.py` | Provider-agnostic chat interface (Groq default; Claude/OpenAI/Gemini swappable via `LLM_PROVIDER`) |
| `document_loader.py` | Parse PDF / DOCX / TXT / Markdown / HTML / CSV → pages via **LangChain** community loaders |
| `chunker.py` | Heading/section chunking for catalogs (one entity per chunk, name prepended) + **LangChain** `RecursiveCharacterTextSplitter` fallback, with full provenance metadata |
| `embeddings.py` | `multilingual-e5-large`, normalized, query/passage prefixes |
| `vector_store.py` | Qdrant persistence + dense search + metadata filtering |
| `doc_ingest.py` | Orchestrates load → chunk → embed → store, records doc metadata |
| `hybrid_retriever.py` | Dense + BM25 fused with **Reciprocal Rank Fusion** (weighted) |
| `reranker.py` | Optional cross-encoder final-stage reranking |
| `retrieval_validator.py` | Context relevance / keyword overlap / thresholds |
| `intent_router.py` | MAP / KNOWLEDGE / HYBRID / UNKNOWN classification |
| `dataset_lookup.py` | Answers place questions from structured PostGIS feature columns (+ surfaces the feature's website) |
| `web_fallback.py` | Official-website-first web retrieval (Tier 0→C) |
| `evaluation.py` | Answer relevancy, context relevancy, faithfulness (RAGAS-style) |
| `guardrails.py` | Layered input / retrieval / output protections |
| `knowledge_pipeline.py` | Tiered answer: dataset → documents → official website → web, each evaluated + guardrailed |

The existing spatial pipeline (`llm.py`, `ingest.py`, `validator.py`,
`registry.py`, `db.py`, `rag.py`) is unchanged and reused.

---

## Setup

### 1. Backing services

```bash
# Qdrant (vectors)
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# PostgreSQL + PostGIS (spatial + registries)
docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=geochat postgis/postgis:16-3.4
```

### 2. Python environment

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in DB_URL, API_KEY, (optional) TAVILY_API_KEY
```

The first document ingestion downloads `intfloat/multilingual-e5-large`
(~2.2 GB). Set `EMBEDDING_MODEL` to a smaller model for quick local iteration.

### 3. Run

```bash
uvicorn main:app --reload
```

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/documents/upload` | Ingest a PDF/DOCX/TXT/MD/HTML/CSV into the RAG index |
| `GET` | `/documents` | List ingested documents |
| `DELETE` | `/documents/{id}` | Remove a document from index + metadata |
| `POST` | `/datasets/upload` | Ingest a GeoJSON/CSV into PostGIS |
| `POST` | `/datasets/describe` | Attach per-column descriptions |
| `GET` | `/datasets` | List datasets |
| `DELETE` | `/datasets/{id}` | Drop a dataset |
| `POST` | `/query` | **Intent-routed** map / knowledge / hybrid query |
| `POST` | `/chat` | Legacy spatial-only NL→SQL |
| `POST` | `/enrich` | Place enrichment card (Wikipedia + Places) |
| `GET` | `/health` | Liveness + indexed-document count |

### `/query` example

```jsonc
// request
{
  "message": "Show Cairo University on the map and how many students enrolled in 2025",
  "dataset_ids": ["<university-dataset-id>"],
  "website": "https://cu.edu.eg"     // optional: official site for fallback
}

// response (abridged)
{
  "intent": { "intent": "HYBRID", "entity_focus": "Cairo University", "needs_map": true, "needs_rag": true },
  "map": { "sql": "SELECT ... ST_AsGeoJSON(wkb_geometry) ...", "results": [ /* features */ ] },
  "knowledge": {
    "answer": "...",
    "found": true,
    "sources": ["Cairo University Facts.pdf"],
    "source_tier": "local_index",
    "citations": [ { "title": "...", "source": "...", "page": 3, "chunk_id": "..." } ],
    "retrieval":  { "context_score": 0.71, "is_sufficient": true, "num_chunks": 5 },
    "evaluation": { "answer_relevancy": 0.9, "context_relevancy": 0.85, "faithfulness": 1.0 },
    "guardrail":  { "valid": true, "reasons": [] }
  }
}
```

---

## Configuration

All knobs are environment variables read once by `config.py`. Highlights
(see `.env.example` for the full list and defaults):

`LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`,
`CHUNK_STRATEGY` (auto/heading/recursive),
`RETRIEVAL_TOP_K`, `DENSE_WEIGHT`, `BM25_WEIGHT`, `RRF_K`, `RERANKER_ENABLED`,
`MIN_CONTEXT_SCORE`, `WEB_FALLBACK_ENABLED`, `GUARDRAILS_ENABLED`,
`EVALUATION_ENABLED`.

---

## Logs — tracing a query end to end

Every stage logs through a single namespaced `geochat.*` logger tree
(`agents/logging_config.py`). At `LOG_LEVEL=INFO` you get a clean high-level
trace of each step; `LOG_LEVEL=DEBUG` adds detail (per-retriever hit counts,
scores, generated SQL, chosen web tier, embedding timings). Timed stages are
wrapped so they print `▶ start` and `✓ done — <ms>` (or `✗ failed — <ms>`).

A single knowledge query looks like:

```
15:14:48 INFO  geochat.api                  POST /query — 'how many students at Cairo University in 2025'
15:14:48 INFO  geochat.intent_router        classifying intent for: 'how many students…'
15:14:48 INFO  geochat.intent_router        intent=KNOWLEDGE (LLM, conf=0.90, entity=Cairo University)
15:14:48 INFO  geochat.knowledge_pipeline   ▶ local retrieval
15:14:48 INFO  geochat.hybrid_retriever     hybrid retrieval for 'how many students…' (pool=20, top_k=5)
15:14:48 INFO  geochat.hybrid_retriever     built BM25 keyword index over 128 chunk(s)
15:14:48 INFO  geochat.hybrid_retriever     hybrid retrieval returned 5 result(s) (RRF fused)
15:14:48 INFO  geochat.retrieval_validator  retrieval validation: context_score=0.71 … -> SUFFICIENT
15:14:48 INFO  geochat.knowledge_pipeline   ✓ local retrieval — 42ms
15:14:48 INFO  geochat.knowledge_pipeline   ▶ grounded generation from local context
15:14:49 INFO  geochat.evaluation           evaluation: faithfulness=1.0 answer_rel=0.9 context_rel=0.85
15:14:49 INFO  geochat.knowledge_pipeline   knowledge answer ready (tier=local_index, 1 source(s))
15:14:49 INFO  geochat.api                  /query done — intent=KNOWLEDGE, map=False, knowledge=True
```

Controls (env vars, see `.env.example`):

| Var | Default | Effect |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | `DEBUG` for full per-stage detail, `WARNING` for quiet |
| `LOG_TO_FILE` | `false` | also write to a rotating file |
| `LOG_FILE` | `geochat.log` | path of that file (5 MB × 3 rotations) |
| `LOG_JSON` | `false` | emit one JSON object per line (for log shippers) |

---

## Safety guardrails

- **Input** — blocks prompt injection, jailbreaks, SQL-injection probes,
  malicious URLs, over-length queries (`guardrails.check_input`).
- **Ingestion / retrieval** — redacts secrets/keys/passwords/env-vars and
  neutralizes instructions hidden inside uploaded documents so they can't hijack
  the model (`guardrails.scan_document_text`, `redact_secrets`).
- **Generation** — the model is instructed to answer *only* from retrieved
  context and to admit when it can't.
- **Output** — validates groundedness, citation presence, and system-prompt /
  secret leakage; an answer that fails is replaced with a safe refusal
  (`guardrails.validate_output`).
- **SQL** — every generated query passes an AST validator and a dataset
  allow-list before it touches the database (`validator.py`).

---

## Tests

```bash
pytest
```

Covers RRF fusion, the recursive chunker + metadata, all guardrail layers,
retrieval validation, intent classification, the SQL validator, and the
document loaders. Tests are designed to run **without** the heavy ML
dependencies (torch / Qdrant / provider SDKs are imported lazily), so the core
logic is verifiable in any environment.
