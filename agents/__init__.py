"""
agents — the RAG / retrieval / routing layer for GeoChat.

This package sits alongside the existing spatial pipeline (``llm.py``,
``ingest.py``, ``validator.py``, ``registry.py``) and adds the document-centric
side of the system:

    document_loader   parse PDF/DOCX/TXT/MD/HTML/CSV into raw text
    chunker           recursive character splitting with structure-aware metadata
    embeddings        multilingual sentence-transformer embeddings (normalized)
    vector_store      Qdrant persistence + dense similarity search
    hybrid_retriever  dense + BM25 fused with Reciprocal Rank Fusion
    reranker          optional cross-encoder reranking
    retrieval_validator   is the retrieved context actually relevant?
    intent_router     MAP / KNOWLEDGE / HYBRID / UNKNOWN classification + routing
    evaluation        answer relevancy, context relevancy, faithfulness
    guardrails        layered input / retrieval / output protections
    web_fallback      official-website-first web retrieval
    llm_client        provider-agnostic chat interface (Groq default)

Modules are written so that pure-logic components (RRF, chunking, guardrail
regexes, intent parsing) import cleanly without heavyweight ML dependencies;
those are imported lazily inside the functions that need them.
"""

__all__ = [
    "llm_client",
    "document_loader",
    "chunker",
    "embeddings",
    "vector_store",
    "hybrid_retriever",
    "reranker",
    "retrieval_validator",
    "intent_router",
    "evaluation",
    "guardrails",
    "web_fallback",
]
