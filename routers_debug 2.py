# routers/debug.py
# ================
# PUT THIS FILE in your  routers/  folder (same folder as chat.py)
# Then in main.py add:
#   from routers.debug import router as debug_router
#   app.include_router(debug_router)
#
# Exposes:
#   POST /debug/chunks  — returns the exact chunks retrieved from vectorstore
#                         that will be passed as context to the LLM

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/debug", tags=["Debug"])


class DebugChunksRequest(BaseModel):
    query: str
    n_results: Optional[int] = 6


class ChunkDetail(BaseModel):
    chunk_number: int
    text: str                    # ← exact text the LLM will see as context
    document_name: str
    page_number: Optional[int] = None
    relevance_score: Optional[float] = None
    char_count: int
    token_estimate: int          # rough: char_count // 4


class DebugChunksResponse(BaseModel):
    query: str
    total_chunks_retrieved: int
    chunks: list[ChunkDetail]
    combined_context_chars: int
    combined_context_token_estimate: int


@router.post("/chunks", response_model=DebugChunksResponse)
async def get_raw_chunks(body: DebugChunksRequest, request: Request):
    """
    Returns the EXACT document chunks retrieved from the vectorstore
    for a given query — before any LLM call is made.

    Use this in Streamlit debug mode to see what context the LLM receives.
    """
    vectorstore = request.app.state.vectorstore

    # similarity_search_with_score returns list of (Document, score)
    # This works with FAISS, Chroma, Azure Search — any LangChain vectorstore
    results = vectorstore.similarity_search_with_score(
        body.query,
        k=body.n_results
    )

    chunk_details = []
    for i, (doc, score) in enumerate(results, 1):
        text = doc.page_content
        meta = doc.metadata or {}

        # Normalise score to 0–1 range
        # FAISS returns L2 distance (lower = better), so convert it
        # Most other stores return cosine similarity (higher = better)
        # We detect by checking if score > 1 (L2 can be > 1, cosine can't)
        if score > 1:
            relevance = round(1 / (1 + score), 4)   # L2 → similarity
        else:
            relevance = round(float(score), 4)       # already cosine similarity

        chunk_details.append(ChunkDetail(
            chunk_number=i,
            text=text,
            document_name=meta.get("source") or meta.get("file_name") or meta.get("document_name") or "Unknown",
            page_number=meta.get("page") or meta.get("page_number"),
            relevance_score=relevance,
            char_count=len(text),
            token_estimate=len(text) // 4
        ))

    combined = "\n\n---\n\n".join(c.text for c in chunk_details)

    return DebugChunksResponse(
        query=body.query,
        total_chunks_retrieved=len(chunk_details),
        chunks=chunk_details,
        combined_context_chars=len(combined),
        combined_context_token_estimate=len(combined) // 4
    )
