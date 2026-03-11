"""
main.py
=======
FastAPI entry point.
No auth/JWT. Frontend passes user_id directly in every request.
Conversation history stored in JSON files (conversation_store/).

Run order:
  1. python scripts/ingest_pdfs.py   (one-time)
  2. python main.py
"""
import sys
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from grievances.grv_assistant import Grievance_Assistant

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.chat  import router as chat_router
from routers.debug import router as debug_router          # ← ADDED
from services.vector_service import build_or_load_vectorstore

REQUIRED_ENV_VARS = [
    "DOC_INTEL_ENDPOINT",
    "DOC_INTEL_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_API_VERSION",
    "CHAT_DEPLOYMENT",
    "EMBED_DEPLOYMENT",
]


def validate_env():
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        print(f"\nERROR: Missing .env variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in all values.\n")
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_env()
    print("[startup] Loading vectorstore...")
    app.state.vectorstore = build_or_load_vectorstore()
    print("[startup] Initializing grievance assistant...")
    app.state.grievance_assistant = Grievance_Assistant()
    print("[startup] Server ready.")
    yield
    print("[shutdown] Stopping.")


app = FastAPI(
    title="Uttar Pradesh Info Chatbot API",
    description=(
        "RAG chatbot over government PDFs. "
        "Frontend sends user_id to maintain conversation history. "
        "Responses are HTML formatted."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(debug_router)                          # ← ADDED


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="10.40.108.197", port=8508, reload=True)
