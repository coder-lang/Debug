# routers/chat.py
# ===============
# POST   /chat              — send message, get HTML response
# GET    /chat/history      — get full conversation for a user_id
# GET    /chat/exists       — check if user_id has existing history
# DELETE /chat/history      — clear conversation for a user_id
# POST   /chat/grievance    — grievance assistant
# POST   /chat/stream       — streaming response (plain text, LLM output only)

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from models.chat import (
    ChatRequest, ChatResponse,
    ClearRequest, HistoryResponse,
    ChatMessage, UserExistsResponse, GrievanceRequest, GrievanceResponse
)
from services.chat_service import chat
from services.conversation_service import (
    clear_conversation,
    get_all_messages,
    user_exists,
)
import asyncio
import re

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def send_message(body: ChatRequest, request: Request):
    """
    Send a message and get a grounded HTML response.
    Same user_id → history continues. New user_id → fresh conversation.
    """
    answer, citations = chat(
        user_id=body.user_id,
        user_message=body.message,
        vectorstore=request.app.state.vectorstore,
    )
    return ChatResponse(answer=answer, sources=citations, is_html=True)


@router.get("/history", response_model=HistoryResponse)
def get_history(user_id: str):
    """GET /chat/history?user_id=abc-123"""
    messages = get_all_messages(user_id)
    return HistoryResponse(
        user_id=user_id,
        messages=[ChatMessage(**m) for m in messages],
    )


@router.get("/exists", response_model=UserExistsResponse)
def check_user(user_id: str):
    """GET /chat/exists?user_id=abc-123"""
    exists = user_exists(user_id)
    return UserExistsResponse(
        user_id=user_id,
        has_history=exists,
        message="Returning user — history loaded." if exists
                else "New user — fresh conversation starts.",
    )


@router.delete("/history", status_code=204)
def delete_history(body: ClearRequest):
    """Clear conversation history for a user_id."""
    clear_conversation(body.user_id)


@router.post("/grievance", response_model=GrievanceResponse)
async def create_grievance(body: GrievanceRequest, request: Request):
    """POST /chat/grievance — Grievance Assistant pipeline."""
    assistant = getattr(request.app.state, "grievance_assistant", None)
    if assistant is None:
        raise HTTPException(status_code=500, detail="Grievance assistant not initialized")

    result_text = assistant.process_user_input(body.message, body.user_id)

    status = "failure"
    grievance_id = None
    txt = (result_text or "").lower()

    if "grievance has been registered" in txt:
        status = "success"
        m = re.search(r"track it with\s+([A-Za-z0-9\-]+)", result_text, flags=re.IGNORECASE)
        if m:
            grievance_id = m.group(1)
    elif "already submited" in txt:
        status = "duplicate"

    return GrievanceResponse(
        status=status,
        message=result_text,
        grievance_id=grievance_id,
    )


@router.post("/stream")
async def stream_chat(body: ChatRequest, request: Request):
    """
    POST /chat/stream
    Streams the LLM answer as plain text chunks (chunked HTTP transfer).

    FIX vs old version:
    - Removed the "⏳ Working..." prefix chunk — it was being saved into
      session_state in Streamlit and corrupting the stored answer.
    - Now only the actual answer HTML is streamed, nothing else.
    - CHUNK_SIZE reduced to 128 for smoother progressive rendering.
    """
    user_id    = body.user_id
    user_msg   = body.message
    vectorstore = request.app.state.vectorstore

    async def generate():
        def run_chat_sync():
            answer, _ = chat(
                user_id=user_id,
                user_message=user_msg,
                vectorstore=vectorstore,
            )
            return answer  # HTML string

        # Run blocking chat() in thread so event loop stays free
        answer_html = await asyncio.to_thread(run_chat_sync)

        # Stream answer in small chunks for progressive rendering
        CHUNK_SIZE = 128
        for i in range(0, len(answer_html), CHUNK_SIZE):
            yield answer_html[i : i + CHUNK_SIZE]
            await asyncio.sleep(0)

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
