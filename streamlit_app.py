import streamlit as st
import requests
import os
import sys

# -------------------- SESSION STATE --------------------
if "info_messages" not in st.session_state:
    st.session_state.info_messages = []
if "grievance_messages" not in st.session_state:
    st.session_state.grievance_messages = []

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Chat & Grievance Portal", layout="wide")

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://10.40.108.197:8508")
DEFAULT_USER_ID  = os.getenv("DEFAULT_USER_ID", "demo_user")

CHAT_URL         = f"{BACKEND_BASE_URL}/chat"           # non-streaming fallback
CHAT_STREAM_URL  = f"{BACKEND_BASE_URL}/chat/stream"    # LLM output streaming
GRIEVANCE_URL    = f"{BACKEND_BASE_URL}/chat/grievance"
DEBUG_CHUNKS_URL = f"{BACKEND_BASE_URL}/debug/chunks"   # RAW chunks passed to LLM


# -------------------- BACKEND HELPERS --------------------

def stream_llm_output(user_message: str):
    """
    Streams the LLM's OUTPUT tokens from /chat/stream.
    These are the ANSWER tokens — NOT the input RAG chunks.
    iter_content(chunk_size=64) gives small pieces for smooth typing effect.
    """
    payload = {"user_id": DEFAULT_USER_ID, "message": user_message}
    try:
        with requests.post(CHAT_STREAM_URL, json=payload, timeout=180, stream=True) as r:
            r.raise_for_status()
            for piece in r.iter_content(chunk_size=64):
                if piece:
                    yield piece.decode("utf-8", errors="ignore")
    except Exception as e:
        yield f"\n\n⚠️ Streaming error: {e}"


def fetch_raw_rag_chunks(query: str, n_results: int = 6) -> dict:
    """
    Calls POST /debug/chunks — returns the EXACT document chunks
    retrieved from the vectorstore that the LLM will receive as context.
    This is completely separate from the LLM's answer stream.
    """
    try:
        r = requests.post(
            DEBUG_CHUNKS_URL,
            json={"query": query, "n_results": n_results},
            timeout=30
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def call_backend_grievance(user_message: str) -> dict:
    try:
        r = requests.post(
            GRIEVANCE_URL,
            json={"user_id": DEFAULT_USER_ID, "message": user_message},
            timeout=20
        )
        return r.json()
    except Exception as e:
        return {"message": f"Connection error: {e}"}


# -------------------- UI --------------------
st.title("Uttar Pradesh AI Chatbot")

tab1, tab2 = st.tabs(["🤖 Information Bot", "📢 Raise Grievance"])


# ==================== INFORMATION BOT ====================
with tab1:
    st.subheader("Information Bot")

    # Debug toggle in top-right
    col1, col2 = st.columns([3, 1])
    with col2:
        debug_mode = st.checkbox(
            "🔍 Show RAG chunks",
            value=False,
            help=(
                "Shows the EXACT document chunks retrieved from the vectorstore "
                "that are passed as context to the LLM — before the LLM answers."
            )
        )
        if debug_mode:
            n_chunks = st.slider("Max chunks", min_value=1, max_value=10, value=6)

    # Render previous messages
    for msg in st.session_state.info_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    info_input = st.chat_input("Ask something...")

    if info_input:
        # Show + save user message
        st.session_state.info_messages.append({"role": "user", "content": info_input})
        with st.chat_message("user"):
            st.markdown(info_input)

        # ── DEBUG: Show RAG chunks BEFORE the LLM answers ──────────────────
        if debug_mode:
            with st.expander("📄 RAW CHUNKS PASSED TO LLM", expanded=True):
                with st.spinner("Fetching chunks from vectorstore..."):
                    debug_data = fetch_raw_rag_chunks(info_input, n_results=n_chunks)

                if "error" in debug_data:
                    st.error(f"Debug endpoint error: {debug_data['error']}")
                else:
                    st.caption(
                        f"**Query:** `{debug_data.get('query', '')}` &nbsp;|&nbsp; "
                        f"**Chunks:** {debug_data.get('total_chunks_retrieved', 0)} &nbsp;|&nbsp; "
                        f"**Total chars:** {debug_data.get('combined_context_chars', 0)} &nbsp;|&nbsp; "
                        f"**~Tokens:** {debug_data.get('combined_context_token_estimate', 0)}"
                    )
                    for chunk in debug_data.get("chunks", []):
                        lang_flag  = "🇮🇳" if chunk.get("language") == "hi" else "🇬🇧"
                        score      = chunk.get("relevance_score", 0)
                        score_color = (
                            "green"  if score > 0.7 else
                            "orange" if score > 0.4 else
                            "red"
                        )
                        st.markdown(
                            f"**Chunk #{chunk['chunk_number']}** {lang_flag} &nbsp;|&nbsp; "
                            f"📄 `{chunk['document_name']}` &nbsp;"
                            f"p.{chunk.get('page_number', '?')} &nbsp;|&nbsp; "
                            f"Score: :{score_color}[**{score}**] &nbsp;|&nbsp; "
                            f"{chunk['char_count']} chars (~{chunk['token_estimate']} tokens)",
                            unsafe_allow_html=True
                        )
                        # ← THIS IS THE EXACT TEXT GOING INTO THE LLM PROMPT
                        st.code(chunk["text"], language=None)
                        st.divider()

                    st.caption("⬆️ Above = exact context the LLM receives. Below = LLM's answer.")

        # ── LLM ANSWER (streaming) ──────────────────────────────────────────
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            for token in stream_llm_output(info_input):
                full_answer += token
                placeholder.markdown(full_answer, unsafe_allow_html=True)

        # Save final answer to history (clean — no "⏳ Working..." prefix)
        st.session_state.info_messages.append({"role": "assistant", "content": full_answer})


# ==================== GRIEVANCE BOT ====================
with tab2:
    st.subheader("Raise a Grievance")

    for msg in st.session_state.grievance_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    grievance_input = st.chat_input("Describe your grievance...")

    if grievance_input:
        st.session_state.grievance_messages.append({
            "role": "user",
            "content": grievance_input
        })

        response   = call_backend_grievance(grievance_input)
        status     = response.get("status")
        ticket     = response.get("grievance_id")
        message    = response.get("message", "")

        if status == "success":
            reply = f"""
            <div style='padding:10px;border-left:4px solid green;'>
                <b>✅ Your grievance has been registered successfully!</b><br>
                Ticket Number: <b>{ticket}</b>
            </div>
            """
        else:
            reply = message

        st.session_state.grievance_messages.append({"role": "assistant", "content": reply})
        st.rerun()
