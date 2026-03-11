"""
services/chat_service.py
=========================
Core RAG pipeline:
  - Persistent conversation history (JSON files)
  - Year-aware retrieval
  - HTML output
  - Strict grounding — no outside knowledge
"""
import re
from typing import List, Tuple

from openai import AzureOpenAI
from langchain_community.vectorstores import Chroma

from core.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
    CHAT_DEPLOYMENT,
    VECTOR_SEARCH_TOP_K,
)
from services.vector_service import search_vectorstore
from services.conversation_service import (
    get_recent_history,
    save_message,
    summarize_if_needed,
)

_openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

SYSTEM_PROMPT = """
You are Uttar Pradesh Info Assistant. You answer questions using ONLY the text
provided in the Context section below. Nothing else.

=== CRITICAL: NO OUTSIDE KNOWLEDGE ===
Every sentence in your answer MUST come directly from the Context text.
Do NOT add background information, general knowledge, or explanations
about schemes, policies, or programs that are not in the Context.
If a fact is not written in the Context — do NOT include it.

=== LANGUAGE HANDLING ===
- Context may be in English, Hindi, or mixed — read ALL chunks.
- If a chunk is in Hindi, translate it internally and extract the answer.
- Never say "not found" just because a chunk is in Hindi script.
- Answer in the SAME language the user asked in.

=== HOW TO ANSWER ===
- Read every single chunk in Context before responding.
- Find chunks that directly answer the question.
- Extract ONLY what is written in those chunks.
- Do NOT add anything not present in the chunks.
- Only say "not found" if the topic is completely absent from every chunk.

=== YEAR HANDLING ===
CASE 1 — User mentions a specific year:
  → Use data from that year in Context.
  → If that year is not found, say:
    "Data for [year] is not available. Closest available data is from [year]:"
    Then show that data. Never just say not found.

CASE 2 — User does NOT mention a year:
  → Search ALL chunks across ALL PDFs and ALL years.
  → Answer from whichever chunk has the most relevant data.
  → Always mention which year and document the answer is from.
  → If multiple years have data, show most recent first.

CASE 3 — Topic genuinely absent from ALL chunks:
  → Only then say not found.

=== OUTPUT FORMAT ===
Pure HTML only. No Markdown. No ``` blocks.
- <div class="answer"> ... </div>       wrap entire response
- <p> ... </p>                          paragraphs
- <h3> ... </h3>                        headings
- <ul><li>...</li></ul>                 bullet lists
- <ol><li>...</li></ol>                 numbered lists
- <table><thead><tbody><tr><th><td>     tables
- <b> ... </b>                          important values
- <p class="source"> ... </p>           citations
- <p class="not-found"> ... </p>        only when truly not found

=== CITATION ===
End EVERY answer with:
<p class="source">Source: Document: [name], Page: [n], Year: [year]</p>

=== FINAL CHECKLIST ===
- Is every fact in my answer present in the Context? If NO → remove it.
- Am I adding general knowledge? If YES → remove it.
- Am I adding "X is not mentioned" sentences from my training? If YES → remove them.
""".strip()


def extract_years_from_query(query: str) -> List[str]:
    years = set()
    for match in re.findall(r"(20\d{2})[-–]((\d{2})|20\d{2})", query):
        years.add(match[0])
        suffix = match[1]
        years.add(match[0][:2] + suffix if len(suffix) == 2 else suffix)
    for y in re.findall(r"\b(20\d{2})\b", query):
        years.add(y)
    return sorted(years)


def filter_chunks_by_year(chunks: List[dict], years: List[str]) -> List[dict]:
    """
    3-pass year filter:
    Pass 1 — exact year in text body or doc_name
    Pass 2 — previous year doc_name (annual reports cover next year data)
    Pass 3 — return all if nothing found
    """
    if not years:
        return chunks

    def matches(chunk, year, prev=False):
        text     = chunk.get("text", "")
        doc_name = str(chunk.get("doc_name", ""))
        if prev:
            prev_year = str(int(year) - 1)
            return prev_year in doc_name
        return year in text or year in doc_name

    # Pass 1 — exact match
    filtered = [c for c in chunks if any(matches(c, y) for y in years)]
    if filtered:
        print(f"[chat_service] Year filter exact: {len(chunks)}→{len(filtered)} for {years}")
        return filtered

    # Pass 2 — previous year PDF
    adjacent = [c for c in chunks if any(matches(c, y, prev=True) for y in years)]
    if adjacent:
        print(f"[chat_service] Year filter adjacent: {len(adjacent)} chunks from year-1 PDFs")
        return adjacent

    # Pass 3 — return all
    print(f"[chat_service] Year {years} not found — returning all chunks")
    return chunks


def _build_context_block(chunks: List[dict]) -> Tuple[str, List[str]]:
    if not chunks:
        return "", []
    parts, citations = [], []
    for i, chunk in enumerate(chunks, 1):
        doc         = chunk.get("doc_name", "unknown")
        page        = chunk.get("page_no",  "?")
        text        = chunk.get("text",     "")
        chunk_years = sorted(set(re.findall(r"\b20\d{2}\b", text)))
        year_label  = f", Years: {', '.join(chunk_years)}" if chunk_years else ""
        citation    = f"Document: {doc}, Page: {page}"
        citations.append(citation)
        parts.append(f"[Chunk {i} | {citation}{year_label}]\n{text}\n")
    return "\n---\n".join(parts), citations


def _translate_to_gujarati(text: str) -> str:
    """
    Translate English query to Gujarati for vector search.
    If already Hindi, returns unchanged.
    Falls back to original text if translation fails.
    """
    gujarati_chars = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
    if gujarati_chars > 0:
        return text  # Already Gujarati

    try:
        response = _openai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the English text to Hindi. "
                        "Return ONLY the Hindi translation, nothing else."
                    )
                },
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=300,
        )
        translated = response.choices[0].message.content.strip()
        print(f"[chat_service] Query translated: {text[:50]} → {translated[:50]}")
        return translated
    except Exception as e:
        print(f"[chat_service] Translation failed, using original: {e}")
        return text


def chat(
    user_id: str,
    user_message: str,
    vectorstore: Chroma,
) -> Tuple[str, List[str]]:

    # Step 1 — Extract years
    query_years = extract_years_from_query(user_message)
    if query_years:
        print(f"[chat_service] Years in query: {query_years}")

    # Step 2 — Translate query to Gujarati for vector search
    # (chunks are in Gujarati — query must match same language)
    search_query = _translate_to_gujarati(user_message)

    # Step 2b — Vector search — fetch more to allow filtering
    fetch_k = 40
    chunks  = search_vectorstore(search_query, vectorstore, k=fetch_k)

    # Step 2c — Limit chunks per document (max 3 per PDF)
    # Prevents any single large PDF (like WP-138) from dominating results
    doc_counts = {}
    balanced_chunks = []
    MAX_PER_DOC = 3
    for c in chunks:
        doc = c.get("doc_name", "unknown")
        if doc_counts.get(doc, 0) < MAX_PER_DOC:
            balanced_chunks.append(c)
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
    chunks = balanced_chunks
    print(f"[chat_service] After balancing: {len(chunks)} chunks from {len(doc_counts)} docs: {list(doc_counts.keys())}")

    # Step 3 — Confidence gate
    if not chunks:
        reply = (
            "<div class='answer'>"
            "<p class='not-found'>I could not find relevant information "
            "in the available documents.</p></div>"
        )
        save_message(user_id, "user",      user_message)
        save_message(user_id, "assistant", reply)
        return reply, []

    # Step 4 — Year filter only if year specified
    if query_years:
        chunks = filter_chunks_by_year(chunks, query_years)

    # Step 5 — Build context
    context_block, citations = _build_context_block(chunks)

    # Step 6 — Augment query
    if query_years:
        augmented_query = (
            f"{user_message}\n\n"
            f"[User requested year(s): {', '.join(query_years)}. "
            f"Prioritize data from these years. If not available, provide "
            f"data from the closest available year and mention it clearly.]"
        )
    else:
        augmented_query = (
            f"{user_message}\n\n"
            f"[No year specified. Search ALL chunks across ALL years and PDFs. "
            f"Answer from whichever chunk has the most relevant data. "
            f"Always mention which year and document the answer is from.]"
        )

    # Step 7 — Load history
    history = get_recent_history(user_id)

    # Step 8 — Assemble messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Context (use ONLY this):\n\n{context_block}"},
        *history,
        {"role": "user", "content": augmented_query},
    ]

    # Step 9 — Call Azure OpenAI
    response = _openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0,
        top_p=0.9,
        max_tokens=1500,
    )
    answer = response.choices[0].message.content or ""

    # Step 10 — Save history
    save_message(user_id, "user",      user_message)
    save_message(user_id, "assistant", answer)
    summarize_if_needed(user_id)

    # Step 11 — Deduplicate citations, keep top 3
    seen = []
    for c in citations:
        if c not in seen:
            seen.append(c)
        if len(seen) == 3:
            break

    return answer, seen
