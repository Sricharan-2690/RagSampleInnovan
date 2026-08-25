"""
STEP 5: RAG INFERENCE (RETRIEVAL -> GENERATION)
------------------------------------------------
- Reuses the existing hybrid retrieval pipeline from query.py:
  Groq metadata filter extraction -> Dense+Sparse -> RRF fusion
  -> Cross-encoder rerank -> top 5 chunks.
- Takes those top 5 chunks, builds a grounded context block (with page
  numbers), and sends it + the user's question to a Groq LLM to generate
  a final natural-language answer.
- The LLM is explicitly instructed to answer ONLY from the provided
  context and say so if the answer isn't there, so it doesn't fall back
  on its own general knowledge and quietly stop being "RAG".

Run order: chunking.py -> store_qdrant.py -> generate.py
(generate.py replaces running query.py directly for end-user Q&A;
query.py's functions are imported, not duplicated.)
"""

from query import (
    search_hybrid,
    rerank_with_cross_encoder,
    get_user_selected_collection,
    groq_client,
    RRF_CANDIDATE_POOL,
    ScopeTooBroadError,
)
import re
from dotenv import load_dotenv
from langsmith import traceable, trace

load_dotenv()

# -----------------------------
# 0. Casual message detection
# -----------------------------
CASUAL_PATTERN = re.compile(
    r"^(hi+|hello+|hey+|yo|sup|good\s*(morning|afternoon|evening|night)|"
    r"how are you( doing)?|what'?s up|thanks?( you)?|thank you|bye+|"
    r"who are you|what can you do|help me|ok(ay)?|cool|nice)[\s!.,?]*$",
    re.IGNORECASE,
)


@traceable(run_type="chain", name="is_casual_message")
def is_casual_message(query: str) -> bool:
    """Cheap, local check — no API call — for greeting/small-talk style messages."""
    return bool(CASUAL_PATTERN.match(query.strip()))

# -----------------------------
# 1. System prompt for grounded generation
# -----------------------------
SYSTEM_PROMPT = """You are a helpful assistant for answering questions about a specific document.

There are two kinds of messages you'll receive:

1. CASUAL CONVERSATION (greetings, small talk, thanks, "who are you",
   "how can you help me", etc.)
   - Respond naturally and briefly, like a normal assistant would.
   - Do NOT apply the context-only rule below to these messages.
   - You can mention that you're here to answer questions about the
     document if it feels natural, but keep it short.

2. QUESTIONS ABOUT THE DOCUMENT (anything asking for facts, figures,
   explanations, or details that should come from the document)
   - Answer strictly using ONLY the context provided below.
   - Do NOT use outside knowledge, even if you happen to know the answer.
   - If the context does not contain the answer, say clearly:
     "I couldn't find this in the document."
   - When you use a fact from the context, mention which page it came
     from, e.g. "(page 6)".
   - Match the LENGTH and DEPTH of your answer to what the user actually
     asked for:
       * If they ask to "explain", "describe in detail", "elaborate", or
         don't specify a length at all — cover ALL relevant points found
         in the context thoroughly, not just a one-line summary.
       * Only give a short, compressed answer if the user explicitly asks
         for "briefly", "in short", "in one line", "summarize", or similar.
       * Never compress a thorough question into a short answer just to
         save space — use as much of the context as is relevant.

Decide which category the user's message falls into before answering.
Never invent facts about the document that are not present in the context.
"""

GENERATION_MODEL = "openai/gpt-oss-120b"


# -----------------------------
# 2. Build the context block from retrieved chunks
# -----------------------------
@traceable(run_type="chain", name="build_context")
def build_context(scored_points) -> str:
    """Formats the top reranked chunks into a single context string for the LLM."""
    blocks = []
    for i, (point, _rerank_score) in enumerate(scored_points, start=1):
        pages = point.payload.get("pages", [])
        text = point.payload["text"]
        blocks.append(f"[Chunk {i} | Pages: {pages}]\n{text}")

    return "\n\n---\n\n".join(blocks)


# -----------------------------
# 3. Generate the final answer with Groq
# -----------------------------
@traceable(run_type="llm", name="generate_answer")
def generate_answer(query: str, context: str) -> str:
    """Sends the retrieved context + user question to Groq and returns the answer."""
    response = groq_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    choice = response.choices[0]
    if choice.finish_reason == "length":
        print("[WARNING] Response was cut off (finish_reason='length'). "
              "Consider raising max_tokens further if this keeps happening.")

    return choice.message.content


# -----------------------------
# 4. Full pipeline: retrieve -> rerank -> generate
# -----------------------------
@traceable(run_type="chain", name="rag_pipeline")
def answer_question(query: str, collection_name: str, top_k: int = 5) -> dict:
    """Runs the full RAG pipeline for one question."""
    if is_casual_message(query):
        answer = generate_answer(query, context="")
        return {"answer": answer, "sources": []}

    try:
        candidates, _dense_only, _sparse_only, is_filtered = search_hybrid(
            query=query,
            collection_name=collection_name,
            candidate_pool=RRF_CANDIDATE_POOL,
        )
    except ScopeTooBroadError as e:
        return {"answer": str(e), "sources": []}

    final_results = rerank_with_cross_encoder(
        query=query,
        candidates=candidates,
        top_k=None if is_filtered else top_k,
    )

    if not final_results:
        return {
            "answer": "I couldn't find any relevant content in the document.",
            "sources": [],
        }

    context = build_context(final_results)
    answer = generate_answer(query, context)

    sources = [
        (i, point.payload.get("pages", []))
        for i, (point, _score) in enumerate(final_results, start=1)
    ]

    return {"answer": answer, "sources": sources}


def print_answer(result: dict):
    """Pretty-print the final generated answer and its sources."""
    print("\n" + "=" * 90)
    print("--- ANSWER ---")
    print("=" * 90)
    print(result["answer"])

    if result["sources"]:
        print("\n" + "-" * 90)
        print("Sources used:")
        for chunk_idx, pages in result["sources"]:
            print(f"  Chunk {chunk_idx} -> Pages {pages}")


EXIT_WORDS = {"exit", "bye", "done", "quit"}


if __name__ == "__main__":
    target_collection = get_user_selected_collection()
    print(f"\nActive Target Collection: '{target_collection}'")
    print("Ask a question about the document. Type 'exit', 'bye', or 'done' to stop.\n")

    while True:
        user_query = input("Enter your question: ").strip()

        if not user_query:
            continue

        if user_query.lower() in EXIT_WORDS:
            print("Goodbye!")
            break

        result = answer_question(query=user_query, collection_name=target_collection, top_k=5)
        print_answer(result)
