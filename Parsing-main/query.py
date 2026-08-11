"""
STEP 4: HYBRID QUERY WITH EXPLICIT COLLECTION SELECTION
-------------------------------------------------------
- Accepts explicit 'collection_name' from caller (UI/API/CLI).
- Uses Groq API (Llama 3.3 70B) to parse metadata pre-filters (pages, labels).
- Executes a single fused Dense (bge-small) + Sparse (BM25) hybrid search
  over the target collection, combined server-side via Qdrant's RRF fusion.

FIXED:
1. Dense and sparse results are no longer returned/printed as two separate,
   unrelated lists. They're now combined with Qdrant's native RRF
   (Reciprocal Rank Fusion) via `prefetch` + `FusionQuery`, so you get one
   ranked hybrid result list — which is what "hybrid search" is supposed
   to mean. (Real BM25 scoring on the sparse side only works correctly
   because store_qdrant.py now sets `modifier=Modifier.IDF` on the
   collection — see that file.)
2. bge-small-en-v1.5 is an asymmetric embedding model: it expects queries
   to be prefixed with an instruction string for best retrieval quality.
   Document embeddings (in store_qdrant.py) are left as-is; only the query
   text gets the prefix.
3. Added a cross-encoder reranking stage. RRF fusion now pulls a WIDE
   candidate pool (RRF_CANDIDATE_POOL = 25) instead of directly returning
   only 5 results. A cross-encoder then reads (query, chunk_text) together
   for all 25 candidates and re-scores them for true relevance, and only
   THEN do we cut down to the final top_k (5). Feeding a cross-encoder
   only 5 pre-cut candidates would defeat the purpose — it can only
   re-sort what it's given, so RRF's rough ranking would have already
   thrown away anything ranked 6th or worse before the more accurate
   model got a chance to look at it.
"""

import sys
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)


load_dotenv()

# -----------------------------
# 1. Initialize Models & Clients
# -----------------------------
print("Loading retrieval models...")
dense_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# bge-small-en-v1.5 is asymmetric: queries need this instruction prefix,
# document/passage embeddings do NOT (store_qdrant.py embeds docs as-is).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# How many candidates RRF fusion should surface BEFORE reranking.
# This must be wider than the final top_k so the cross-encoder has real
# options to re-sort, not just 5 pre-decided results to shuffle.
RRF_CANDIDATE_POOL = 25

# Safety cap for the "fetch ALL chunks matching an explicit filter" path
# (used when the user names specific pages/a filename — see fetch_all_filtered_chunks).
# This is NOT a normal working limit — a real page/page-range question will
# land nowhere near it. It only exists to fail predictably (ask the user to
# narrow scope) instead of silently sending a huge, expensive request if a
# filter ever matches an unexpectedly large number of chunks.
FILTERED_FETCH_CAP = 100

client = QdrantClient(host="localhost", port=6333)
groq_client = Groq()


class ScopeTooBroadError(Exception):
    """Raised when an explicit filter (page_nos/filename) matches more than
    FILTERED_FETCH_CAP chunks. Callers should catch this and ask the user
    to narrow their question, rather than silently truncating or sending
    an oversized request to the LLM."""
    pass

# -----------------------------
# 2. Define Metadata Schema
# -----------------------------
class ExtractedMetadata(BaseModel):
    page_nos: Optional[List[int]] = Field(
        default=None,
        description="Page number(s) explicitly mentioned in query, as a list (e.g., [4], [5, 6])"
    )
    label: Optional[str] = Field(
        default=None, 
        description="Content label if specifically requested: 'text', 'section_header', 'table', 'list_item'"
    )
    filename: Optional[str] = Field(
        default=None, 
        description="Filename if explicitly mentioned in query"
    )

# -----------------------------
# 3. Dynamic Metadata Extractor
# -----------------------------
def extract_metadata_filters(query: str) -> ExtractedMetadata:
    """Uses Groq Llama 3 to parse intent and extract metadata criteria as JSON."""
    prompt = f"""You are a query parser for a database. Analyze the query and extract metadata filters.
Available content labels: 'text', 'section_header', 'table', 'list_item', 'page_header', 'page_footer'.
- If the user asks for tables, set "label": "table".
- If one or more page numbers are mentioned, set "page_nos" to a list of ALL
  of them (e.g. "page 5" -> [5], "page 5 and 6" -> [5, 6], "pages 3, 4, 7" -> [3, 4, 7]).
- If no specific page, label, or filename is requested, set their values to null.

Return ONLY a JSON object matching this schema:
{{
  "page_nos": array of integers or null,
  "label": string or null,
  "filename": string or null
}}

Query: "{query}" """

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    raw_json = response.choices[0].message.content
    return ExtractedMetadata.model_validate_json(raw_json)


def build_qdrant_filter(extracted: ExtractedMetadata) -> Filter:
    """Translates extracted metadata into a Qdrant Filter object."""
    must_conditions = []
    
    if extracted.page_nos:
        # MatchAny: matches a chunk if ITS "pages" array contains ANY of the
        # requested page numbers — this is what makes "page 5 and 6" actually
        # search both pages, instead of only the first one found.
        must_conditions.append(
            FieldCondition(key="pages", match=MatchAny(any=extracted.page_nos))
        )
    if extracted.label is not None:
        must_conditions.append(
            FieldCondition(key="labels", match=MatchValue(value=extracted.label))
        )
    if extracted.filename is not None:
        must_conditions.append(
            FieldCondition(key="filename", match=MatchValue(value=extracted.filename))
        )

    must_not_conditions = []
    if extracted.label not in ["page_header", "page_footer"]:
        must_not_conditions.extend([
            FieldCondition(key="labels", match=MatchValue(value="page_header")),
            FieldCondition(key="labels", match=MatchValue(value="page_footer")),
        ])

    return Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None
    )


# -----------------------------
# 3b. Filter-Only Fetch (for explicit page/filename scope — no vector search)
# -----------------------------
def fetch_all_filtered_chunks(collection_name: str, qdrant_filter: Filter, cap: int = FILTERED_FETCH_CAP):
    """Fetches ALL chunks matching an explicit filter, using Qdrant's scroll
    API — this is a plain filter lookup, NOT a semantic/keyword search, so
    no embeddings are computed for this path at all.

    Use this when the user names an explicit scope (specific page(s) or a
    filename) — they've already told us exactly what's in scope, so nothing
    that matches should be silently dropped the way a fixed top_k would.

    Raises ScopeTooBroadError if more than `cap` chunks match — this is a
    safety guardrail, not a normal-case limit (see FILTERED_FETCH_CAP).
    """
    points, _next_offset = client.scroll(
        collection_name=collection_name,
        scroll_filter=qdrant_filter,
        limit=cap + 1,  # +1 lets us detect ">cap matches" in a single call
        with_payload=True,
        with_vectors=False,
    )

    if len(points) > cap:
        raise ScopeTooBroadError(
            f"This filter matches more than {cap} chunks — that's too broad "
            f"to answer accurately in one go. Try narrowing to fewer pages "
            f"or a more specific question."
        )

    return points

# -----------------------------
# 4. Search Execution Engine (fused hybrid search)
# -----------------------------
def search_hybrid(query: str, collection_name: str, candidate_pool: int = RRF_CANDIDATE_POOL, prefetch_limit: int = 40, debug_top_k: int = 5):
    """Runs retrieval for a query, choosing one of two strategies:

    - EXPLICIT SCOPE (page_nos and/or filename given): skips vector search
      entirely and fetches ALL chunks matching that filter via Qdrant scroll
      (see fetch_all_filtered_chunks). The user already told us the exact
      scope — a fixed top_k would silently drop chunks that belong there.
    - NO EXPLICIT SCOPE (open-ended question): dense + sparse search fused
      with RRF, as before — this is genuine "find the most relevant chunks
      out of the whole collection" search, where a relevance cutoff makes sense.

    Returns a tuple: (candidates, dense_only_results, sparse_only_results, is_filtered)
    - candidates: either ALL filtered chunks (explicit-scope path) or the
      WIDE RRF-fused candidate_pool (semantic path) — either way, meant to
      be fed into rerank_with_cross_encoder() next, NOT the final answer.
    - dense_only_results / sparse_only_results: debug-only, top `debug_top_k`
      from each branch individually. Empty lists on the explicit-scope path,
      since no vector search runs there at all.
    - is_filtered: True if the explicit-scope path was used. Callers should
      use this to avoid re-truncating an already-exact scope down to a
      small top_k during reranking.

    Raises ScopeTooBroadError if an explicit filter matches too many chunks
    (see FILTERED_FETCH_CAP) — callers should catch this and ask the user
    to narrow their question.
    """
    
    # Check if target collection exists
    if not client.collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist in Qdrant!")

    # Check the collection actually has the "dense" and "sparse" named vectors
    # this pipeline expects. Without this, Qdrant fails with a raw 400 error
    # ("Not existing vector name error: dense") that doesn't tell you WHY —
    # usually it means you picked a stale/incompatible collection (e.g. one
    # built before this pipeline's schema, or by a different script).
    collection_info = client.get_collection(collection_name)
    existing_vectors = set(collection_info.config.params.vectors.keys())
    required_vectors = {"dense"}
    existing_sparse = set(collection_info.config.params.sparse_vectors.keys()) if collection_info.config.params.sparse_vectors else set()
    if not required_vectors.issubset(existing_vectors) or "sparse" not in existing_sparse:
        raise ValueError(
            f"Collection '{collection_name}' is missing expected vectors. "
            f"Found dense vectors: {existing_vectors}, sparse vectors: {existing_sparse}. "
            f"Expected 'dense' and 'sparse'. This collection was probably not built by "
            f"store_qdrant.py in this pipeline — pick a different collection."
        )

    # A. Extract query metadata pre-filters
    extracted_meta = extract_metadata_filters(query)
    print(f"\n[Groq Extracted Filters]: {extracted_meta.model_dump_json(exclude_none=True)}")
    
    # B. Build Qdrant filter
    qdrant_filter = build_qdrant_filter(extracted_meta)

    # ---- BRANCH: explicit scope given -> fetch ALL matching chunks, skip vector search ----
    has_explicit_scope = bool(extracted_meta.page_nos) or bool(extracted_meta.filename)
    if has_explicit_scope:
        all_filtered = fetch_all_filtered_chunks(collection_name, qdrant_filter)
        return all_filtered, [], [], True

    # ---- BRANCH: no explicit scope -> existing semantic dense+sparse+RRF search ----
    # C. Embed query text into dense and sparse spaces
    dense_vec = dense_model.encode(
        BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()
    sparse_vec = list(sparse_model.embed([query]))[0]
    sparse_query = SparseVector(
        indices=sparse_vec.indices.tolist(),
        values=sparse_vec.values.tolist(),
    )

    # D. Fused hybrid search: prefetch dense + sparse candidates, then
    #    combine them server-side with Reciprocal Rank Fusion (RRF).
    #    NOTE: limit=candidate_pool (25), not the final top_k — we deliberately
    #    keep this pool wide so the cross-encoder has real material to rerank.
    fused_results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                filter=qdrant_filter,
                limit=prefetch_limit,
            ),
            Prefetch(
                query=sparse_query,
                using="sparse",
                filter=qdrant_filter,
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=candidate_pool,
        with_payload=True,
    ).points

    # E. Standalone dense-only and sparse-only searches, purely for visibility
    #    into what each retrieval method finds on its own. These do NOT feed
    #    into the fusion/reranking pipeline above — they're just for printing.
    dense_only_results = client.query_points(
        collection_name=collection_name,
        query=dense_vec,
        using="dense",
        query_filter=qdrant_filter,
        with_payload=True,
        limit=debug_top_k,
    ).points

    sparse_only_results = client.query_points(
        collection_name=collection_name,
        query=sparse_query,
        using="sparse",
        query_filter=qdrant_filter,
        with_payload=True,
        limit=debug_top_k,
    ).points

    return fused_results, dense_only_results, sparse_only_results, False


# -----------------------------
# 4b. Cross-Encoder Reranking Stage
# -----------------------------
def rerank_with_cross_encoder(query: str, candidates, top_k: Optional[int] = 5):
    """Re-scores candidates with a cross-encoder for true relevance ordering.

    Unlike the bi-encoder (dense_model), which embeds the query and each
    chunk SEPARATELY and compares vectors, a cross-encoder reads the query
    and chunk TOGETHER in a single forward pass — much more accurate at
    judging "does this chunk really answer this query", but too slow to
    run over an entire collection, which is why it's only used to rerank
    a shortlist rather than the whole corpus.

    Pass top_k=None to keep ALL candidates, just reordered by relevance —
    use this for the explicit-scope (filtered) retrieval path, where the
    user already defined the exact scope and truncation would silently
    drop chunks that belong there. Pass a number (default 5) to truncate,
    for the open-ended semantic search path.

    Returns a list of (point, rerank_score) tuples, sorted by rerank_score
    descending, truncated to top_k (or all of them, if top_k is None).

    NOTE: qdrant_client's ScoredPoint is a Pydantic model, which does NOT
    allow setting arbitrary new attributes on an instance (e.g.
    `point.rerank_score = x` raises ValueError: "ScoredPoint" object has
    no field "rerank_score"). So instead of mutating the point, we keep
    the point and its cross-encoder score as a separate (point, score) pair.
    """
    if not candidates:
        return []

    pairs = [(query, point.payload["text"]) for point in candidates]
    scores = cross_encoder.predict(pairs)

    scored = list(zip(candidates, (float(s) for s in scores)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored if top_k is None else scored[:top_k]


def print_retrieval_breakdown(dense_points, sparse_points):
    """Pretty-print the standalone dense-only and sparse-only results.

    Purely informational — shows what each retrieval method found
    independently, before RRF fusion and cross-encoder reranking touch them.
    Empty on the explicit-scope (filtered) retrieval path, since no vector
    search runs there at all.
    """
    print("\n" + "=" * 90)
    print("--- [DEBUG] DENSE-ONLY RESULTS (Semantic / Cosine Similarity) ---")
    print("=" * 90)
    for i, point in enumerate(dense_points, start=1):
        print(f"Result {i} | Cosine Score: {point.score:.4f} | Pages: {point.payload.get('pages')}")
        print("Text:", point.payload["text"][:200] + "...\n")

    print("=" * 90)
    print("--- [DEBUG] SPARSE-ONLY RESULTS (BM25 Keyword Match) ---")
    print("=" * 90)
    for i, point in enumerate(sparse_points, start=1):
        print(f"Result {i} | BM25 Score: {point.score:.4f} | Pages: {point.payload.get('pages')}")
        print("Text:", point.payload["text"][:200] + "...\n")


def print_search_results(scored_points, is_filtered: bool = False):
    """Pretty-print final, cross-encoder-reranked search results.

    Expects a list of (point, rerank_score) tuples, as returned by
    rerank_with_cross_encoder(). On the explicit-scope (filtered) path,
    points come from Qdrant's scroll() and have no similarity `.score`
    (they were fetched by exact filter match, not ranked search) — so
    that column is skipped there instead of erroring.
    """
    print("\n" + "=" * 90)
    header = "--- FINAL RESULTS (Explicit scope: all matching chunks, cross-encoder ordered) ---" \
        if is_filtered else \
        "--- FINAL RESULTS (Dense + Sparse -> RRF fusion -> Cross-Encoder rerank) ---"
    print(header)
    print("=" * 90)
    for i, (point, rerank_score) in enumerate(scored_points, start=1):
        if is_filtered:
            print(f"Result {i} | Cross-Encoder Score: {rerank_score:.4f} | Pages: {point.payload.get('pages')}")
        else:
            print(f"Result {i} | Cross-Encoder Score: {rerank_score:.4f} | RRF Score: {point.score:.4f} | Pages: {point.payload.get('pages')}")
        print("Text:", point.payload["text"][:200] + "...\n")


# -----------------------------
# 5. CLI Collection Selection Helper
# -----------------------------
def get_user_selected_collection() -> str:
    """Helper for local testing: Lists collections and gets explicit user choice.

    Accepts either the list index (e.g. "2") OR the collection name typed
    directly (e.g. "sample"). Previously, typing a name here silently fell
    through to available_names[0] with no warning — you could end up
    querying a completely different (and possibly schema-incompatible)
    collection than the one you intended.
    """
    collections = client.get_collections().collections
    if not collections:
        print("[ERROR] No collections found in Qdrant! Please run store_qdrant.py first.")
        sys.exit(1)

    available_names = [c.name for c in collections]

    if len(available_names) == 1:
        print(f"Found 1 collection: '{available_names[0]}'")
        return available_names[0]

    print("\nAvailable PDF Collections in Qdrant:")
    for idx, name in enumerate(available_names, start=1):
        print(f"  [{idx}] {name}")

    selection = input(f"\nSelect PDF collection (1-{len(available_names)} or type the name): ").strip()

    # Numeric index
    if selection.isdigit() and 1 <= int(selection) <= len(available_names):
        return available_names[int(selection) - 1]

    # Typed name (case-insensitive exact match)
    for name in available_names:
        if name.lower() == selection.lower():
            return name

    # Nothing matched — fail loudly instead of silently picking available_names[0]
    print(f"[ERROR] '{selection}' is not a valid index or collection name. "
          f"Valid options: {available_names}")
    sys.exit(1)


if __name__ == "__main__":
    # Explicit collection selection (Simulates UI/frontend active document selection)
    target_collection = get_user_selected_collection()
    print(f"\nActive Target Collection: '{target_collection}'")

    user_query = input("Enter search query: ")

    try:
        # Stage 1 (recall-focused): either fetch-all-by-filter (explicit scope)
        # or dense + sparse + RRF -> wide candidate pool (open-ended question)
        candidates, dense_only, sparse_only, is_filtered = search_hybrid(
            query=user_query,
            collection_name=target_collection,
            candidate_pool=RRF_CANDIDATE_POOL,
            debug_top_k=5,
        )
    except ScopeTooBroadError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print_retrieval_breakdown(dense_only, sparse_only)

    # Stage 2 (precision-focused): cross-encoder orders the candidates.
    # Explicit scope -> keep ALL of them (top_k=None); open-ended -> top 5.
    final_results = rerank_with_cross_encoder(
        query=user_query,
        candidates=candidates,
        top_k=None if is_filtered else 5,
    )

    print_search_results(final_results, is_filtered=is_filtered)