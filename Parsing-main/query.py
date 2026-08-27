"""
STEP 4: HYBRID QUERY WITH EXPLICIT COLLECTION SELECTION
-------------------------------------------------------
- Accepts explicit 'collection_name' from caller (UI/API/CLI).
- Uses Groq API to parse metadata pre-filters (pages, labels).
- Executes a single fused Dense (bge-m3) + Sparse (BM25) hybrid search
  over the target collection, combined server-side via Qdrant's RRF fusion.
- Cross-encoder reranking stage for precision ordering.

Key changes from original:
1. Switched to bge-m3 (1024 dim) for dense embeddings.
2. Labels metadata filter now supports List[str] (multiple labels).
3. LangSmith tracing on all functions.
4. Model updated from deprecated llama-3.3-70b-versatile to openai/gpt-oss-120b.
"""

import sys
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langsmith import traceable, trace
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
dense_model = SentenceTransformer("BAAI/bge-m3")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# bge-m3 is symmetric: no instruction prefix needed for queries.
BGE_QUERY_PREFIX = ""

# How many candidates RRF fusion should surface BEFORE reranking.
RRF_CANDIDATE_POOL = 25

# Safety cap for the filter-only fetch path.
FILTERED_FETCH_CAP = 100

client = QdrantClient(host="localhost", port=6333)
groq_client = Groq()


class ScopeTooBroadError(Exception):
    """Raised when an explicit filter matches more than FILTERED_FETCH_CAP chunks."""
    pass

# -----------------------------
# 2. Define Metadata Schema
# -----------------------------
class ExtractedMetadata(BaseModel):
    page_nos: Optional[List[int]] = Field(
        default=None,
        description="Page number(s) explicitly mentioned in query, as a list (e.g., [4], [5, 6])"
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Content labels if specifically requested: 'text', 'section_header', 'table', 'list_item', 'picture'"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Filename if explicitly mentioned in query"
    )

# -----------------------------
# 3. Dynamic Metadata Extractor
# -----------------------------
@traceable(run_type="chain", name="extract_metadata_filters")
def extract_metadata_filters(query: str) -> ExtractedMetadata:
    """Uses Groq LLM to parse intent and extract metadata criteria as JSON."""
    prompt = f"""You are a query parser for a database. Analyze the query and extract metadata filters.
Available content labels: 'text', 'section_header', 'table', 'list_item', 'picture', 'page_header', 'page_footer'.
- If the user asks for tables, set "labels": ["table"].
- If the user asks for multiple types (e.g. tables and pictures), set "labels": ["table", "picture"].
- If one or more page numbers are mentioned, set "page_nos" to a list of ALL
  of them (e.g. "page 5" -> [5], "page 5 and 6" -> [5, 6], "pages 3, 4, 7" -> [3, 4, 7]).
- If no specific page, labels, or filename is requested, set their values to null.

Return ONLY a JSON object matching this schema:
{{
  "page_nos": array of integers or null,
  "labels": array of strings or null,
  "filename": string or null
}}

Query: "{query}" """

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    raw_json = response.choices[0].message.content
    return ExtractedMetadata.model_validate_json(raw_json)


@traceable(run_type="chain", name="build_qdrant_filter")
def build_qdrant_filter(extracted: ExtractedMetadata) -> Filter:
    """Translates extracted metadata into a Qdrant Filter object."""
    must_conditions = []

    if extracted.page_nos:
        must_conditions.append(
            FieldCondition(key="pages", match=MatchAny(any=extracted.page_nos))
        )
    if extracted.labels is not None:
        must_conditions.append(
            FieldCondition(key="labels", match=MatchAny(any=extracted.labels))
        )
    if extracted.filename is not None:
        must_conditions.append(
            FieldCondition(key="filename", match=MatchValue(value=extracted.filename))
        )

    # Exclude headers/footers unless the user specifically asked for them
    must_not_conditions = []
    excluded_labels = {"page_header", "page_footer"}
    requested_labels = set(extracted.labels) if extracted.labels else set()
    if not requested_labels.intersection(excluded_labels):
        must_not_conditions.extend([
            FieldCondition(key="labels", match=MatchValue(value="page_header")),
            FieldCondition(key="labels", match=MatchValue(value="page_footer")),
        ])

    return Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None
    )


# -----------------------------
# 3b. Filter-Only Fetch (for explicit page/filename scope)
# -----------------------------
@traceable(run_type="chain", name="fetch_all_filtered_chunks")
def fetch_all_filtered_chunks(collection_name: str, qdrant_filter: Filter, cap: int = FILTERED_FETCH_CAP):
    """Fetches ALL chunks matching an explicit filter via Qdrant's scroll API."""
    points, _next_offset = client.scroll(
        collection_name=collection_name,
        scroll_filter=qdrant_filter,
        limit=cap + 1,
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
@traceable(run_type="retriever", name="search_hybrid")
def search_hybrid(query: str, collection_name: str, candidate_pool: int = RRF_CANDIDATE_POOL, prefetch_limit: int = 40, debug_top_k: int = 5):
    """Runs retrieval for a query, choosing one of two strategies:

    - EXPLICIT SCOPE (page_nos and/or filename given): skips vector search
      entirely and fetches ALL chunks matching that filter via Qdrant scroll.
    - NO EXPLICIT SCOPE (open-ended question): dense + sparse search fused
      with RRF.

    Returns a tuple: (candidates, dense_only_results, sparse_only_results, is_filtered)
    """

    # Check if target collection exists
    if not client.collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist in Qdrant!")

    # Validate collection schema
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

    # ---- BRANCH: explicit scope given -> fetch ALL matching chunks ----
    has_explicit_scope = bool(extracted_meta.page_nos) or bool(extracted_meta.filename)
    if has_explicit_scope:
        all_filtered = fetch_all_filtered_chunks(collection_name, qdrant_filter)
        return all_filtered, [], [], True

    # ---- BRANCH: no explicit scope -> semantic dense+sparse+RRF search ----
    # C. Embed query text into dense and sparse spaces
    with trace(name="query_embedding", run_type="chain"):
        dense_vec = dense_model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
        ).tolist()
        sparse_vec = list(sparse_model.embed([query]))[0]
        sparse_query = SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist(),
        )

    # D. Fused hybrid search via RRF
    with trace(name="rrf_fusion_search", run_type="chain"):
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

    # E. Standalone searches for debug visibility
    with trace(name="debug_individual_searches", run_type="chain"):
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
@traceable(run_type="chain", name="rerank_with_cross_encoder")
def rerank_with_cross_encoder(query: str, candidates, top_k: Optional[int] = 5):
    """Re-scores candidates with a cross-encoder for true relevance ordering."""
    if not candidates:
        return []

    pairs = [(query, point.payload["text"]) for point in candidates]
    scores = cross_encoder.predict(pairs)

    scored = list(zip(candidates, (float(s) for s in scores)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored if top_k is None else scored[:top_k]


def print_retrieval_breakdown(dense_points, sparse_points):
    """Pretty-print standalone dense-only and sparse-only results (debug)."""
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
    """Pretty-print final cross-encoder-reranked results."""
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
    """Helper for local testing: Lists collections and gets explicit user choice."""
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

    if selection.isdigit() and 1 <= int(selection) <= len(available_names):
        return available_names[int(selection) - 1]

    for name in available_names:
        if name.lower() == selection.lower():
            return name

    print(f"[ERROR] '{selection}' is not a valid index or collection name. "
          f"Valid options: {available_names}")
    sys.exit(1)


if __name__ == "__main__":
    target_collection = get_user_selected_collection()
    print(f"\nActive Target Collection: '{target_collection}'")

    user_query = input("Enter search query: ")

    try:
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

    final_results = rerank_with_cross_encoder(
        query=user_query,
        candidates=candidates,
        top_k=None if is_filtered else 5,
    )

    print_search_results(final_results, is_filtered=is_filtered)
