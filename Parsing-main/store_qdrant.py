"""
STEP 3: STORE DENSE + SPARSE EMBEDDINGS IN QDRANT
--------------------------------------------------
- Input  : output/chunks/chunked_data.pkl (from chunking.py)
- Output : Populated dual-vector collection in Qdrant

This script:
1. Loads document chunks and metadata.
2. Dynamically derives and sanitizes a collection name from the PDF filename.
3. Generates Dense Embeddings (bge-m3) and Sparse BM25 Embeddings (FastEmbed).
4. Configures a dual-vector Qdrant collection (named 'dense' and 'sparse').
5. Creates payload indexes for fast metadata pre-filtering.
6. Upserts points into Qdrant.

Usage:
    python store_qdrant.py
    (or import run_storage() from pipeline.py)
"""

import pickle
import uuid
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable, trace

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    PayloadSchemaType,
    Modifier,
)

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Paths
# -----------------------------
INPUT_PKL = Path("output/chunks/chunked_data.pkl")


def sanitize_collection_name(filename: str) -> str:
    """Converts a PDF filename into a valid Qdrant collection name.

    Example: 'Sustainability Report 2023.pdf' -> 'sustainability_report_2023'
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name).lower().strip('_')
    return clean_name or "default_collection"


@traceable(run_type="chain", name="run_storage")
def run_storage(input_pkl: Path = INPUT_PKL, qdrant_host: str = "localhost", qdrant_port: int = 6333) -> str:
    """Embeds chunked data and upserts it into a Qdrant collection.

    Parameters
    ----------
    input_pkl : Path, optional
        Path to chunked_data.pkl produced by chunking.py.
    qdrant_host, qdrant_port : connection details for the local Qdrant instance.

    Returns
    -------
    str: the name of the Qdrant collection that was created/populated.
    """
    input_pkl = Path(input_pkl)

    if not input_pkl.exists():
        raise FileNotFoundError(
            f"'{input_pkl}' not found. Run chunking.py first (or pass its output path here)."
        )

    # -----------------------------
    # 1. Load Chunks
    # -----------------------------
    with trace(name="load_chunks", run_type="chain"):
        with open(input_pkl, "rb") as f:
            data = pickle.load(f)

        documents = data["documents"]
        metadatas = data["metadatas"]

        raw_filename = metadatas[0].get("filename", "document.pdf")
        collection_name = sanitize_collection_name(raw_filename)

    print(f"Loaded {len(documents)} document chunks from '{raw_filename}'.")
    print(f"Target Collection Name: '{collection_name}'")

    # -----------------------------
    # 2. Generate Dense Embeddings
    # -----------------------------
    with trace(name="dense_embedding", run_type="chain"):
        print("Generating dense embeddings (BAAI/bge-m3)...")
        dense_model = SentenceTransformer("BAAI/bge-m3")
        dense_embeddings = dense_model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    # -----------------------------
    # 3. Generate Sparse BM25 Vectors
    # -----------------------------
    with trace(name="sparse_embedding", run_type="chain"):
        print("Generating sparse BM25 vectors (FastEmbed / Qdrant-BM25)...")
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        sparse_embeddings = list(sparse_model.embed(documents))

    # -----------------------------
    # 4. Configure Dual-Vector Collection in Qdrant
    # -----------------------------
    with trace(name="qdrant_setup_and_upsert", run_type="chain"):
        client = QdrantClient(host=qdrant_host, port=qdrant_port)

        if client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' already exists. Recreating...")
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE)  # bge-m3 dimension
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    modifier=Modifier.IDF  # required for correct BM25 scoring
                )
            }
        )
        print(f"Created Qdrant collection '{collection_name}' (sparse vector uses IDF modifier).")

        # -----------------------------
        # 5. Create Payload Indexes for Pre-Filtering
        # -----------------------------
        client.create_payload_index(
            collection_name=collection_name,
            field_name="pages",
            field_schema=PayloadSchemaType.INTEGER
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="labels",
            field_schema=PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="filename",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print("Created payload indexes for 'pages', 'labels', and 'filename'.")

        # -----------------------------
        # 6. Build Points & Upsert to Qdrant
        # -----------------------------
        points = []
        for dense_emb, sparse_emb, doc, meta in zip(dense_embeddings, sparse_embeddings, documents, metadatas):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_emb.tolist(),
                    "sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist()
                    )
                },
                payload={
                    "text": doc,
                    **meta
                }
            )
            points.append(point)

        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print(f"\n[SUCCESS] Uploaded {len(points)} points to collection '{collection_name}'.")
    info = client.get_collection(collection_name)
    print(f"Total points stored in '{collection_name}': {info.points_count}")

    return collection_name


if __name__ == "__main__":
    run_storage()
