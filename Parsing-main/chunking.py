"""
STEP 2: CHUNKING
-----------------
Takes the parsed docling document (output/parsed/sample_doc.pkl, produced
by parse.py) and splits it into small, meaningful chunks using docling's
HybridChunker.

Input  : output/parsed/sample_doc.pkl   (docling Document object)
Output : output/chunks/chunked_data.pkl (a dict with "documents" and "metadatas" lists)

The tokenizer used to size chunks matches the tokenizer of the model
actually used to embed chunks in store_qdrant.py / query.py (bge-m3),
so chunk sizes respect the real embedding model's token limit.

Usage:
    python chunking.py
    (or import run_chunking() from pipeline.py)
"""

import pickle
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable, trace

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Paths
# -----------------------------
INPUT_PKL = Path("output/parsed/sample_doc.pkl")
OUTPUT_DIR = Path("output/chunks")
OUTPUT_PKL = OUTPUT_DIR / "chunked_data.pkl"

# IMPORTANT: this must be the SAME model used for embedding in store_qdrant.py
# and query.py, so chunk sizes line up with what the embedding model expects.
EMBED_MODEL_ID = "BAAI/bge-m3"


@traceable(run_type="chain", name="build_chunker")
def build_chunker() -> HybridChunker:
    """Sets up the HybridChunker with the embedding model's own tokenizer."""
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
        max_tokens=512,
    )

    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,        # merge small neighboring chunks together
        merge_list_items=True,   # keep bullet-point lists together in one chunk
    )


def slim_metadata(chunk):
    """Extract only the useful metadata fields we need for filtering later."""
    meta = chunk.meta
    pages = sorted({
        prov.page_no
        for item in meta.doc_items
        for prov in item.prov
    })
    labels = sorted({item.label.value for item in meta.doc_items})

    return {
        "headings": meta.headings or [],
        "pages": pages,
        "labels": labels,
        "filename": meta.origin.filename,
    }


@traceable(run_type="chain", name="run_chunking")
def run_chunking(input_pkl: Path = INPUT_PKL, output_pkl: Path = OUTPUT_PKL):
    """Runs chunking on a parsed docling Document and saves the result.

    Parameters
    ----------
    input_pkl : Path, optional
        Path to sample_doc.pkl produced by parse.py.
    output_pkl : Path, optional
        Where chunked_data.pkl is written.

    Returns
    -------
    dict with "documents" (list[str]) and "metadatas" (list[dict]).
    """
    input_pkl = Path(input_pkl)
    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    if not input_pkl.exists():
        raise FileNotFoundError(
            f"'{input_pkl}' not found. Run parse.py first (or pass its output path here)."
        )

    with trace(name="load_document", run_type="chain"):
        with open(input_pkl, "rb") as f:
            docs = pickle.load(f)

    chunker = build_chunker()

    with trace(name="chunking", run_type="chain"):
        chunks = list(chunker.chunk(dl_doc=docs))
        print(f"Total chunks created: {len(chunks)}")

        documents = []   # the actual chunk text (with context prepended)
        metadatas = []   # metadata dict per chunk

        for chunk in chunks:
            documents.append(chunker.contextualize(chunk=chunk))
            metadatas.append(slim_metadata(chunk))

    data = {
        "documents": documents,
        "metadatas": metadatas,
    }

    with trace(name="save_chunks", run_type="chain"):
        with open(output_pkl, "wb") as f:
            pickle.dump(data, f)

    print(f"Saved '{output_pkl}' with {len(documents)} chunks.")
    return data


if __name__ == "__main__":
    run_chunking()
