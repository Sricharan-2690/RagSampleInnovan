"""
INGESTION ADAPTER MODULE
-------------------------
Bridges the Streamlit UI with your teammate's offline ingestion pipeline (Parsing-main).
Allows users to upload PDFs directly from the UI, triggers Docling parsing,
Hybrid Chunking, Embedding generation, and Qdrant storage.
"""

import os
import sys
from pathlib import Path
from langsmith import Client

# Disable Hugging Face symlinks & PyTorch Dynamo/Inductor JIT compilers on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# Add Parsing-main to sys.path so we can import parse, chunking, store_qdrant
PARSING_MAIN_DIR = Path(__file__).resolve().parent.parent / "Parsing-main"
if str(PARSING_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(PARSING_MAIN_DIR))

# Import teammate's ingestion functions
from parse import run_parsing  # type: ignore
from chunking import run_chunking  # type: ignore
from store_qdrant import run_storage  # type: ignore


def ingest_uploaded_pdf(uploaded_file) -> str:
    """
    Saves a Streamlit UploadedFile to the input directory,
    runs the full parsing -> chunking -> embedding pipeline,
    and returns the resulting Qdrant collection name.

    Parameters
    ----------
    uploaded_file : st.runtime.uploaded_file_manager.UploadedFile
        The file object uploaded via Streamlit st.file_uploader().

    Returns
    -------
    str
        The name of the newly created/populated Qdrant collection.
    """
    # Ensure input folder exists
    input_dir = PARSING_MAIN_DIR / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    target_path = input_dir / uploaded_file.name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    print(f"[IngestAdapter] Saved uploaded file to: {target_path}")

    # Stage 1: Parsing
    print("[IngestAdapter] Step 1/3: Running Docling parsing...")
    docling_doc = run_parsing(pdf_path=target_path)

    # Stage 2: Chunking
    print("[IngestAdapter] Step 2/3: Running Hybrid chunking...")
    chunk_data = run_chunking()

    # Stage 3: Vector Storage
    print("[IngestAdapter] Step 3/3: Generating embeddings & storing in Qdrant...")
    collection_name = run_storage()

    print(f"[IngestAdapter] Ingestion complete. Collection: '{collection_name}'")

    # Flush LangSmith traces so runs don't stay in "running" state
    try:
        Client().flush()
    except Exception:
        pass

    return collection_name


def ingest_local_pdf(pdf_path: Path) -> str:
    """
    Ingests an existing local PDF file using the teammate's pipeline.

    Parameters
    ----------
    pdf_path : Path
        Absolute or relative path to the target PDF file.

    Returns
    -------
    str
        The resulting Qdrant collection name.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"[IngestAdapter] Processing local PDF: {pdf_path}")
    run_parsing(pdf_path=pdf_path)
    run_chunking()
    collection_name = run_storage()

    # Flush LangSmith traces
    try:
        Client().flush()
    except Exception:
        pass

    return collection_name
