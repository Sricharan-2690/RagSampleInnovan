"""
INGESTION PIPELINE (Parse -> Chunk -> Store)
---------------------------------------------
Runs the full ingestion pipeline end-to-end for a single PDF:

    input/*.pdf
        -> parse.py       -> output/parsed/sample_doc.pkl
        -> chunking.py    -> output/chunks/chunked_data.pkl
        -> store_qdrant.py -> Qdrant collection (named after the PDF)

Each stage still writes its own intermediate pickle to disk (not just
passed in-memory) — this keeps every stage independently re-runnable /
debuggable, same as when you ran them as three separate scripts.
p
Querying/answering is NOT part of this pipeline — that's an ongoing,
separate step handled by generate.py, which lets you interactively pick
which ingested collection to ask questions against.

Usage:
    python pipeline.py
"""

from parse import run_parsing
from chunking import run_chunking
from store_qdrant import run_storage


def run_pipeline():
    print("=" * 90)
    print("STAGE 1/3: PARSING")
    print("=" * 90)
    run_parsing()

    print("\n" + "=" * 90)
    print("STAGE 2/3: CHUNKING")
    print("=" * 90)
    run_chunking()

    print("\n" + "=" * 90)
    print("STAGE 3/3: EMBEDDING + STORING IN QDRANT")
    print("=" * 90)
    collection_name = run_storage()

    print("\n" + "=" * 90)
    print("PIPELINE COMPLETE")
    print("=" * 90)
    print(f"Your document is ready to query in Qdrant collection: '{collection_name}'")
    print("Run 'python generate.py' to ask questions about it.")

    return collection_name


if __name__ == "__main__":
    run_pipeline()
