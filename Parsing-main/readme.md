# PDF RAG Pipeline (Docling + Qdrant + Groq)

A local Retrieval-Augmented Generation (RAG) pipeline that parses PDFs (including OCR, tables, and image/chart descriptions), chunks them, stores dense + sparse embeddings in Qdrant, and answers questions grounded in the document using hybrid search, cross-encoder reranking, and Groq's Llama 3.3 70B.

## How it works

```
input/*.pdf
    │
    ▼
┌─────────────┐
│  parse.py   │  docling: OCR + table structure + picture (chart/image) description
└─────────────┘
    │  → output/parsed/sample_doc.pkl, result.md, result.json
    ▼
┌──────────────┐
│ chunking.py  │  HybridChunker, tokenized with the SAME model used for embedding
└──────────────┘
    │  → output/chunks/chunked_data.pkl
    ▼
┌────────────────┐
│ store_qdrant.py │  Dense (bge-small-en-v1.5) + Sparse (BM25) embeddings → Qdrant
└────────────────┘
    │  → Qdrant collection named after the PDF filename
    ▼
┌─────────────┐
│ generate.py │  Query → Dense+Sparse+RRF fusion → Cross-encoder rerank → Groq LLM answer
└─────────────┘
```

- `pipeline.py` runs **parse → chunk → store** in one command — this is the ingestion step, run once per PDF.
- `generate.py` is the ongoing querying step — run any time you want to ask questions. It lets you pick which ingested document (Qdrant collection) to query if more than one exists.
- `query.py` is the retrieval engine underneath `generate.py` (also runnable standalone for debugging retrieval without generation).

## Models used

| Stage | Model | Purpose |
|---|---|---|
| Parsing — OCR | EasyOCR (`en`) | Extract text from scanned/image-based PDF content |
| Parsing — image description | `Qwen/Qwen2.5-VL-3B-Instruct` | Generates factual descriptions of charts/images/tables in the PDF |
| Chunking | `BAAI/bge-small-en-v1.5` tokenizer | Sizes chunks to match the embedding model's token limit (not used for embedding here) |
| Dense embedding | `BAAI/bge-small-en-v1.5` | 384-dim, cosine distance, normalized. Queries use an asymmetric instruction prefix; documents do not |
| Sparse embedding | `Qdrant/bm25` (FastEmbed) | BM25-style keyword matching, IDF modifier enabled in Qdrant |
| Fusion | Qdrant native RRF | Combines dense + sparse ranked lists server-side |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Rescopes the fused candidate pool by reading (query, chunk) pairs jointly |
| Query filter extraction | `llama-3.3-70b-versatile` (Groq) | Parses natural-language questions into structured filters (page, label, filename) |
| Answer generation | `llama-3.3-70b-versatile` (Groq) | Generates the final grounded answer from retrieved context |

## Project structure

```
project/
├── input/                  # Drop PDF(s) here before running pipeline.py
├── output/
│   ├── parsed/              # sample_doc.pkl, result.md, result.json
│   └── chunks/               # chunked_data.pkl
├── parse.py                 # Stage 1: parsing
├── chunking.py               # Stage 2: chunking
├── store_qdrant.py           # Stage 3: embedding + storing in Qdrant
├── pipeline.py                # Runs stages 1-3 in sequence
├── query.py                   # Hybrid retrieval engine (dense+sparse+RRF+rerank)
├── generate.py                 # RAG Q&A loop (uses query.py + Groq)
├── requirements.txt
├── .env                        # GROQ_API_KEY (not committed — see .gitignore)
└── README.md
```

## Setup

### 1. Clone and create a virtual environment
```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> First run of `parse.py` will download several models (OCR, VLM, embedding models) — expect this to take a while and require internet access. Models are cached locally afterward.

### 3. Start Qdrant locally
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Add your Groq API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com) → API Keys.

## Usage

### Ingest a PDF
1. Drop a PDF into `input/`.
2. Run:
   ```bash
   python pipeline.py
   ```
   If `input/` has multiple PDFs, you'll be prompted to pick one (only one PDF is processed per run).
3. This creates a Qdrant collection named after the PDF (sanitized, e.g. `Sustainability Report 2023.pdf` → `sustainability_report_2023`).

### Ask questions
```bash
python generate.py
```
- If multiple collections exist, you'll be prompted to pick one.
- Type `exit`, `bye`, `done`, or `quit` to end the session.
- Casual messages (greetings, thanks, etc.) skip retrieval entirely and are answered directly.
- Questions naming a specific page or filename skip vector search and fetch **all** matching chunks exactly (capped at 100 chunks as a safety guardrail).
- Open-ended questions use dense + sparse hybrid search (RRF fusion) over a wide candidate pool, then cross-encoder reranking narrows it to the top 5 most relevant chunks before generation.

### Debug retrieval only (no generation)
```bash
python query.py
```
Prints dense-only, sparse-only, and final reranked results with scores — useful for tuning/debugging without spending Groq calls.

## Notes

- One PDF = one Qdrant collection. Re-running `pipeline.py` on a PDF with the same filename will **recreate** (overwrite) that collection.
- The dense embedding model must stay identical between ingestion (`store_qdrant.py`) and retrieval (`query.py`) — both are hardcoded to `BAAI/bge-small-en-v1.5` for this reason.
- `query.py` and `generate.py` require both Qdrant (running) and a valid `GROQ_API_KEY` to function.
