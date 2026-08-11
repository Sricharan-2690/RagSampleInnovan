#  End-to-End Agentic RAG Application

An end-to-end **Agentic Retrieval-Augmented Generation (RAG)** system integrated with an offline **Docling PDF Ingestion Pipeline** and a **Streamlit Web Interface**.

---

##  Architectural Overview & System Flow

The system consists of **two main workflows**:

1. **Ingestion Flow (PDF Upload)**: Converts raw PDFs ➔ parsed text/tables/images ➔ 512-token chunks ➔ dual dense/sparse vectors stored in **Qdrant**.
2. **Query Flow (Agentic RAG Chat)**: User question ➔ LangChain Agent reasoning ➔ Tool execution (Qdrant search / Calculator / Web fallback) ➔ Grounded response with page citations `(Page X)`.

```text
                                  STREAMLIT FRONTEND
                                      (app.py)
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            │ [Path 1: User Uploads PDF]                                │ [Path 2: User Asks Question]
            ▼                                                           ▼
 ┌─────────────────────┐                                     ┌─────────────────────┐
 │ Ingestion Adapter   │                                     │ Agent Executor      │
 │ (ingest_adapter.py) │                                     │ (executor.py)       │
 └──────────┬──────────┘                                     └──────────┬──────────┘
            │                                                           │
            ▼                                                           ▼
 ┌─────────────────────┐                                     ┌─────────────────────┐
 │ Teammate Pipeline   │                                     │ LangChain Agent     │
 │ (parse.py)          │                                     │ (agent.py)          │
 │ (chunking.py)       │                                     └──────────┬──────────┘
 │ (store_qdrant.py)   │                                                │
 └──────────┬──────────┘                                                ▼
            │                                                ┌─────────────────────┐
            │ Writes Vectors                                 │ Tools Module        │
            ▼                                                │ (tools.py)          │
┌─────────────────────────┐                                  └──────────┬──────────┘
│ Qdrant Vector Database  │◄────────────────────────────────────────────┘
│ (Collection)            │                Reads & Searches Vectors (query.py)
└─────────────────────────┘
```

---

##  Directory Structure

```text
c:\TASKS\T2\AgenticRAG\
├── Parsing-main/                    # Teammate's Ingestion & Retrieval Pipeline
│   ├── parse.py                     # Stage 1: Docling OCR, table extraction, VLM image descriptions
│   ├── chunking.py                  # Stage 2: HybridChunker (512 tokens with heading context)
│   ├── store_qdrant.py              # Stage 3: Dense (bge-small) + Sparse (BM25) vector storage
│   ├── pipeline.py                  # Orchestrator for stages 1-3
│   ├── query.py                     # Hybrid vector search + RRF + Cross-encoder reranker
│   └── requirements.txt
│
├── agentic_rag/                     # YOUR AGENTIC RAG SYSTEM (Dedicated Folder)
│   ├── __init__.py                  # Package initializer
│   ├── config.py                    # Environment settings, Qdrant host/port, & collection helpers
│   ├── ingest_adapter.py            # Bridge to trigger teammate's PDF ingestion from UI
│   ├── tools.py                     # LangChain Tools (Qdrant search, calculator, web search)
│   ├── agent.py                     # LangChain ReAct / Tool-Calling Agent & Prompts
│   └── executor.py                  # Agent executor runner with memory & trace logs
│
├── app.py                           # Streamlit UI (PDF Upload + Interactive Agentic Chat)
├── requirements.txt                 # Unified project dependencies
├── .env                             # Environment variables (GROQ_API_KEY, QDRANT_HOST, etc.)
└── README.md                        # Complete project documentation
```

---

## 🔄 Sequential File-by-File Flow

### 1️⃣ Ingestion Flow (When a User Uploads a PDF)

1. **[app.py](file:///c:/TASKS/T2/AgenticRAG/app.py)**: The user selects a PDF in the sidebar and clicks **`Process & Index PDF`**. Passes the file buffer to `ingest_uploaded_pdf()`.
2. **[agentic_rag/ingest_adapter.py](file:///c:/TASKS/T2/AgenticRAG/agentic_rag/ingest_adapter.py)**: Saves the uploaded PDF to `Parsing-main/input/<filename>.pdf` and executes the ingestion stages sequentially.
3. **[Parsing-main/parse.py](file:///c:/TASKS/T2/AgenticRAG/Parsing-main/parse.py)**: 
   - Uses **Docling `DocumentConverter`** with **EasyOCR** and **TableFormer** (`ACCURATE` mode).
   - Uses `Qwen2.5-VL-3B-Instruct` VLM to generate factual text descriptions of charts and images.
   - Outputs `sample_doc.pkl`, `result.md`, and `result.json`.
4. **[Parsing-main/chunking.py](file:///c:/TASKS/T2/AgenticRAG/Parsing-main/chunking.py)**:
   - Uses Docling `HybridChunker` tokenized with `BAAI/bge-small-en-v1.5` (max 512 tokens).
   - Prepends section heading hierarchies to each chunk via `chunker.contextualize()`.
   - Extracts metadata: `headings`, `pages`, `labels`, `filename`.
5. **[Parsing-main/store_qdrant.py](file:///c:/TASKS/T2/AgenticRAG/Parsing-main/store_qdrant.py)**:
   - Generates **Dense vectors** (`BAAI/bge-small-en-v1.5`, 384 dimensions, Cosine distance).
   - Generates **Sparse vectors** (`Qdrant/bm25` FastEmbed with `Modifier.IDF`).
   - Sanitizes PDF filename into a clean Qdrant collection name (e.g. `resume_ggl`).
   - Creates payload indexes for `pages`, `labels`, and `filename`, and upserts points.

---

### 2️⃣ Query Flow (When a User Asks a Question)

1. **[app.py](file:///c:/TASKS/T2/AgenticRAG/app.py)**: User inputs a message in the chat input box. Passes the prompt to `runner.run()`.
2. **[agentic_rag/executor.py](file:///c:/TASKS/T2/AgenticRAG/agentic_rag/executor.py)**: `RAGAgentRunner` invokes `AgentExecutor` with `return_intermediate_steps=True`.
3. **[agentic_rag/agent.py](file:///c:/TASKS/T2/AgenticRAG/agentic_rag/agent.py)**: Configures **ChatGroq LLM** (`llama-3.3-70b-versatile`) and system prompt guardrails. The agent evaluates the question and selects a tool:
   - Document question ➔ Calls `search_knowledge_base`
   - Math calculation ➔ Calls `calculator_tool`
   - External web query ➔ Calls `web_search_tool`
4. **[agentic_rag/tools.py](file:///c:/TASKS/T2/AgenticRAG/agentic_rag/tools.py)**:
   - For document questions, calls `search_hybrid()` and `rerank_with_cross_encoder()` from **[Parsing-main/query.py](file:///c:/TASKS/T2/AgenticRAG/Parsing-main/query.py)**.
   - `query.py` extracts metadata filters with Groq LLM (e.g. `page_nos: [1]`), executes server-side RRF fusion in Qdrant over dense + sparse vectors to get top 25 candidates, and re-scores candidates with `ms-marco-MiniLM-L-6-v2` cross-encoder to select top 5 chunks.
5. **[app.py](file:///c:/TASKS/T2/AgenticRAG/app.py)**: Receives final response and step traces. Renders an expandable **`🛠️ Agent Tool Execution Trace`** box and prints the answer with page citations `(Page X)`.

---

## 🧠 Key Technical Concepts & Features

### Dense vs. Sparse Hybrid Search

| Feature | Dense Vector (`bge-small-en-v1.5`) | Sparse Vector (`Qdrant/bm25`) |
| :--- | :--- | :--- |
| **What it captures** | **Semantic Meaning & Synonyms** | **Exact Keywords, Names, & Numbers** |
| **Data Structure** | 384 floating-point numbers | High-dimensional dictionary of token weights |
| **Example Match** | *"vacation"* matching *"time off"* | *"ERR-404-X"* matching *"ERR-404-X"* |

* **Reciprocal Rank Fusion (RRF)**: Qdrant merges dense and sparse ranked candidate pools server-side.
* **Cross-Encoder Reranking**: Jointly scores `(query, chunk)` candidate pairs using `ms-marco-MiniLM-L-6-v2` for precise top-k relevance ordering.

### Streamlit Architecture
* Streamlit runs both the UI and local Python backend in a single process.
* `app.py` imports backend functions directly without requiring intermediate REST APIs (FastAPI/Flask).

---

## 🛠️ Resolved Issues & Stability Enhancements

1. **LangChain 1.3+ Compatibility**: Added dynamic fallback imports for `AgentExecutor` and `create_tool_calling_agent` from `langchain_classic.agents`.
2. **Windows Symlinks Fix (`[WinError 1314]`)**: Set `HF_HUB_DISABLE_SYMLINKS=1` so HuggingFace Hub copies model weights directly without requiring Windows Administrator or Developer Mode privileges.
3. **MSVC Compiler Missing Fix (`InvalidCxxCompiler`)**: Set `TORCHINDUCTOR_DISABLE=1` and `TORCHDYNAMO_DISABLE=1` to run PyTorch in eager mode without needing Microsoft Visual C++ `cl.exe`.
4. **Automatic Layout Fallback**: Added automatic fallback to high-accuracy layout parsing if VLM execution encounters hardware limits on CPU.
5. **Pylance Linter Cleanup**: Removed unused imports (`shutil`, `Type`) and annotated dynamic imports.

---

## 🚀 How to Run the Application

### 1. Prerequisites
Ensure Docker is running and launch Qdrant:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Install Dependencies
```bash
cd c:\TASKS\T2\AgenticRAG
pip install -r requirements.txt
```

### 3. Launch Streamlit Web UI
```bash
streamlit run app.py
```

### 4. Access Application
Open your browser at **[http://localhost:8501](http://localhost:8501)** (or `http://localhost:8502`).
