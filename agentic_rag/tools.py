"""
LANGCHAIN TOOLS MODULE
----------------------
Defines tools available to the LangChain Agent:
1. search_knowledge_base: Performs hybrid Qdrant search & cross-encoder reranking
   over document collections (uses teammate's retrieval engine).
2. calculator_tool: Executes mathematical calculations cleanly.
3. web_search_tool: Fallback tool for general internet searches via DuckDuckGo.
"""

import sys
import numexpr
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Add Parsing-main directory to sys.path
PARSING_MAIN_DIR = Path(__file__).resolve().parent.parent / "Parsing-main"
if str(PARSING_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(PARSING_MAIN_DIR))

# Import retrieval engine & collection config
from query import search_hybrid, rerank_with_cross_encoder, ScopeTooBroadError  # type: ignore
from agentic_rag.config import list_qdrant_collections


# Active target collection state (can be set by Streamlit UI or agent wrapper)
ACTIVE_COLLECTION: Optional[str] = None

def set_active_collection(collection_name: str):
    """Sets the global active Qdrant collection for document search queries."""
    global ACTIVE_COLLECTION
    ACTIVE_COLLECTION = collection_name
    print(f"[Tools] Active Qdrant collection set to: '{collection_name}'")


def get_active_collection() -> Optional[str]:
    """Gets the currently selected Qdrant collection, or auto-detects if 1 exists."""
    global ACTIVE_COLLECTION
    if ACTIVE_COLLECTION:
        return ACTIVE_COLLECTION
    
    collections = list_qdrant_collections()
    if collections:
        ACTIVE_COLLECTION = collections[0]
        return ACTIVE_COLLECTION
    return None


class DocumentSearchInput(BaseModel):
    query: str = Field(
        description="The search query or topic to look up in the document knowledge base."
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Optional Qdrant collection name to search. If omitted, uses the active collection."
    )


@tool(args_schema=DocumentSearchInput)
def search_knowledge_base(query: str, collection_name: Optional[str] = None) -> str:
    """
    Searches the indexed PDF document knowledge base in Qdrant using hybrid vector search
    (Dense + BM25) fused via RRF and reranked with a Cross-Encoder.
    Use this tool whenever you need to find facts, figures, policies, tables, or text
    from the uploaded PDF documents.
    """
    target_col = collection_name or get_active_collection()
    
    if not target_col:
        return "ERROR: No PDF collection is selected or available in Qdrant. Please upload a PDF first."

    try:
        # Step 1: Recall-focused Hybrid Search
        candidates, _, _, is_filtered = search_hybrid(
            query=query,
            collection_name=target_col,
            candidate_pool=25,
            debug_top_k=5
        )

        if not candidates:
            return f"No relevant content found in collection '{target_col}' for query: '{query}'."

        # Step 2: Precision-focused Cross-Encoder Reranking
        scored_points = rerank_with_cross_encoder(
            query=query,
            candidates=candidates,
            top_k=None if is_filtered else 5
        )

        if not scored_points:
            return f"No matching chunks passed relevance threshold for query: '{query}'."

        # Step 3: Format Context Output with Metadata & Page Citations
        context_blocks = [f"[Active Qdrant Collection Searched: '{target_col}']"]
        for idx, (point, rerank_score) in enumerate(scored_points, start=1):
            text = point.payload.get("text", "").strip()
            pages = point.payload.get("pages", [])
            filename = point.payload.get("filename", "document.pdf")
            headings = point.payload.get("headings", [])

            page_str = f"Page {', '.join(str(p) for p in pages)}" if pages else "Page unknown"
            heading_str = f" > ".join(headings) if headings else ""

            header = f"[Chunk {idx} | Source: {filename} | {page_str}]"
            if heading_str:
                header += f" ({heading_str})"

            context_blocks.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(context_blocks)

    except ScopeTooBroadError as e:
        return f"SCOPE ERROR: {str(e)}"
    except Exception as e:
        return f"ERROR during document retrieval: {str(e)}"


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Mathematical expression to evaluate, e.g. '2500 * 0.15' or '120 + 450'."
    )


@tool(args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """
    Calculates numerical and mathematical expressions.
    Use this tool whenever you need to compute mathematical totals, percentages,
    differences, or averages accurately.
    """
    try:
        # Sanitize and evaluate math expression safely using numexpr
        clean_expr = expression.replace("^", "**").strip()
        result = numexpr.evaluate(clean_expr).item()
        return f"Calculation Result: {clean_expr} = {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


class WebSearchInput(BaseModel):
    query: str = Field(
        description="Search query to run on the public internet."
    )


@tool(args_schema=WebSearchInput)
def web_search_tool(query: str) -> str:
    """
    Searches the live public web using DuckDuckGo.
    Use this tool ONLY when information is explicitly NOT present in the PDF document
    knowledge base or when recent real-world internet search is requested.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        ddg = DuckDuckGoSearchRun()
        results = ddg.run(query)
        return f"Web Search Results for '{query}':\n{results}"
    except Exception as e:
        return f"Web Search unavailable or failed: {str(e)}"


def get_all_tools():
    """Returns the list of all tools available to the Agent."""
    return [search_knowledge_base, calculator_tool, web_search_tool]
