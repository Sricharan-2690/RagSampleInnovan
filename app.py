"""
STREAMLIT AGENTIC RAG APPLICATION (app.py)
------------------------------------------
A rich, modern web interface supporting:
1. PDF Upload & Ingestion: Parses via Docling, chunks, embeds, and stores in Qdrant.
2. Collection Selection: Switch between multiple indexed Qdrant collections.
3. Interactive Agentic Chat: Agent decides when to search Qdrant, calculate, or web search.
4. Tool Execution Trace: Expandable UI blocks showing agent reasoning and tool outputs.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag.config import list_qdrant_collections, QDRANT_HOST, QDRANT_PORT
from agentic_rag.tools import set_active_collection, get_active_collection
from agentic_rag.ingest_adapter import ingest_uploaded_pdf
from agentic_rag.executor import get_agent_runner

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        background-color: #E2E8F0;
        color: #0F172A;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .tool-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = None


def render_sidebar():
    """Renders the Streamlit sidebar for PDF upload & collection management."""
    st.sidebar.title(" Knowledge Base")
    st.sidebar.markdown("---")

    # Qdrant Status Badge
    st.sidebar.markdown(f"**Qdrant Vector DB:** `<a href='http://{QDRANT_HOST}:{QDRANT_PORT}' target='_blank'>{QDRANT_HOST}:{QDRANT_PORT}</a>`", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Step 1: PDF Upload Section
    st.sidebar.subheader("1. Upload New PDF Document")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF to parse and index into Qdrant",
        type=["pdf"],
        help="Triggers Docling parsing, VLM image descriptions, 512 hybrid chunking & Qdrant embedding."
    )

    if uploaded_file is not None:
        if st.sidebar.button("Process & Index PDF", use_container_width=True):
            with st.spinner(f"Parsing & indexing '{uploaded_file.name}' using Docling pipeline..."):
                try:
                    new_col = ingest_uploaded_pdf(uploaded_file)
                    st.session_state.selected_collection = new_col
                    set_active_collection(new_col)
                    st.sidebar.success(f"✅ Successfully indexed into collection: **{new_col}**")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"❌ Ingestion failed: {str(e)}")

    st.sidebar.markdown("---")

    # Step 2: Collection Selector
    st.sidebar.subheader("2. Active Collection")
    collections = list_qdrant_collections()

    if collections:
        default_index = 0
        if st.session_state.selected_collection in collections:
            default_index = collections.index(st.session_state.selected_collection)

        chosen_col = st.sidebar.selectbox(
            "Select Document Collection",
            options=collections,
            index=default_index,
            help="Choose which document collection the agent should query."
        )

        if chosen_col != st.session_state.selected_collection:
            st.session_state.selected_collection = chosen_col
            set_active_collection(chosen_col)

        st.sidebar.info(f"Target Collection: **{chosen_col}**")
    else:
        st.sidebar.warning("No collections found in Qdrant. Please upload a PDF above to create one.")


def render_main():
    """Renders the main Chat Interface."""
    st.markdown("<div class='main-header'> Agentic RAG Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Hybrid Vector Search (Qdrant + BM25 + Cross-Encoder) powered by LangChain Agent & Groq</div>", unsafe_allow_html=True)

    active_col = get_active_collection()
    if active_col:
        st.info(f"🔍 Currently querying Qdrant collection: **{active_col}**")
    else:
        st.warning("⚠️ No active collection selected. Please upload a PDF in the sidebar or start Qdrant.")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display past tool execution steps if available
            if "intermediate_steps" in msg and msg["intermediate_steps"]:
                with st.expander("🛠️ Agent Tool Execution Trace"):
                    for action, output in msg["intermediate_steps"]:
                        tool_name = getattr(action, "tool", "Tool")
                        tool_input = getattr(action, "tool_input", "")
                        st.markdown(f"**Tool Invoked:** `{tool_name}`")
                        st.markdown(f"**Input:** `{tool_input}`")
                        st.text_area("Output", value=str(output), height=120, disabled=True)

    # Chat Input Box
    if prompt := st.chat_input("Ask a question about the document, request calculations, or ask for internet info..."):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Agent is reasoning and executing tools..."):
                try:
                    runner = get_agent_runner()
                    result = runner.run(user_input=prompt)
                    
                    answer = result.get("output", "")
                    steps = result.get("intermediate_steps", [])

                    # Display Tool Execution Expander if tools were used
                    if steps:
                        with st.expander("🛠️ Agent Tool Execution Trace", expanded=True):
                            for action, output in steps:
                                tool_name = getattr(action, "tool", "Tool")
                                tool_input = getattr(action, "tool_input", "")
                                st.markdown(f"**Tool Invoked:** `{tool_name}`")
                                st.markdown(f"**Input:** `{tool_input}`")
                                st.text_area("Tool Output", value=str(output), height=140, disabled=True)

                    # Display Final Answer
                    st.markdown(answer)

                    # Append to Session State
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "intermediate_steps": steps
                    })

                except Exception as e:
                    error_msg = f"❌ Error running agent: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
