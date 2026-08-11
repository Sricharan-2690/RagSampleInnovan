"""
LANGCHAIN AGENT & PROMPT MODULE
--------------------------------
Defines the Agentic RAG system prompt, configures the Groq LLM model,
binds all custom tools (Qdrant search, calculator, web search),
and constructs the Tool-Calling / ReAct Agent.
"""

import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from agentic_rag.config import GROQ_API_KEY, DEFAULT_MODEL
from agentic_rag.tools import get_all_tools


SYSTEM_PROMPT = """You are an expert Agentic RAG assistant specializing in answering user queries using document knowledge bases, math tools, and web search.

You have access to the following tools:
1. `search_knowledge_base`: Search through uploaded PDF document chunks in Qdrant (hybrid dense+sparse search & reranking).
2. `calculator_tool`: Evaluate mathematical calculations, totals, percentages, and formulas.
3. `web_search_tool`: Search the public internet if the document does NOT contain the requested information.

INSTRUCTIONS & GUARDRAILS:
- Always decide whether to use a tool before answering.
- For questions about documents, company policies, reports, facts, or tables: FIRST call `search_knowledge_base`.
- If the question involves calculations, call `calculator_tool`.
- When answering from document context:
  * Ground your answer STRICTLY on the retrieved context.
  * Cite the exact page number(s) where facts were found, e.g., "(Page 12)".
  * If the document context does NOT contain the answer and web search is not requested, clearly state: "I couldn't find this information in the document."
- For casual greetings (e.g. "hi", "hello"), respond naturally without using tools.
- Never hallucinate facts or pretend to know document details that were not retrieved.
"""


def get_llm(model_name: str = DEFAULT_MODEL, temperature: float = 0.0):
    """
    Initializes and returns the ChatGroq LLM model instance.
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")

    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=temperature
    )


def create_rag_agent(tools: List = None):
    """
    Creates and configures the LangChain Tool-Calling Agent.

    Returns
    -------
    tuple: (agent, tools)
    """
    if tools is None:
        tools = get_all_tools()

    llm = get_llm()

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Import agent creator with compatibility fallback
    try:
        from langchain.agents import create_tool_calling_agent
    except (ImportError, AttributeError):
        from langchain_classic.agents import create_tool_calling_agent

    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return agent, tools
