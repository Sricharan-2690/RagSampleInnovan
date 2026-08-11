"""
AGENT EXECUTOR MODULE
---------------------
Runs the LangChain Agent, manages chat history, extracts intermediate steps
(tool call logs and execution traces), and structures final responses with citations.
"""

from typing import Dict, Any, List
try:
    from langchain.agents import AgentExecutor
except (ImportError, AttributeError):
    from langchain_classic.agents import AgentExecutor
from agentic_rag.agent import create_rag_agent
from agentic_rag.tools import get_all_tools


class RAGAgentRunner:
    """
    Runner class that initializes and manages the AgentExecutor instance.
    """

    def __init__(self):
        self.tools = get_all_tools()
        self.agent, _ = create_rag_agent(self.tools)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=10
        )

    def run(self, user_input: str, chat_history: List = None) -> Dict[str, Any]:
        """
        Executes a user question through the Agentic RAG pipeline.

        Parameters
        ----------
        user_input : str
            The question or command from the user.
        chat_history : list, optional
            Prior chat history messages.

        Returns
        -------
        dict
            {
                "output": str,               # Final answer
                "intermediate_steps": list,  # Tool execution trace
                "tools_used": list[str]      # Names of tools invoked
            }
        """
        if chat_history is None:
            chat_history = []

        response = self.executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        output_text = response.get("output", "")
        steps = response.get("intermediate_steps", [])

        # Collect tool names invoked during execution
        tools_used = []
        for action, _ in steps:
            if hasattr(action, "tool") and action.tool not in tools_used:
                tools_used.append(action.tool)

        return {
            "output": output_text,
            "intermediate_steps": steps,
            "tools_used": tools_used
        }


# Singleton instance helper
_runner_instance = None

def get_agent_runner() -> RAGAgentRunner:
    """Returns or initializes the shared RAGAgentRunner instance."""
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = RAGAgentRunner()
    return _runner_instance
