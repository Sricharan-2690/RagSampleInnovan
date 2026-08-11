"""
CONFIGURATION MODULE FOR AGENTIC RAG
--------------------------------------
Manages environment variables, Qdrant client connection parameters,
LLM defaults, and collection resolution helpers.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Disable Hugging Face symlinks & PyTorch Dynamo/Inductor JIT compilers on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# Load .env file from workspace root or Parsing-main directory
workspace_root = Path(__file__).resolve().parent.parent
env_paths = [
    workspace_root / ".env",
    workspace_root / "Parsing-main" / ".env"
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

# Qdrant Parameters
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# LLM Parameters
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

def get_qdrant_client() -> QdrantClient:
    """Returns a connected QdrantClient instance."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def list_qdrant_collections() -> list[str]:
    """Retrieves all collection names currently available in the Qdrant instance."""
    try:
        client = get_qdrant_client()
        collections_res = client.get_collections()
        return [col.name for col in collections_res.collections]
    except Exception as e:
        print(f"[Warning] Failed to list Qdrant collections: {e}")
        return []
