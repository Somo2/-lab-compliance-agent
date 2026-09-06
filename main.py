"""Application composition entry point."""

from app.agent.graph import build_graph
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.service import RetrievalService
from app.tools.search_sops import SearchSOPsTool
from app.tools.validate_parameter import ValidateParameterTool


def create_application() -> object:
    """Compose the default local SOP retrieval and compliance-agent workflow."""
    search_tool = SearchSOPsTool(RetrievalService(ChromaVectorStore()))
    return build_graph(search_tool, ValidateParameterTool())
