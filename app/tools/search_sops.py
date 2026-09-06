"""Serialization-friendly SOP search tool for use by an agent."""

import re
from typing import Any

from langchain_core.tools import StructuredTool

from app.retrieval.base import Retriever


class SearchSOPsTool:
    """Search the SOP knowledge base without coupling to an LLM framework."""

    name = "search_sops"

    description = (
        "Search the laboratory SOP knowledge base and return relevant SOP "
        "sections. Use this tool when the user asks about procedures, "
        "acceptance criteria, limits, requirements, or compliance rules."
    )

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def run(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return relevant SOP chunks as serialization-friendly dictionaries."""
        results = self.retriever.search(query=query, top_k=top_k)

        sop_match = re.search(r"SOP-\d+", query, re.IGNORECASE)
        if sop_match:
            requested_sop = sop_match.group(0).upper()
            results = [
                result
                for result in results
                if result.chunk.sop_id.upper() == requested_sop
            ]

        return [
            {
                "sop_id": result.chunk.sop_id,
                "section": result.chunk.section,
                "content": result.chunk.content,
                "score": round(result.score, 4),
                "metadata": result.chunk.metadata,
            }
            for result in results
        ]


def create_search_sops_tool(retriever: Retriever) -> StructuredTool:
    """Create a LangChain tool adapter for SOP search."""
    search_tool = SearchSOPsTool(retriever)
    return StructuredTool.from_function(
        func=search_tool.run,
        name=search_tool.name,
        description=search_tool.description,
    )
