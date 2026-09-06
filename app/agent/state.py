from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ComplianceAgentState(TypedDict):
    """State maintained throughout the compliance workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    sources: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    audit_trace: list[dict[str, Any]]
    confidence: str
