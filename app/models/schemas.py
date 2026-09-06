"""Shared domain models for retrieval, tools, and agent responses."""

from dataclasses import dataclass
from typing import Any


@dataclass
class DocumentChunk:
    """A chunk of content retrieved from an SOP."""
    sop_id: str
    section: str
    content: str
    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    """A retrieved SOP chunk with its relevance score."""
    chunk: DocumentChunk
    score: float


@dataclass
class ToolCall:
    """Record of a tool invocation."""
    tool_name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class AgentResponse:
    """Structured response returned by the compliance agent."""
    answer: str
    sources: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    confidence: str
    audit_trace: list[dict[str, Any]]
