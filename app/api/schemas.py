from typing import Any

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    agent: str = "compliance"


class AgentQueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    confidence: str
    audit_trace: list[dict[str, Any]]
