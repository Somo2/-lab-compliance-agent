"""Composition of the default agent routing graph."""

from app.agent.compliance_agent import ComplianceAgent
from app.agent.orchestrator import AgentOrchestrator
from app.tools.search_sops import SearchSOPsTool
from app.tools.validate_parameter import ValidateParameterTool


def build_graph(
    search_tool: SearchSOPsTool,
    validation_tool: ValidateParameterTool,
) -> AgentOrchestrator:
    """Build the orchestrator with its injected compliance-agent dependencies."""
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent("compliance", ComplianceAgent(search_tool, validation_tool))
    return orchestrator
