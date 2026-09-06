"""Routing of requests to registered platform agents."""

from app.agent.base import Agent
from app.models.schemas import AgentResponse


class AgentOrchestrator:
    """Routes requests to registered agents; new types can be added by registration."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register_agent(self, name: str, agent: Agent) -> None:
        """Register an agent under a stable routing name."""
        self._agents[name] = agent

    def list_agents(self) -> list[str]:
        """Return the names of all registered agents."""
        return sorted(self._agents.keys())

    def get_agent(self, name: str) -> Agent:
        """Return a registered agent or raise a clear domain error."""
        try:
            return self._agents[name]
        except KeyError as exc:
            raise ValueError(f"Unknown agent: {name}") from exc

    def run(
        self,
        agent_name: str,
        query: str,
        request_id: str | None = None,
    ) -> AgentResponse:
        """Execute a query with the selected agent."""
        return self.get_agent(agent_name).run(
            query=query,
            request_id=request_id,
        )
