"""Shared contracts for compliance agents."""

from abc import ABC, abstractmethod

from app.models.schemas import AgentResponse


class Agent(ABC):
    """Base interface for all platform agents."""

    @abstractmethod
    def run(self, query: str) -> AgentResponse:
        """Execute the agent against a user query."""
        raise NotImplementedError
