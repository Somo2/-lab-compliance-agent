"""Provider-agnostic chat-model contract."""

from abc import ABC, abstractmethod
from typing import Any


class ChatModel(ABC):
    """Provider-agnostic interface for chat models."""

    @abstractmethod
    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> Any:
        """Generate a response from the model."""
        raise NotImplementedError
