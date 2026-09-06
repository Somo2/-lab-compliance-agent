"""OpenAI implementation of the provider-agnostic chat-model contract."""

from typing import Any

from langchain_openai import ChatOpenAI

from app.llm.base import ChatModel


class OpenAIChatModel(ChatModel):
    """OpenAI-backed implementation of ChatModel."""

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = ChatOpenAI(model=model, temperature=temperature)

    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> Any:
        """Invoke the model, binding tools only for this request when supplied."""
        model = self.model.bind_tools(tools) if tools else self.model
        return model.invoke(messages)
