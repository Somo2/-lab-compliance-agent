from typing import Any

from app.llm.base import ChatModel


class FallbackChatModel(ChatModel):
    """Use a primary model and fall back to a secondary model on failure."""

    def __init__(
        self,
        primary: ChatModel,
        fallback: ChatModel,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self._using_fallback = False
        self._tools: list[Any] | None = None

    def bind_tools(self, tools: list[Any]) -> "FallbackChatModel":
        """Return a model configured to use the supplied tools."""
        bound_model = FallbackChatModel(
            primary=self.primary,
            fallback=self.fallback,
        )
        bound_model._tools = tools
        return bound_model

    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> Any:
        active_tools = tools if tools is not None else self._tools

        if self._using_fallback:
            return self.fallback.invoke(messages, tools=active_tools)

        try:
            return self.primary.invoke(messages, tools=active_tools)
        except Exception as exc:
            print(
                "Primary LLM unavailable. "
                "Falling back to MockLLM. "
                f"Reason: {exc}"
            )

            self._using_fallback = True

            return self.fallback.invoke(messages, tools=active_tools)
