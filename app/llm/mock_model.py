from typing import Any

from langchain_core.messages import AIMessage

from app.llm.base import ChatModel


class MockChatModel(ChatModel):
    """Deterministic model used for local development and evaluation."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = responses or []
        self.call_count = 0

    def bind_tools(self, tools: list[Any]) -> "MockChatModel":
        """Return a tool-aware mock model compatible with LangChain."""
        return self

    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> AIMessage:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1

            if isinstance(response, AIMessage):
                return response

            if isinstance(response, dict) and "content" in response:
                return AIMessage(content=str(response["content"]))

            return AIMessage(content=str(response))

        user_query = self._get_user_query(messages).lower()
        tool_names = self._get_tool_names(messages)

        if "conductivity" in user_query and "search_sops" in tool_names:
            return AIMessage(
                content=(
                    "According to SOP-201 §4.2, the acceptable conductivity "
                    "range is 1200–1800 µS/cm."
                )
            )

        if "sop-305" in user_query and "peak area" in user_query:
            if "search_sops" in tool_names:
                return AIMessage(
                    content=(
                        "According to SOP-305 §4.1, the maximum acceptable "
                        "peak area RSD is 2%."
                    )
                )

        if "temperature" in user_query and "search_sops" in tool_names:
            if "validate_parameter" not in tool_names:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "validate_parameter",
                            "args": {
                                "parameter": "Temperature",
                                "value": 30.0,
                                "lower_bound": 23.0,
                                "upper_bound": 27.0,
                            },
                            "id": "mock_validate_temperature",
                            "type": "tool_call",
                        }
                    ],
                )

            return AIMessage(
                content=(
                    "The buffer temperature of 30°C is NON-COMPLIANT. "
                    "According to SOP-201 §4.2, the acceptable temperature "
                    "range is 23.0–27.0°C. The reading is above the upper "
                    "limit of 27.0°C."
                )
            )

        if "validate_parameter" in tool_names:
            if "7.52" in user_query:
                return AIMessage(
                    content=(
                        "The pH reading of 7.52 is NON-COMPLIANT. "
                        "SOP-201 §4.2 specifies an acceptable pH range of "
                        "7.35–7.45. The reading is above the upper limit of 7.45."
                    )
                )

            if "7.45" in user_query:
                return AIMessage(
                    content=(
                        "The pH reading of 7.45 is COMPLIANT. "
                        "SOP-201 §4.2 specifies an acceptable pH range of "
                        "7.35–7.45, and the reading is exactly at the upper limit."
                    )
                )

            if "7.35" in user_query:
                return AIMessage(
                    content=(
                        "The pH reading of 7.35 is COMPLIANT. "
                        "SOP-201 §4.2 specifies an acceptable pH range of "
                        "7.35–7.45, and the reading is exactly at the lower limit."
                    )
                )

        if "search_sops" in tool_names:
            validation_values = {
                "7.52": ("mock_validate_1", 7.52),
                "7.45": ("mock_validate_boundary", 7.45),
                "7.35": ("mock_validate_lower_boundary", 7.35),
            }
            for value_text, (call_id, value) in validation_values.items():
                if value_text in user_query:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "validate_parameter",
                                "args": {
                                    "parameter": "pH",
                                    "value": value,
                                    "lower_bound": 7.35,
                                    "upper_bound": 7.45,
                                },
                                "id": call_id,
                                "type": "tool_call",
                            }
                        ],
                    )

            return AIMessage(
                content=(
                    "According to SOP-201 §4.2, the acceptable pH range "
                    "is 7.35–7.45."
                )
            )

        if "sop-201" in user_query and "ph" in user_query:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sops",
                        "args": {
                            "query": "SOP-201 acceptable pH range",
                            "top_k": 3,
                        },
                        "id": "mock_search_1",
                        "type": "tool_call",
                    }
                ],
            )

        if "sop-201" in user_query and "conductivity" in user_query:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sops",
                        "args": {
                            "query": "SOP-201 acceptable conductivity range",
                            "top_k": 3,
                        },
                        "id": "mock_search_conductivity",
                        "type": "tool_call",
                    }
                ],
            )

        if "sop-201" in user_query and "temperature" in user_query:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sops",
                        "args": {
                            "query": "SOP-201 acceptable temperature range",
                            "top_k": 3,
                        },
                        "id": "mock_search_temperature",
                        "type": "tool_call",
                    }
                ],
            )

        if "sop-305" in user_query and "peak area" in user_query:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sops",
                        "args": {
                            "query": "SOP-305 maximum peak area RSD",
                            "top_k": 3,
                        },
                        "id": "mock_search_sop305_peak_area",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content="Mock response: I could not determine a tool call for this query."
        )

    @staticmethod
    def _get_user_query(messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))

            if getattr(message, "type", None) == "human":
                return str(message.content)

        return ""

    @staticmethod
    def _get_tool_names(messages: list[dict[str, Any]]) -> list[str]:
        names = []

        for message in messages:
            if getattr(message, "type", None) == "ai":
                for tool_call in getattr(message, "tool_calls", []):
                    names.append(tool_call["name"])

        return names
