import os

from app.llm.base import ChatModel
from app.llm.mock_model import MockChatModel
from app.llm.openai_model import OpenAIChatModel


def create_chat_model() -> ChatModel:
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    model = os.getenv("LLM_MODEL", "gpt-5.6-luna")

    if provider == "openai":
        return OpenAIChatModel(model=model)

    if provider == "mock":
        return MockChatModel()

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. "
        "Supported providers: openai, mock."
    )
