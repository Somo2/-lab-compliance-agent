from app.llm.mock_model import MockChatModel


def test_mock_model_returns_configured_response():
    model = MockChatModel(
        responses=[
            {
                "content": "Mock response"
            }
        ]
    )

    response = model.invoke(
        [
            {
                "role": "user",
                "content": "Hello",
            }
        ]
    )

    assert response.content == "Mock response"
