from unittest.mock import Mock

from app.llm.fallback_model import FallbackChatModel


def test_fallback_uses_secondary_model_when_primary_fails():
    primary = Mock()
    fallback = Mock()

    primary.invoke.side_effect = RuntimeError("Primary unavailable")
    fallback.invoke.return_value = "fallback response"

    model = FallbackChatModel(
        primary=primary,
        fallback=fallback,
    )

    result = model.invoke(
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result == "fallback response"

    primary.invoke.assert_called_once()
    fallback.invoke.assert_called_once()


def test_fallback_stays_on_secondary_after_failure():
    primary = Mock()
    fallback = Mock()

    primary.invoke.side_effect = RuntimeError("Primary unavailable")
    fallback.invoke.return_value = "fallback response"

    model = FallbackChatModel(
        primary=primary,
        fallback=fallback,
    )

    first_result = model.invoke(
        messages=[{"role": "user", "content": "First"}],
    )

    second_result = model.invoke(
        messages=[{"role": "user", "content": "Second"}],
    )

    assert first_result == "fallback response"
    assert second_result == "fallback response"

    assert primary.invoke.call_count == 1
    assert fallback.invoke.call_count == 2
