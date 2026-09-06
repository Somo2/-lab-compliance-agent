from app.observability.logging import generate_request_id


def test_generate_request_id_returns_unique_ids():
    first = generate_request_id()
    second = generate_request_id()

    assert first != second
    assert len(first) == 36
