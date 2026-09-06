from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_agents():
    response = client.get("/agents")

    assert response.status_code == 200

    data = response.json()

    agent_names = {
        agent["name"]
        for agent in data["agents"]
    }

    assert "compliance" in agent_names
    assert "sop_assistant" in agent_names


def test_list_sops():
    response = client.get("/sops")

    assert response.status_code == 200

    data = response.json()

    sop_ids = {
        item["sop_id"]
        for item in data
    }

    assert {"SOP-108", "SOP-201", "SOP-305"} <= sop_ids


def test_get_sop():
    response = client.get("/sops/SOP-201")

    assert response.status_code == 200

    data = response.json()

    assert data["sop_id"] == "SOP-201"
    assert len(data["sections"]) > 0


def test_get_sop_with_normalized_id():
    response = client.get("/sops/201")

    assert response.status_code == 200
    assert response.json()["sop_id"] == "SOP-201"


def test_get_unknown_sop():
    response = client.get("/sops/SOP-999")

    assert response.status_code == 404


def test_query_compliance_agent():
    response = client.post(
        "/query",
        json={
            "query": "Is pH 7.52 compliant with SOP-201?",
            "agent": "compliance",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "NON-COMPLIANT" in data["answer"].upper()
    assert "search_sops" in [
        call["tool_name"]
        for call in data["tool_calls"]
    ]
    assert "validate_parameter" in [
        call["tool_name"]
        for call in data["tool_calls"]
    ]
    assert data["confidence"] == "HIGH"


def test_query_sop_assistant_agent():
    response = client.post(
        "/query",
        json={
            "query": "What is the pH range in SOP-201?",
            "agent": "sop_assistant",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "7.35" in data["answer"]
    assert "7.45" in data["answer"]

    tool_names = [
        call["tool_name"]
        for call in data["tool_calls"]
    ]

    assert tool_names == ["search_sops"]
    assert data["confidence"] == "MEDIUM"


def test_compliance_agent_endpoint():
    response = client.post(
        "/agents/compliance/query",
        json={
            "query": "Is pH 7.52 compliant with SOP-201?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "NON-COMPLIANT" in data["answer"].upper()

    tool_names = [
        call["tool_name"]
        for call in data["tool_calls"]
    ]

    assert tool_names == [
        "search_sops",
        "validate_parameter",
    ]

    assert data["confidence"] == "HIGH"


def test_sop_assistant_agent_endpoint():
    response = client.post(
        "/agents/sop-assistant/query",
        json={
            "query": "What is the pH range in SOP-201?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "7.35" in data["answer"]
    assert "7.45" in data["answer"]

    tool_names = [
        call["tool_name"]
        for call in data["tool_calls"]
    ]

    assert tool_names == ["search_sops"]

    assert data["confidence"] == "MEDIUM"
