from pathlib import Path

from langchain_core.messages import AIMessage

from app.agent.compliance_agent import ComplianceAgent
from app.llm.mock_model import MockChatModel
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.chunker import SOPChunker
from app.retrieval.service import RetrievalService
from app.tools.search_sops import create_search_sops_tool
from app.tools.validate_parameter import validate_parameter


def populate_test_store(vector_store: ChromaVectorStore) -> None:
    chunker = SOPChunker()
    all_chunks = []

    for sop_file in Path("data/sops").glob("*.md"):
        content = sop_file.read_text(encoding="utf-8")
        all_chunks.extend(
            chunker.chunk(
                sop_id=sop_file.stem,
                content=content,
            )
        )

    vector_store.clear()
    vector_store.add_documents(all_chunks)


def test_compliance_agent_can_call_search_tool():
    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="sops",
    )

    populate_test_store(vector_store)
    retriever = RetrievalService(vector_store)
    search_tool = create_search_sops_tool(retriever)

    mock_model = MockChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sops",
                        "args": {
                            "query": "acceptable pH SOP-201",
                            "top_k": 3,
                        },
                        "id": "call_search_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "According to SOP-201 section 4.2, "
                    "the acceptable pH range is 7.35–7.45."
                ),
            ),
        ],
    )

    agent = ComplianceAgent(
        model=mock_model,
        tools=[
            search_tool,
            validate_parameter,
        ],
    )

    response = agent.run("What is the acceptable pH range for SOP-201?")

    assert "7.35" in response.answer
    assert "7.45" in response.answer
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "search_sops"
    assert len(response.sources) > 0


def test_compliance_agent_validates_non_compliant_ph():
    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="sops",
    )

    populate_test_store(vector_store)
    retriever = RetrievalService(vector_store)
    search_tool = create_search_sops_tool(retriever)

    mock_model = MockChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sops",
                        "args": {
                            "query": "SOP-201 acceptable pH range",
                            "top_k": 3,
                        },
                        "id": "call_search_2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "validate_parameter",
                        "args": {
                            "parameter": "pH",
                            "value": 7.52,
                            "lower_bound": 7.35,
                            "upper_bound": 7.45,
                        },
                        "id": "call_validate_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "The measured pH of 7.52 is NON-COMPLIANT. "
                    "SOP-201 section 4.2 specifies an acceptable "
                    "range of 7.35–7.45. A deviation should be "
                    "reported according to the applicable SOP."
                ),
            ),
        ],
    )

    agent = ComplianceAgent(
        model=mock_model,
        tools=[
            search_tool,
            validate_parameter,
        ],
    )

    response = agent.run(
        "The measured pH for SOP-201 is 7.52. Is it compliant?"
    )

    assert "NON-COMPLIANT" in response.answer
    assert "7.52" in response.answer
    assert response.confidence == "HIGH"
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].tool_name == "search_sops"
    assert response.tool_calls[1].tool_name == "validate_parameter"

    validation_result = response.tool_calls[1].result

    assert validation_result["compliant"] is False
    assert validation_result["lower_bound"] == 7.35
    assert validation_result["upper_bound"] == 7.45
    assert len(response.sources) > 0


def test_compliance_agent_accepts_boundary_value():
    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="sops",
    )

    populate_test_store(vector_store)
    retriever = RetrievalService(vector_store)
    search_tool = create_search_sops_tool(retriever)

    mock_model = MockChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sops",
                        "args": {
                            "query": "SOP-201 acceptable pH range",
                            "top_k": 3,
                        },
                        "id": "call_search_boundary",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "validate_parameter",
                        "args": {
                            "parameter": "pH",
                            "value": 7.45,
                            "lower_bound": 7.35,
                            "upper_bound": 7.45,
                        },
                        "id": "call_validate_boundary",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "A pH of 7.45 is COMPLIANT because it is "
                    "within the inclusive SOP-201 range of 7.35–7.45."
                ),
            ),
        ],
    )

    agent = ComplianceAgent(
        model=mock_model,
        tools=[
            search_tool,
            validate_parameter,
        ],
    )

    response = agent.run(
        "The measured pH for SOP-201 is 7.45. Is it compliant?"
    )

    assert "COMPLIANT" in response.answer
    assert response.confidence == "HIGH"

    validation_result = response.tool_calls[-1].result

    assert validation_result["compliant"] is True


def test_compliance_agent_returns_audit_trace():
    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="sops",
    )

    populate_test_store(vector_store)
    retriever = RetrievalService(vector_store)
    search_tool = create_search_sops_tool(retriever)

    mock_model = MockChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sops",
                        "args": {
                            "query": "SOP-201 pH acceptance criteria",
                            "top_k": 3,
                        },
                        "id": "audit_search_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "validate_parameter",
                        "args": {
                            "parameter": "pH",
                            "value": 7.52,
                            "lower_bound": 7.35,
                            "upper_bound": 7.45,
                        },
                        "id": "audit_validate_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="pH 7.52 is NON-COMPLIANT."),
        ],
    )

    agent = ComplianceAgent(
        model=mock_model,
        tools=[
            search_tool,
            validate_parameter,
        ],
    )

    response = agent.run("Is pH 7.52 compliant with SOP-201?")

    assert response.answer
    assert response.sources
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].tool_name == "search_sops"
    assert response.tool_calls[1].tool_name == "validate_parameter"
    assert response.tool_calls[1].arguments["value"] == 7.52
    assert response.tool_calls[1].result["compliant"] is False
    assert response.confidence == "HIGH"
