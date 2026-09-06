from app.agent.compliance_agent import ComplianceAgent
from app.llm.mock_model import MockChatModel
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.service import RetrievalService
from app.tools.search_sops import create_search_sops_tool
from app.tools.validate_parameter import validate_parameter


def build_agent() -> ComplianceAgent:
    vector_store = ChromaVectorStore()
    retriever = RetrievalService(vector_store)
    search_sops = create_search_sops_tool(retriever)

    return ComplianceAgent(
        model=MockChatModel(),
        tools=[search_sops, validate_parameter],
    )


def run_case(name: str, query: str, expected: dict) -> bool:
    response = build_agent().run(query)
    actual_tools = [call.tool_name for call in response.tool_calls]
    answer = response.answer.upper()

    passed = all(
        expected_text.upper() in answer
        for expected_text in expected.get("answer_contains", [])
    )
    passed = passed and all(
        expected_tool in actual_tools
        for expected_tool in expected.get("tools", [])
    )

    if "confidence" in expected:
        passed = passed and response.confidence == expected["confidence"]

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")

    if not passed:
        print(f"  Answer: {response.answer}")
        print(f"  Tools: {actual_tools}")
        print(f"  Confidence: {response.confidence}")

    return passed


def main() -> None:
    cases = [
        {
            "name": "SOP factual retrieval",
            "query": "What is the acceptable pH range in SOP-201?",
            "expected": {
                "answer_contains": ["7.35", "7.45"],
                "tools": ["search_sops"],
                "confidence": "MEDIUM",
            },
        },
        {
            "name": "Non-compliant parameter validation",
            "query": "The pH reading is 7.52. Is it compliant according to SOP-201?",
            "expected": {
                "answer_contains": ["NON-COMPLIANT", "7.52", "7.45"],
                "tools": ["search_sops", "validate_parameter"],
                "confidence": "HIGH",
            },
        },
        {
            "name": "Upper boundary validation",
            "query": "The pH reading is exactly 7.45. Is it compliant according to SOP-201?",
            "expected": {
                "answer_contains": ["COMPLIANT", "7.45"],
                "tools": ["search_sops", "validate_parameter"],
                "confidence": "HIGH",
            },
        },
        {
            "name": "Lower boundary validation",
            "query": "The pH reading is exactly 7.35. Is it compliant according to SOP-201?",
            "expected": {
                "answer_contains": ["COMPLIANT", "7.35"],
                "tools": ["search_sops", "validate_parameter"],
                "confidence": "HIGH",
            },
        },
        {
            "name": "Unknown SOP handling",
            "query": "What is the acceptable pressure range in SOP-999?",
            "expected": {
                "answer_contains": ["COULD NOT"],
                "tools": [],
                "confidence": "LOW",
            },
        },
    ]

    passed = sum(
        run_case(
            name=case["name"],
            query=case["query"],
            expected=case["expected"],
        )
        for case in cases
    )
    total = len(cases)

    print(f"\nEvaluation result: {passed}/{total} passed")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
