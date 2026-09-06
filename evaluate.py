from app.agent.compliance_agent import ComplianceAgent
from app.llm.mock_model import MockChatModel
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.service import RetrievalService
from app.retrieval.sop_loader import SOPLoader
from app.tools.search_sops import create_search_sops_tool
from app.tools.validate_parameter import validate_parameter


def build_agent() -> ComplianceAgent:
    vector_store = ChromaVectorStore()
    vector_retriever = RetrievalService(vector_store)

    documents = SOPLoader().load()
    bm25_retriever = BM25Retriever(documents)

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    search_sops = create_search_sops_tool(hybrid_retriever)

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

    if "tool_sequence" in expected:
        passed = passed and actual_tools == expected["tool_sequence"]

    if "confidence" in expected:
        passed = passed and response.confidence == expected["confidence"]

    if "source" in expected:
        expected_source = expected["source"]
        passed = passed and any(
            source.get("sop_id") == expected_source["sop_id"]
            and source.get("section") == expected_source["section"]
            for source in response.sources
        )

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")

    if not passed:
        print(f"  Answer: {response.answer}")
        print(f"  Tools: {actual_tools}")
        print(f"  Sources: {response.sources}")
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
                "source": {
                    "sop_id": "SOP-201",
                    "section": "4.2 Acceptance Criteria",
                },
            },
        },
        {
            "name": "Non-compliant parameter validation",
            "query": "The pH reading is 7.52. Is it compliant according to SOP-201?",
            "expected": {
                "answer_contains": ["NON-COMPLIANT", "7.52", "7.45"],
                "tools": ["search_sops", "validate_parameter"],
                "tool_sequence": ["search_sops", "validate_parameter"],
                "confidence": "HIGH",
                "source": {
                    "sop_id": "SOP-201",
                    "section": "4.2 Acceptance Criteria",
                },
            },
        },
        {
            "name": "Upper boundary validation",
            "query": "The pH reading is exactly 7.45. Is it compliant according to SOP-201?",
            "expected": {
                "answer_contains": ["COMPLIANT", "7.45"],
                "tools": ["search_sops", "validate_parameter"],
                "tool_sequence": ["search_sops", "validate_parameter"],
                "confidence": "HIGH",
                "source": {
                    "sop_id": "SOP-201",
                    "section": "4.2 Acceptance Criteria",
                },
            },
        },
        {
            "name": "Lower boundary validation",
            "query": "The pH reading is exactly 7.35. Is it compliant according to SOP-201?",
            "expected": {
                "answer_contains": ["COMPLIANT", "7.35"],
                "tools": ["search_sops", "validate_parameter"],
                "tool_sequence": ["search_sops", "validate_parameter"],
                "confidence": "HIGH",
                "source": {
                    "sop_id": "SOP-201",
                    "section": "4.2 Acceptance Criteria",
                },
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
        {
            "name": "Conductivity retrieval",
            "query": "What is the acceptable conductivity range in SOP-201?",
            "expected": {
                "answer_contains": ["1200", "1800"],
                "tools": ["search_sops"],
                "confidence": "MEDIUM",
                "source": {
                    "sop_id": "SOP-201",
                    "section": "4.2 Acceptance Criteria",
                },
            },
        },
        {
            "name": "Non-compliant temperature validation",
            "query": (
                "The buffer temperature is 30 degrees C. Is it compliant "
                "according to SOP-201?"
            ),
            "expected": {
                "answer_contains": ["NON-COMPLIANT", "30", "27"],
                "tools": ["search_sops", "validate_parameter"],
                "tool_sequence": ["search_sops", "validate_parameter"],
                "confidence": "HIGH",
                "source": {
                    "sop_id": "SOP-201",
                    "section": "4.2 Acceptance Criteria",
                },
            },
        },
        {
            "name": "HPLC system suitability retrieval",
            "query": "What is the maximum peak area RSD in SOP-305?",
            "expected": {
                "answer_contains": ["2%"],
                "tools": ["search_sops"],
                "confidence": "MEDIUM",
                "source": {
                    "sop_id": "SOP-305",
                    "section": "3.2 Acceptance Criteria",
                },
            },
        },
    ]

    passed = 0
    failed_cases = []

    for case in cases:
        if run_case(case["name"], case["query"], case["expected"]):
            passed += 1
        else:
            failed_cases.append(case["name"])

    total = len(cases)
    failed = total - passed

    print("\nEvaluation Summary")
    print("===================")
    print(f"Total cases: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed_cases:
        print("\nFailed cases:")
        for name in failed_cases:
            print(f"- {name}")

    print("\nCoverage:")
    print("- SOP factual retrieval")
    print("- Parameter compliance validation")
    print("- Boundary-value validation")
    print("- Unknown SOP / missing-knowledge handling")
    print("- Multi-step tool sequencing")
    print("- Multiple SOP coverage")


if __name__ == "__main__":
    main()
