from pathlib import Path

from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.chunker import SOPChunker
from app.retrieval.service import RetrievalService
from app.tools.search_sops import SearchSOPsTool
from app.tools.validate_parameter import ValidateParameterTool


def test_search_sops_tool(tmp_path: Path) -> None:
    """The tool exposes SOP retrieval as simple, traceable dictionaries."""
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_sops",
    )
    content = Path("data/sops/SOP-201.md").read_text(encoding="utf-8")
    store.add_documents(SOPChunker().chunk("SOP-201", content))
    tool = SearchSOPsTool(retriever=RetrievalService(vector_store=store))

    results = tool.run("acceptable pH range for SOP-201")

    assert results
    criteria = next(result for result in results if result["section"] == "4.2 Acceptance Criteria")
    assert criteria["sop_id"] == "SOP-201"
    assert "7.35" in criteria["content"]
    assert "7.45" in criteria["content"]
    assert {"sop_id", "section", "content", "score", "metadata"} <= criteria.keys()
    store.clear()


def test_validate_parameter_compliant() -> None:
    result = ValidateParameterTool().run("pH", 7.40, 7.35, 7.45)

    assert result.compliant is True
    assert result.value == 7.40


def test_validate_parameter_non_compliant_high() -> None:
    result = ValidateParameterTool().run("pH", 7.52, 7.35, 7.45)

    assert result.compliant is False
    assert result.value == 7.52
    assert result.upper_bound == 7.45


def test_validate_parameter_lower_boundary() -> None:
    result = ValidateParameterTool().run("pH", 7.35, 7.35, 7.45)

    assert result.compliant is True


def test_validate_parameter_upper_boundary() -> None:
    result = ValidateParameterTool().run("pH", 7.45, 7.35, 7.45)

    assert result.compliant is True


def test_validate_parameter_invalid_bounds() -> None:
    tool = ValidateParameterTool()

    try:
        tool.run("pH", 7.40, 7.50, 7.30)
        assert False, "Expected ValueError"
    except ValueError:
        pass
