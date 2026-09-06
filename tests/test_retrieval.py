from pathlib import Path

from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.chunker import SOPChunker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.service import RetrievalService
from app.models.schemas import DocumentChunk


def test_sop_retrieval(tmp_path: Path) -> None:
    """SOP-201's pH criteria are returned from a fresh local index."""
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_sops",
    )
    content = Path("data/sops/SOP-201.md").read_text(encoding="utf-8")
    store.add_documents(SOPChunker().chunk("SOP-201", content))
    retriever = RetrievalService(vector_store=store)

    results = retriever.search("What is the acceptable pH range for SOP-201?")

    assert results
    assert all(result.chunk.sop_id == "SOP-201" for result in results)
    assert any("7.35" in result.chunk.content for result in results)
    assert any("7.45" in result.chunk.content for result in results)
    store.clear()


def test_bm25_retrieves_matching_sop_content():
    documents = [
        DocumentChunk(
            sop_id="SOP-201",
            section="4.2 Acceptance Criteria",
            content="pH acceptable range is 7.35 - 7.45.",
            metadata={},
        ),
        DocumentChunk(
            sop_id="SOP-305",
            section="4.1 System Suitability",
            content="Peak area RSD must be less than or equal to 2%.",
            metadata={},
        ),
    ]

    retriever = BM25Retriever(documents)

    results = retriever.search(
        query="acceptable pH range",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.sop_id == "SOP-201"
    assert results[0].chunk.section == "4.2 Acceptance Criteria"
