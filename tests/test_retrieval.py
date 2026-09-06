from pathlib import Path

from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.chunker import SOPChunker
from app.retrieval.service import RetrievalService


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
