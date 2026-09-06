"""Ingest the Markdown SOP corpus into the local Chroma store."""

from pathlib import Path

from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.chunker import SOPChunker


SOP_DIRECTORY = Path("data/sops")


def main() -> None:
    """Chunk all SOPs and upsert them into the default local Chroma store."""
    chunker = SOPChunker()
    vector_store = ChromaVectorStore()
    vector_store.clear()
    all_chunks = []
    for sop_file in SOP_DIRECTORY.glob("*.md"):
        chunks = chunker.chunk(
            sop_id=sop_file.stem,
            content=sop_file.read_text(encoding="utf-8"),
        )
        all_chunks.extend(chunks)
        print(f"{sop_file.name}: {len(chunks)} chunks")
    vector_store.add_documents(all_chunks)
    print(f"\nIndexed {len(all_chunks)} chunks.")


if __name__ == "__main__":
    main()
