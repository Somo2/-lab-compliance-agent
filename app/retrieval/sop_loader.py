from pathlib import Path

from app.models.schemas import DocumentChunk
from app.retrieval.chunker import SOPChunker


class SOPLoader:
    """Loads and chunks SOP Markdown documents."""

    def __init__(self, sop_directory: str | Path = "data/sops") -> None:
        self.sop_directory = Path(sop_directory)
        self.chunker = SOPChunker()

    def load(self) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []

        for sop_file in sorted(self.sop_directory.glob("*.md")):
            content = sop_file.read_text(encoding="utf-8")

            chunks = self.chunker.chunk(
                sop_id=sop_file.stem,
                content=content,
            )

            all_chunks.extend(chunks)

        return all_chunks
