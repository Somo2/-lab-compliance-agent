"""Abstraction for a vector storage implementation."""

from abc import ABC, abstractmethod

from app.models.schemas import DocumentChunk


class VectorStore(ABC):
    """Abstract interface for vector storage."""

    @abstractmethod
    def add_documents(self, documents: list[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        """Return similar document chunks with relevance scores."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all indexed documents."""
        raise NotImplementedError
