"""Contracts for SOP retrieval backends."""

from abc import ABC, abstractmethod

from app.models.schemas import RetrievalResult


class Retriever(ABC):
    """Abstract interface for SOP retrieval."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the most relevant SOP chunks."""
        raise NotImplementedError
