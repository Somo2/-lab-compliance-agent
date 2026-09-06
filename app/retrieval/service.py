"""Application service for querying indexed SOP content."""

from app.models.schemas import RetrievalResult
from app.retrieval.base import Retriever
from app.retrieval.vector_store import VectorStore


class RetrievalService(Retriever):
    """Application-level retrieval service backed by a vector-store contract."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        results = self.vector_store.similarity_search(query=query, top_k=top_k)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in results]
