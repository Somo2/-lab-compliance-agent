from rank_bm25 import BM25Okapi

from app.models.schemas import DocumentChunk, RetrievalResult
from app.retrieval.base import Retriever


class BM25Retriever(Retriever):
    """Lexical retriever over the indexed SOP chunks."""

    def __init__(self, documents: list[DocumentChunk]) -> None:
        self.documents = documents

        tokenized_documents = [
            self._tokenize(document.content)
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not self.documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            RetrievalResult(
                chunk=self.documents[index],
                score=float(scores[index]),
            )
            for index in ranked_indices[:top_k]
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
