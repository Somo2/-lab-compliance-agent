from app.models.schemas import RetrievalResult
from app.retrieval.base import Retriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.service import RetrievalService


class HybridRetriever(Retriever):
    """Combines semantic vector retrieval and BM25 lexical retrieval."""

    def __init__(
        self,
        vector_retriever: RetrievalService,
        bm25_retriever: BM25Retriever,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        vector_results = self.vector_retriever.search(
            query=query,
            top_k=top_k,
        )

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=top_k,
        )

        return self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=top_k,
        )

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k: int,
        k: int = 60,
    ) -> list[RetrievalResult]:
        """Fuse rankings using Reciprocal Rank Fusion (RRF)."""

        fused_scores: dict[str, float] = {}
        chunks: dict[str, RetrievalResult] = {}

        for results in (vector_results, bm25_results):
            for rank, result in enumerate(results, start=1):
                document_key = (
                    f"{result.chunk.sop_id}:"
                    f"{result.chunk.section}"
                )

                fused_scores[document_key] = (
                    fused_scores.get(document_key, 0.0)
                    + 1.0 / (k + rank)
                )

                chunks[document_key] = result

        ranked_keys = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        # Score represents the fused RRF ranking score,
        # not a normalized similarity probability.
        return [
            RetrievalResult(
                chunk=chunks[key].chunk,
                score=fused_scores[key],
            )
            for key in ranked_keys[:top_k]
        ]
