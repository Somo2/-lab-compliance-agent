"""Chroma-backed implementation of the vector-store contract."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.models.schemas import DocumentChunk
from app.retrieval.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """Chroma-backed implementation of the VectorStore interface."""

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "sops",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, documents: list[DocumentChunk]) -> None:
        """Upsert document chunks and their traceability metadata into Chroma."""
        if not documents:
            return

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for index, document in enumerate(documents):
            document_id = f"{document.sop_id}_{document.section}_{index}"
            ids.append(document_id)
            texts.append(document.content)
            metadatas.append(
                {
                    "sop_id": document.sop_id,
                    "section": document.section,
                    **document.metadata,
                }
            )

        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def similarity_search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        """Return similar chunks, converting Chroma distances to relevance scores."""
        result = self.collection.query(query_texts=[query], n_results=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        results: list[tuple[DocumentChunk, float]] = []
        for content, stored_metadata, distance in zip(documents, metadatas, distances):
            metadata = dict(stored_metadata or {})
            sop_id = str(metadata.pop("sop_id", "UNKNOWN"))
            section = str(metadata.pop("section", "UNKNOWN"))
            chunk = DocumentChunk(
                sop_id=sop_id,
                section=section,
                content=content,
                metadata=metadata,
            )
            results.append((chunk, 1.0 / (1.0 + float(distance))))
        return results

    def clear(self) -> None:
        """Remove all indexed content while keeping this store usable."""
        collection_name = self.collection.name
        self.client.delete_collection(name=collection_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)
