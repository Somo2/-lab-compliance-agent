from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.sop_loader import SOPLoader


def main() -> None:
    loader = SOPLoader()
    vector_store = ChromaVectorStore()

    vector_store.clear()

    chunks = loader.load()
    vector_store.add_documents(chunks)

    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
