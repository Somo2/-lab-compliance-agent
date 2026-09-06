from app.agent.compliance_agent import ComplianceAgent
from app.agent.orchestrator import AgentOrchestrator
from app.llm.factory import create_chat_model
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.service import RetrievalService
from app.retrieval.sop_loader import SOPLoader
from app.tools.search_sops import create_search_sops_tool
from app.tools.validate_parameter import validate_parameter


def create_orchestrator() -> AgentOrchestrator:
    vector_store = ChromaVectorStore()
    vector_retriever = RetrievalService(vector_store)

    sop_loader = SOPLoader()
    documents = sop_loader.load()
    bm25_retriever = BM25Retriever(documents)

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    search_sops = create_search_sops_tool(hybrid_retriever)

    model = create_chat_model()

    compliance_agent = ComplianceAgent(
        model=model,
        tools=[
            search_sops,
            validate_parameter,
        ],
    )

    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(
        "compliance",
        compliance_agent,
    )

    return orchestrator
