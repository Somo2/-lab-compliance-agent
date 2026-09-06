"""Manual runner for the OpenAI-backed compliance agent."""

from dotenv import load_dotenv

from app.agent.compliance_agent import ComplianceAgent
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.service import RetrievalService
from app.tools.search_sops import SearchSOPsTool
from app.tools.validate_parameter import ValidateParameterTool


def main() -> None:
    """Run a manual SOP question through the agent."""
    load_dotenv()
    agent = ComplianceAgent(
        search_tool=SearchSOPsTool(RetrievalService(ChromaVectorStore())),
        validation_tool=ValidateParameterTool(),
    )
    result = agent.run("What is the acceptable pH range for Buffer Preparation per SOP-201?")
    print("\nAGENT RESPONSE\n")
    print(result)


if __name__ == "__main__":
    main()
