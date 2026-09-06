from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from app.api.schemas import AgentQueryRequest, QueryRequest, QueryResponse
from app.api.sops import router as sops_router
from app.bootstrap import create_orchestrator
from app.utils.normalization import normalize_query

app = FastAPI(
    title="Lab Compliance Agent",
    description="Agentic AI system for laboratory SOP compliance.",
    version="1.0.0",
)

app.include_router(sops_router)

orchestrator = create_orchestrator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents")
def list_agents() -> dict:
    """List agents available on the platform."""
    return {
        "agents": [
            {
                "name": name,
            }
            for name in orchestrator.list_agents()
        ]
    }


def _query_response(agent_name: str, query: str) -> QueryResponse:
    response = orchestrator.run(
        query=normalize_query(query),
        agent_name=agent_name,
    )

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        tool_calls=[
            asdict(tool_call)
            for tool_call in response.tool_calls
        ],
        confidence=response.confidence,
        audit_trace=response.audit_trace,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        normalized_query = normalize_query(request.query)
        response = orchestrator.run(
            agent_name=request.agent,
            query=normalized_query,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        tool_calls=[
            asdict(tool_call)
            for tool_call in response.tool_calls
        ],
        confidence=response.confidence,
        audit_trace=response.audit_trace,
    )


@app.post("/agents/compliance/query", response_model=QueryResponse)
def compliance_query(request: AgentQueryRequest) -> QueryResponse:
    return _query_response("compliance", request.query)


@app.post("/agents/sop-assistant/query", response_model=QueryResponse)
def sop_assistant_query(request: AgentQueryRequest) -> QueryResponse:
    return _query_response("sop_assistant", request.query)
