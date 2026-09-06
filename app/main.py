from dataclasses import asdict
import uuid

from fastapi import FastAPI, HTTPException, Request

from app.api.schemas import AgentQueryRequest, QueryRequest, QueryResponse
from app.api.sops import router as sops_router
from app.bootstrap import create_orchestrator
from app.observability.logging import configure_logging, log_event
from app.utils.normalization import normalize_query

app = FastAPI(
    title="Lab Compliance Agent",
    description="Agentic AI system for laboratory SOP compliance.",
    version="1.0.0",
)

configure_logging()

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
def query(
    request: QueryRequest,
    http_request: Request,
) -> QueryResponse:
    request_id = str(uuid.uuid4())

    log_event(
        "api.request.started",
        request_id=request_id,
        endpoint="/query",
        method="POST",
        agent=request.agent,
    )

    try:
        normalized_query = normalize_query(request.query)
        response = orchestrator.run(
            agent_name=request.agent,
            query=normalized_query,
            request_id=request_id,
        )
    except Exception as exc:
        log_event(
            "api.request.failed",
            request_id=request_id,
            endpoint="/query",
            method="POST",
            agent=request.agent,
            error_type=type(exc).__name__,
            error=str(exc),
        )

        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        raise

    log_event(
        "api.request.completed",
        request_id=request_id,
        endpoint="/query",
        method="POST",
        agent=request.agent,
        confidence=response.confidence,
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


@app.post("/agents/compliance/query", response_model=QueryResponse)
def compliance_query(request: AgentQueryRequest) -> QueryResponse:
    return _query_response("compliance", request.query)


@app.post("/agents/sop-assistant/query", response_model=QueryResponse)
def sop_assistant_query(request: AgentQueryRequest) -> QueryResponse:
    return _query_response("sop_assistant", request.query)
