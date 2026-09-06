from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from app.api.schemas import QueryRequest, QueryResponse
from app.bootstrap import create_orchestrator

app = FastAPI(
    title="Lab Compliance Agent",
    description="Agentic AI system for laboratory SOP compliance.",
    version="1.0.0",
)

orchestrator = create_orchestrator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        response = orchestrator.run(
            agent_name=request.agent,
            query=request.query,
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
