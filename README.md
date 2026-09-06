# Lab Compliance Agent

An Agentic AI system for querying laboratory SOPs, validating experimental parameters against compliance rules, and producing structured, audit-ready responses.

## 1. Problem Statement

The system acts as a laboratory compliance assistant that can:

- Answer questions using laboratory Standard Operating Procedures (SOPs)
- Retrieve relevant SOP sections using semantic search
- Validate numeric experimental readings against SOP-defined limits
- Provide traceable sources for every answer
- Record tool usage and audit events
- Return structured responses with confidence information

The implementation is intentionally scoped to a small SOP corpus while keeping the architecture extensible for additional agents, retrieval strategies, and LLM providers.

---

## 2. Architecture

```text
                         User
                           |
                           v
                    REST API /query
                           |
                           v
                  Agent Orchestrator
                           |
                           v
                    Compliance Agent
                       (LangGraph)
                           |
              +------------+------------+
              |                         |
              v                         v
        search_sops              validate_parameter
              |                         |
              v                         v
      Retrieval Service          Deterministic Python
              |
              v
          Retriever
              |
              v
        VectorStore
              |
              v
       Chroma Vector DB
              |
              v
          SOP Chunks
```

### LLM abstraction

```
                    Compliance Agent
                           |
                       ChatModel
                           |
                 +---------+---------+
                 |                   |
                 v                   v
             OpenAI              Mock LLM
           Chat Model         Deterministic Tests
```

The LLM is responsible for understanding the user request, deciding when tools are required, orchestrating the workflow, and composing the final response.

Numeric compliance decisions are delegated to a deterministic Python validation tool rather than relying on the LLM to perform the final comparison.

---

## 3. Project Structure

```
lab-compliance-agent/
├── app/
│   ├── agent/
│   │   ├── base.py
│   │   ├── compliance_agent.py
│   │   ├── orchestrator.py
│   │   └── state.py
│   │
│   ├── api/
│   │   └── schemas.py
│   │
│   ├── llm/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── mock_model.py
│   │   └── openai_model.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   ├── retrieval/
│   │   ├── base.py
│   │   ├── chunker.py
│   │   ├── chroma_store.py
│   │   ├── service.py
│   │   └── vector_store.py
│   │
│   └── tools/
│       ├── search_sops.py
│       └── validate_parameter.py
│
├── data/
│   └── sops/
│       ├── SOP-201.md
│       ├── SOP-305.md
│       └── SOP-108.md
│
├── tests/
│   ├── test_compliance_agent.py
│   ├── test_llm.py
│   ├── test_retrieval.py
│   └── test_tools.py
│
├── evaluate.py
├── ingest.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. Retrieval Design

The system uses a retrieval abstraction:

```
Retriever
   |
   +-- RetrievalService
           |
           v
      VectorStore
           |
           +-- ChromaVectorStore
```

This separates the agent from the underlying vector database.

The agent therefore does not directly depend on Chroma.

### Why Chroma?

The assessment contains a small SOP corpus, so a local persistent vector database provides a good balance between:

- Simple setup
- Semantic retrieval
- Persistence
- Low operational overhead
- Fast implementation within the assessment scope

For a larger production system, the `VectorStore` abstraction could be implemented using another vector database without changing the agent or tool interface.

### Metadata-aware retrieval

Each indexed chunk contains metadata including:

- SOP ID
- Section
- Revision
- Effective date
- Source document

When a query explicitly identifies an SOP, the retrieval tool applies deterministic SOP-ID filtering in addition to semantic retrieval.

This prevents semantically similar content from unrelated SOPs from being returned when the user has explicitly requested a specific SOP.

---

## 5. Agent Workflow

The compliance agent uses a LangGraph state machine.

```
START
  |
  v
Agent / LLM
  |
  +---- no tool call ----> END
  |
  +---- tool call -------> Tools
                              |
                              v
                            Agent
                              |
                              v
                             END
```

For a compliance query:

```
User:
"Is pH 7.52 compliant with SOP-201?"

        |
        v

search_sops
        |
        v

SOP-201 §4.2
pH: 7.35 - 7.45
        |
        v

validate_parameter
        |
        v

7.52 > 7.45
        |
        v

NON-COMPLIANT
        |
        v

Structured audit-ready response
```

---

## 6. Tools

### `search_sops`

Searches the SOP knowledge base and returns relevant SOP chunks.

The result includes:

- SOP ID
- Section
- Content
- Retrieval score
- Revision
- Effective date
- Source document

If an SOP ID is explicitly present in the query, results are filtered to that SOP.

### `validate_parameter`

Performs deterministic numeric validation.

Example:

```
parameter: pH
value: 7.52
lower_bound: 7.35
upper_bound: 7.45

Result:
NON-COMPLIANT
```

Boundary values are treated as compliant.

For example:

```
7.35 -> COMPLIANT
7.45 -> COMPLIANT
7.52 -> NON-COMPLIANT
```

The validation tool rejects invalid bounds rather than allowing an invalid range to silently produce a result.

---

## 7. Auditability

The API returns structured trace information.

Example:

```
{
  "sources": [
    {
      "sop_id": "SOP-201",
      "section": "4.2 Acceptance Criteria",
      "revision": 3,
      "effective_date": "2025-03-15"
    }
  ],
  "tool_calls": [
    {
      "tool_name": "search_sops"
    },
    {
      "tool_name": "validate_parameter"
    }
  ],
  "confidence": "HIGH",
  "audit_trace": [
    {
      "event": "retrieval",
      "tool": "search_sops"
    },
    {
      "event": "validation",
      "tool": "validate_parameter"
    }
  ]
}
```

The system deliberately records concise audit summaries rather than exposing hidden model chain-of-thought.

---

## 8. Confidence

Confidence is derived from available evidence:

EvidenceConfidenceNo retrieved SOP evidenceLOWSOP evidence retrievedMEDIUMSOP evidence + deterministic validationHIGH

This is an implementation-level confidence signal rather than a statistical probability.

---

## 9. LLM Provider Design

The LLM layer uses a provider abstraction.

Supported providers:

- OpenAI
- Mock

The OpenAI provider allows the application to use a real tool-calling LLM.

The Mock provider provides deterministic behavior for tests and evaluation, avoiding dependence on external API availability, network conditions, or API quota.

Configure the provider through `.env`:

```
LLM_PROVIDER=mock
LLM_MODEL=gpt-5.6-luna
OPENAI_API_KEY=
```

For OpenAI:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.6-luna
OPENAI_API_KEY=your-key
```

The application does not require Ollama or another local model runtime.

---

## 10. Evaluation

The project includes a repeatable evaluation script:

```
python evaluate.py
```

The evaluation currently covers:

1. Factual SOP retrieval
2. Non-compliant parameter validation
3. Upper-boundary validation
4. Lower-boundary validation
5. Unknown SOP / insufficient evidence handling

Expected output:

```
[PASS] ...
[PASS] ...
[PASS] ...
[PASS] ...
[PASS] ...

Evaluation result: 5/5 passed
```

The evaluation uses deterministic behavior so that results remain repeatable.

This is intentionally separate from live LLM evaluation.

---

## 11. Testing

Run the unit and integration tests with:

```
pytest -v
```

The test suite covers:

- Agent tool invocation
- Compliance validation
- Boundary conditions
- Audit trace generation
- Retrieval
- Tool behavior
- LLM abstraction

Warnings from third-party dependencies may appear during testing but do not affect the current test results.

---

## 12. Running the Application

### Install dependencies

```
pip install -r requirements.txt
```

### Configure environment

Copy:

```
.env.example
```

to:

```
.env
```

Then configure the desired LLM provider.

### Ingest SOPs

```
python ingest.py
```

This creates the local persistent Chroma index.

### Start the API

```
uvicorn app.main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

### Health check

```
GET /health
```

### Query

```
POST /query
```

Example request:

```
{
  "query": "Is a pH reading of 7.52 compliant with SOP-201?"
}
```

---

## 13. Production Considerations

### Tenant isolation

A production implementation should associate every document, retrieval request, and agent execution with a tenant identifier.

For example:

```
tenant_id
    |
    +-- retrieval filter
    +-- vector namespace
    +-- audit records
    +-- authorization
```

Tenant filtering should occur at the data-access layer rather than relying only on the LLM to respect tenant boundaries.

### Error handling

Production deployments should additionally handle:

- LLM provider failures
- Retrieval service failures
- Vector database failures
- Tool execution errors
- Invalid user input
- Timeouts
- Rate limits

Errors should be surfaced through controlled API responses and captured in observability infrastructure.

### Observability

A production deployment should capture:

- Request IDs
- Tenant IDs
- Agent execution duration
- Retrieval latency
- Retrieved document IDs
- Tool execution latency
- Tool failures
- LLM latency
- Token usage
- Model/provider information

Sensitive SOP content should not be indiscriminately logged.

### Security

Production deployments should also include:

- Authentication and authorization
- Input validation
- Rate limiting
- Secret management
- Encryption in transit and at rest
- Access-controlled audit logs

### Retrieval scalability

Chroma is appropriate for the assessment's small corpus.

At larger scale, the same `Retriever` and `VectorStore` interfaces could support a managed vector database or hybrid search backend.

---

## 14. Design Trade-offs

### Why deterministic validation instead of LLM reasoning?

Numeric compliance is a rules-based operation.

Delegating the final comparison to Python provides:

- Deterministic results
- Reproducibility
- Easier testing
- Lower hallucination risk
- Clear auditability

The LLM can identify the relevant parameter and SOP rule, but the validation tool performs the authoritative numeric comparison.

### Why use an agent?

The agent provides flexibility for multi-step workflows.

For example:

```
Retrieve SOP
    ↓
Interpret requirement
    ↓
Validate reading
    ↓
Compose response
```

This also allows additional compliance tools to be added later without rewriting the API layer.

### Why abstract retrieval?

The application should not be coupled to a particular vector database.

The current implementation can replace Chroma with another backend while preserving the agent and tool interfaces.

---

## 15. Current Scope

Implemented:

- SOP ingestion
- Markdown chunking
- Persistent vector retrieval
- Metadata-aware retrieval
- SOP-ID filtering
- Compliance validation
- LangGraph agent workflow
- LLM abstraction
- Mock LLM
- OpenAI integration
- Structured responses
- Audit trace
- Confidence scoring
- REST `/query` endpoint
- Health endpoint
- Automated evaluation
- Automated tests

Future extensions can include:

- Additional REST endpoints
- Expanded evaluation datasets
- Metadata/version selection
- Observability
- Tenant isolation
- Additional laboratory agents
- Streaming responses

