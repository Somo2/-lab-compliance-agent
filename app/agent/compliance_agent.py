from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from app.agent.base import Agent
from app.agent.state import ComplianceAgentState
from app.llm.base import ChatModel
from app.models.schemas import AgentResponse, ToolCall

SYSTEM_PROMPT = """You are a laboratory compliance assistant.

Your job is to answer questions using the provided laboratory SOP
knowledge base.

Rules:
1. Use search_sops when you need information from an SOP.
2. Use validate_parameter for numeric compliance validation.
3. Never decide numeric compliance yourself when validation can be
   performed by the validation tool.
4. Base answers only on retrieved SOP information.
5. Clearly identify the SOP and section supporting your answer.
6. If the available information is insufficient, say so.
"""


class ComplianceAgent(Agent):
    """LangGraph-based laboratory compliance agent."""

    def __init__(
        self,
        model: ChatModel,
        tools: list[BaseTool],
    ) -> None:
        self.model = model
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}

        graph = StateGraph(ComplianceAgentState)

        graph.add_node("agent", self._call_model)
        graph.add_node("tools", self._execute_tools)

        graph.add_edge(START, "agent")

        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )

        graph.add_edge("tools", "agent")

        self.graph = graph.compile()

    def _call_model(
        self,
        state: ComplianceAgentState,
    ) -> dict[str, Any]:
        response = self.model.invoke(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]
            + state["messages"],
            tools=self.tools,
        )

        return {
            "messages": [response],
        }

    def _should_continue(
        self,
        state: ComplianceAgentState,
    ) -> str:
        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"

        return "end"

    def _execute_tools(
        self,
        state: ComplianceAgentState,
    ) -> dict[str, Any]:
        last_message = state["messages"][-1]

        tool_messages = []
        tool_calls = list(state.get("tool_calls", []))
        sources = list(state.get("sources", []))
        audit_trace = list(state.get("audit_trace", []))

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tool = self.tools_by_name.get(tool_name)

            if tool is None:
                raise ValueError(f"Unknown tool requested: {tool_name}")

            result = tool.invoke(tool_args)

            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

            tool_calls.append(
                {
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "result": result,
                }
            )

            if tool_name == "search_sops":
                evidence_count = len(result) if isinstance(result, list) else 0
                sections = []

                if isinstance(result, list):
                    for item in result:
                        source = {
                            "sop_id": item.get("sop_id"),
                            "section": item.get("section"),
                            "revision": item.get("metadata", {}).get("revision"),
                            "effective_date": item.get("metadata", {}).get(
                                "effective_date"
                            ),
                        }

                        if source not in sources:
                            sources.append(source)

                        sop_id = item.get("sop_id")
                        section = item.get("section")
                        if sop_id and section:
                            sections.append(f"{sop_id} §{section}")

                audit_trace.append(
                    {
                        "event": "retrieval",
                        "tool": tool_name,
                        "summary": (
                            f"Retrieved {', '.join(sections)}"
                            if sections
                            else "No SOP evidence retrieved."
                        ),
                        "evidence_count": evidence_count,
                    }
                )

            elif tool_name == "validate_parameter":
                compliant = (
                    result.get("compliant")
                    if isinstance(result, dict)
                    else None
                )
                parameter = (
                    result.get("parameter")
                    if isinstance(result, dict)
                    else tool_args.get("parameter")
                )
                value = (
                    result.get("value")
                    if isinstance(result, dict)
                    else tool_args.get("value")
                )
                lower_bound = (
                    result.get("lower_bound")
                    if isinstance(result, dict)
                    else tool_args.get("lower_bound")
                )
                upper_bound = (
                    result.get("upper_bound")
                    if isinstance(result, dict)
                    else tool_args.get("upper_bound")
                )
                range_text = f"{lower_bound}–{upper_bound}"

                audit_trace.append(
                    {
                        "event": "validation",
                        "tool": tool_name,
                        "summary": (
                            f"Validated {parameter} {value} "
                            f"against range {range_text}"
                        ),
                        "result": (
                            "COMPLIANT"
                            if compliant is True
                            else "NON-COMPLIANT"
                            if compliant is False
                            else "UNKNOWN"
                        ),
                    }
                )

        return {
            "messages": tool_messages,
            "tool_calls": tool_calls,
            "sources": sources,
            "audit_trace": audit_trace,
        }

    @staticmethod
    def _calculate_confidence(
        sources: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
    ) -> str:
        """Determine confidence from available evidence."""
        has_sources = len(sources) > 0

        has_validation = any(
            call["tool_name"] == "validate_parameter"
            for call in tool_calls
        )

        if has_sources and has_validation:
            return "HIGH"

        if has_sources:
            return "MEDIUM"

        return "LOW"

    def run(self, query: str) -> AgentResponse:
        initial_state: ComplianceAgentState = {
            "messages": [
                HumanMessage(content=query),
            ],
            "sources": [],
            "tool_calls": [],
            "audit_trace": [],
            "confidence": "MEDIUM",
        }

        result = self.graph.invoke(initial_state)

        final_message = result["messages"][-1]
        sources = result.get("sources", [])
        tool_calls = result.get("tool_calls", [])
        audit_trace = result.get("audit_trace", [])

        confidence = self._calculate_confidence(
            sources=sources,
            tool_calls=tool_calls,
        )

        return AgentResponse(
            answer=final_message.content,
            sources=sources,
            tool_calls=[
                ToolCall(
                    tool_name=call["tool_name"],
                    arguments=call["arguments"],
                    result=call["result"],
                )
                for call in tool_calls
            ],
            confidence=confidence,
            audit_trace=audit_trace,
        )
