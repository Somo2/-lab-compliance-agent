from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph

from app.agent.state import ComplianceAgentState as AgentState
from app.models.schemas import AgentResponse, ToolCall


class SOPAssistantAgent:
    """Agent for answering factual questions from laboratory SOPs."""

    def __init__(self, model, tools):
        self.model = model
        self.tools = tools

        self.tool_map = {
            tool.name: tool
            for tool in tools
        }

        self.model_with_tools = model.bind_tools(tools)

        graph = StateGraph(AgentState)

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

    def _call_model(self, state: AgentState) -> dict:
        messages = [
            SystemMessage(
                content=(
                    "You are an SOP Assistant. "
                    "Answer questions using the laboratory SOP knowledge base. "
                    "Use search_sops for SOP procedures, requirements, "
                    "acceptance criteria, and factual information. "
                    "Do not invent information that is not supported by "
                    "retrieved SOP content."
                )
            ),
            *state["messages"],
        ]

        response = self.model_with_tools.invoke(messages)

        return {
            "messages": [response],
        }

    def _should_continue(self, state: AgentState) -> str:
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return "end"

    def _execute_tools(self, state: AgentState) -> dict:
        last_message = state["messages"][-1]

        tool_messages = []
        tool_calls = list(state.get("tool_calls", []))
        sources = list(state.get("sources", []))
        audit_trace = list(state.get("audit_trace", []))

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call["args"]

            tool = self.tool_map[tool_name]
            result = tool.invoke(arguments)

            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

            tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                )
            )

            audit_trace.append(
                {
                    "step": "tool_execution",
                    "tool": tool_name,
                    "arguments": arguments,
                }
            )

            if tool_name == "search_sops" and isinstance(result, list):
                for item in result:
                    sources.append(
                        {
                            "sop_id": item.get("sop_id"),
                            "section": item.get("section"),
                            "revision": item.get("metadata", {}).get("revision"),
                            "effective_date": item.get("metadata", {}).get(
                                "effective_date"
                            ),
                        }
                    )

        return {
            "messages": tool_messages,
            "tool_calls": tool_calls,
            "sources": sources,
            "audit_trace": audit_trace,
        }

    def run(self, query: str) -> AgentResponse:
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "tool_calls": [],
            "sources": [],
            "audit_trace": [],
        }

        final_state = self.graph.invoke(initial_state)

        answer = ""

        for message in reversed(final_state["messages"]):
            if isinstance(message, AIMessage) and message.content:
                answer = message.content
                break

        sources = final_state.get("sources", [])
        confidence = "MEDIUM" if sources else "LOW"

        return AgentResponse(
            answer=answer,
            sources=sources,
            tool_calls=final_state.get("tool_calls", []),
            confidence=confidence,
            audit_trace=final_state.get("audit_trace", []),
        )
