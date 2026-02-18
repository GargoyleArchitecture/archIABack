
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.graph.resources import llm
from src.graph.utils import _push_turn
from src.graph.consts import prompt_creator

def creator_node(state: GraphState) -> GraphState:
    """Preparation / validation node. Diagram generation is handled by diagram_orchestrator_node (DOT/Graphviz)."""
    user_q = state["userQuestion"]
    effective_q = state.get("localQuestion") or user_q

    prompt = f"""{prompt_creator}

User request:
{effective_q}

If an ASR is provided, ensure components and connectors explicitly support the Response and Response Measure.
"""
    _push_turn(state, role="system", name="creator_system", content=prompt)

    response = llm.invoke(prompt)
    content = getattr(response, "content", "")

    _push_turn(state, role="assistant", name="creator", content=content)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=content, name="creator")],
        "hasVisitedCreator": True
    }
