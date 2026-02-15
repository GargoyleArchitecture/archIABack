
from typing import Annotated, Literal, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

# ========== Schemas

class supervisorResponse(TypedDict):
    localQuestion: Annotated[str, ..., "What is the question for the worker node?"]
    nextNode: Literal[
        "investigator",
        "creator",
        "evaluator",
        "diagram_agent",
        "tactics",
        "asr",
        "style",
        "unifier",
    ]

supervisorSchema = {
    "title": "SupervisorResponse",
    "description": "Response from the supervisor indicating the next node and the setup question.",
    "type": "object",
    "properties": {
        "localQuestion": {
            "type": "string",
            "description": "What is the question for the worker node?"
        },
        "nextNode": {
            "type": "string",
            "description": "The next node to act.",
            "enum": [
                "investigator",
                "creator",
                "evaluator",
                "unifier",
                "asr",
                "diagram_agent",
                "tactics",
                "style"
            ]
        }
    },
    "required": ["localQuestion", "nextNode"]
}

class evaluatorResponse(TypedDict):
    positiveAspects: Annotated[str, ..., "What are the positive aspects of the user's idea?"]
    negativeAspects: Annotated[str, ..., "What are the negative aspects of the user's idea?"]
    suggestions: Annotated[str, ..., "What are the suggestions for improvement?"]

evaluatorSchema = {
    "title": "EvaluatorResponse",
    "description": "Response from the evaluator indicating the positive and negative aspects of the user's idea.",
    "type": "object",
    "properties": {
        "positiveAspects": {"type": "string"},
        "negativeAspects": {"type": "string"},
        "suggestions": {"type": "string"}
    },
    "required": ["positiveAspects", "negativeAspects", "suggestions"]
}

class investigatorResponse(TypedDict):
    definition: Annotated[str, ..., "What is the definition of the concept?"]
    useCases: Annotated[str, ..., "What are the use cases of the concept?"]
    examples: Annotated[str, ..., "What are the examples of the concept?"]

investigatorSchema = {
    "title": "InvestigatorResponse",
    "description": "Response from the investigator indicating the definition, use cases, and examples of the concept.",
    "type": "object",
    "properties": {
        "definition": {"type": "string"},
        "useCases": {"type": "string"},
        "examples": {"type": "string"}
    },
    "required": ["definition", "useCases", "examples"]
}

class ClassifyOut(TypedDict):
    language: Literal["en","es"]
    intent: Literal["greeting","smalltalk","architecture","diagram","asr","tactics","style","other"]
    use_rag: bool

# ========== Tactics Schemas

TACTIC_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "purpose": {"type": "string"},
        "rationale": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "tradeoffs": {"type": "array", "items": {"type": "string"}},
        "categories": {"type": "array", "items": {"type": "string"}},
        "traces_to_asr": {"type": "string"},
        "expected_effect": {"type": "string"},
        "success_probability": {"type": "number"},
        "rank": {"type": "integer"}
    },
    "required": ["name","rationale","categories","success_probability","rank"]
}

TACTICS_ARRAY_SCHEMA = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": TACTIC_ITEM_SCHEMA
}

# ========== Graph State

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    userQuestion: str
    localQuestion: str
    
    # Flags de visita
    hasVisitedInvestigator: bool
    hasVisitedCreator: bool
    hasVisitedEvaluator: bool
    hasVisitedASR: bool
    hasVisitedDiagram: bool
    
    nextNode: Literal["investigator", "creator", "evaluator", "diagram_agent", "tactics", "asr", "style", "unifier"]

    # planificación multi-intent por turno
    requested_nodes: list
    pending_nodes: list
    completed_nodes: list
    
    doc_only: bool
    doc_context: str

    imagePath1: str
    imagePath2: str

    endMessage: str
    mermaidCode: str

    diagram: dict

    # buffers / RAG / memoria liviana
    turn_messages: list
    retrieved_docs: list
    memory_text: str
    suggestions: list

    # memoria de conversación útil
    last_asr: str
    asr_sources_list: list

    # control de idioma/intención/forcing RAG
    language: Literal["en","es"]
    intent: Literal["general","greeting","smalltalk","architecture","diagram","asr","tactics","style"]
    force_rag: bool

    # etapa actual del pipeline ASR -> estilos -> tacticas -> despliegue
    arch_stage: str
    quality_attribute: str
    add_context: str
    tactics_list: list
    current_asr: str # ASR vigente
    tactics_struct: list # salida JSON parseada del tactics_node
    tactics_md: str # salida markdown del tactics_node

    style: str # estilo actual
    selected_style: str
    last_style: str

class AgentState(TypedDict):
    messages: list
    userQuestion: str
    localQuestion: str
    imagePath1: str
    imagePath2: str
