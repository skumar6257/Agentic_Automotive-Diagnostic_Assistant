from typing import TypedDict, List, Dict, Any

class DiagnosticState(TypedDict):
    """
    The shared state payload that passes through the LangGraph nodes.
    Every agent reads from and writes to this state.
    """

    # Inputs from the user
    vehicle_info: str
    dtc_code: str
    symptoms: str

    # RAG Context
    topology_nodes: List[Dict[str, Any]] # From Neo4j
    retrieved_manuals: List[str]         # From Qdrant

    # From Web Search Agent
    web_search_results: str

    # Agent Reasoning
    preliminary_plan: str
    critique_feedback: str
    verified_plan: str
    
    # Routing Flags
    safety_cleared: bool      # True = Input passed NeMo safety guardrails
    is_approved: bool         # True = Critique agent verified the repair plan
    requires_more_info: bool  # True = Critique found missing manuals/specs