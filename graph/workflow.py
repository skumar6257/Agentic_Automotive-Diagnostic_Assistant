'''
The LangGraph Orchestrator
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph.graph import StateGraph, END
from graph.state import DiagnosticState
from agents.retrievers import query_vector_agent, query_graph_agent
from agents.planner import diagnostic_planner_agent, critique_agent
from agents.search import web_search_agent
from agents.guardrail import safety_guardrail_agent
import time

def timed_node(node_name: str, node_func):
    """
    Decorator/Wrapper to measure exact millisecond execution time of any node.
    Updates state['node_latencies'] and state['total_latency_ms'] automatically.
    """
    def wrapper(state: DiagnosticState) -> DiagnosticState:
        # Initialize telemetry dicts if first node
        if "node_latencies" not in state or state["node_latencies"] is None:
            state["node_latencies"] = {}
        if "total_latency_ms" not in state or state["total_latency_ms"] is None:
            state["total_latency_ms"] = 0.0
        start_time = time.perf_counter()
        
        # Execute the actual agent node logic
        updated_state = node_func(state)
        
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        # Record latency
        updated_state["node_latencies"][node_name] = elapsed_ms
        updated_state["total_latency_ms"] = round(
            sum(updated_state["node_latencies"].values()), 2
        )
        
        print(f"⏱️ [{node_name}] Execution time: {elapsed_ms} ms")
        return updated_state
    return wrapper


def build_workflow():
    print("--- Building LangGraph State Machine ---")
    
    # 1. Initialize the Graph with our State schema
    workflow = StateGraph(DiagnosticState)

    # 2. Add all our Agent Nodes wrapped with the timing instrumentation
    workflow.add_node("safety_guardrail", timed_node("safety_guardrail", safety_guardrail_agent))
    workflow.add_node("graph_retriever", timed_node("graph_retriever", query_graph_agent))
    workflow.add_node("vector_retriever", timed_node("vector_retriever", query_vector_agent))
    workflow.add_node("planner", timed_node("planner", diagnostic_planner_agent))
    workflow.add_node("critique", timed_node("critique", critique_agent))
    workflow.add_node("web_search", timed_node("web_search", web_search_agent))

    # 3. Define the Edges (The Flow)
    workflow.set_entry_point("safety_guardrail")
    workflow.add_edge("graph_retriever","vector_retriever")
    workflow.add_edge("vector_retriever","planner")
    workflow.add_edge("planner","critique")
    # After a web search, always go back to the planner to rewrite the plan
    workflow.add_edge("web_search", "planner")

    def guardrail_router(state: DiagnosticState):
        """
        Decides whether to continue to the retriever (safe) or end (unsafe).
        """
        if state.get("safety_cleared"):
            print("-> Guardrail: Input is safe. Proceeding to retriever.")
            return "graph_retriever"  # Go to the retriever node
        else:
            print("-> Guardrail: Unsafe input detected. Halting workflow.")
            return "end"  # End the workflow

    workflow.add_conditional_edges(
        "safety_guardrail",
        guardrail_router,
        {
            "graph_retriever": "graph_retriever",
            "end": END
        }
    )
    
    # 5. Define the Conditional Edge (The loop!)
    def routing_logic(state: DiagnosticState):
        if state.get("is_approved"):
            print("-> ROUTING: Plan Approved by Master Tech. Ending workflow.")
            return "end"
        elif state.get("requires_more_info") and not state.get("web_search_results"):
            print("-> ROUTING: Missing info in manuals. Routing to Web Search Fallback.")
            return "web_search"
        else:
            print("-> ROUTING: Plan Rejected. Looping back to Planner.")
            return "replan"

    # Hook the conditional logic to the output of the Critique node
    workflow.add_conditional_edges(
        "critique",
        routing_logic,
        {
            "end": END,
            "web_search": "web_search",
            "replan": "planner"
        }
    )

    # Compile the graph into a runnable application
    app = workflow.compile()
    return app