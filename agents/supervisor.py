"""
Supervisor & Master Synthesizer Agents
1. Supervisor Agent: Acts as Shop Foreman, triaging faults to ELECTRICAL, MECHANICAL, or BOTH.
2. Master Synthesizer Agent: Combines domain reports into a cohesive, prioritized Master Repair Plan.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_core.prompts import ChatPromptTemplate
from agents.planner import get_llm, update_token_metrics
from agents.thought_parser import parse_thought_and_content
from infra.factory import factory

def supervisor_agent(state: dict) -> dict:
    """
    Supervisor Node: Triage the diagnostic trouble code and symptoms to determine
    which specialist sub-agents must be dispatched.
    """
    print("--- [Supervisor Agent] Evaluating Fault Triage & Dispatching Specialists ---")
    if "reasoning_traces" not in state or state["reasoning_traces"] is None:
        state["reasoning_traces"] = {}
    vehicle = state.get("vehicle_info", "")
    dtc = state.get("dtc_code", "")
    symptoms = state.get("symptoms", "")
    topology = state.get("topology_nodes", [])
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Automotive Shop Foreman and Diagnostic Supervisor.
Analyze the vehicle problem, DTC code, customer symptoms, and parts topology.
Determine which diagnostic disciplines are required to solve the issue:
1. 'ELECTRICAL': The fault is primarily an electrical, sensor circuit, voltage drop, wiring, ground, or network communication issue.
2. 'MECHANICAL': The fault is primarily mechanical wear, vacuum leak, compression loss, hydraulic pressure, valve timing, or physical component failure.
3. 'BOTH': The fault is complex, ambiguous, or could stem from either electrical or mechanical root causes (e.g., misfires, rough idle, stalling).
MANDATORY INSTRUCTIONS:
- First, write a private <thought>...</thought> block explaining your triage rationale.
- Second, after </thought>, output EXACTLY ONE WORD: ELECTRICAL, MECHANICAL, or BOTH."""),
        ("human", """Vehicle: {vehicle}
DTC: {dtc}
Symptoms: {symptoms}
Topology: {topology}
Triage Decision:""")
    ])
    llm = get_llm(tier="fast")
    chain = prompt | llm
    response = chain.invoke({
        "vehicle": vehicle,
        "dtc": dtc,
        "symptoms": symptoms,
        "topology": topology
    })
    # Track tokens
    llm_config = factory.get_llm_config(tier="fast")
    update_token_metrics(state, response, llm_config.get("model_name", "unknown"))
    # Extract thought and decision
    thought_trace, decision_raw = parse_thought_and_content(response.content)
    decision = decision_raw.strip().upper()
    if "ELECTRICAL" in decision and "BOTH" not in decision:
        final_decision = "ELECTRICAL"
    elif "MECHANICAL" in decision and "BOTH" not in decision:
        final_decision = "MECHANICAL"
    else:
        final_decision = "BOTH"
    state["supervisor_decision"] = final_decision
    state["reasoning_traces"]["supervisor"] = thought_trace
    print(f"-> [Supervisor] Triage Decision: {final_decision}")
    return state

def master_synthesizer_agent(state: dict) -> dict:
    """
    Synthesizer Node: Merges the specialist sub-agent reports into a single,
    unified, prioritized Master Diagnostic Plan.
    """
    print("--- [Master Synthesizer] Compiling Unified Diagnostic Plan ---")
    if "reasoning_traces" not in state or state["reasoning_traces"] is None:
        state["reasoning_traces"] = {}
    electrical_report = state.get("electrical_report") or "No electrical tests required for this triage."
    mechanical_report = state.get("mechanical_report") or "No mechanical tests required for this triage."
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Lead Master Automotive Diagnostic Architect.
Your task is to merge the findings from the Electrical Specialist and Mechanical Specialist into a single, cohesive, chronological Master Diagnostic Plan.
Structure the final plan into 3 logical phases:
1. Phase 1: Rapid / Non-Invasive Checks (Visual, OBD-II live data, Multimeter tests, Safety pre-checks).
2. Phase 2: In-Depth Component Diagnosis & Teardown (Compression, smoke test, pinout probing, torque specs).
3. Phase 3: Final Repair Verification (Part replacement, clearing DTCs, drive-cycle verification).
MANDATORY INSTRUCTIONS:
- First, write a private <thought>...</thought> block where you reconcile conflicting steps and discard redundant instructions.
- After </thought>, write the clear, numbered diagnostic plan incorporating torque specs and manual details.
- Incorporate any previous critique feedback if provided!"""),
        ("human", """Vehicle: {vehicle}
DTC: {dtc}
Customer Symptoms: {symptoms}
[Electrical Specialist Report]
{electrical}
[Mechanical Specialist Report]
{mechanical}
[OEM Manuals Context]
{manuals}
[Web Fallback Results]
{web_results}
[Previous Critique Feedback]
{feedback}
Draft the consolidated Master Diagnostic Plan:""")
    ])
    llm = get_llm(tier="reasoning")
    chain = prompt | llm
    response = chain.invoke({
        "vehicle": state.get("vehicle_info", ""),
        "dtc": state.get("dtc_code", ""),
        "symptoms": state.get("symptoms", ""),
        "electrical": electrical_report,
        "mechanical": mechanical_report,
        "manuals": state.get("retrieved_manuals", []),
        "web_results": state.get("web_search_results", "None"),
        "feedback": state.get("critique_feedback", "None")
    })
    # Track tokens
    llm_config = factory.get_llm_config(tier="reasoning")
    update_token_metrics(state, response, llm_config.get("model_name", "unknown"))
    # Extract thought and final preliminary plan
    thought_trace, consolidated_plan = parse_thought_and_content(response.content)
    state["reasoning_traces"]["synthesizer"] = thought_trace
    state["preliminary_plan"] = consolidated_plan
    print(f"-> [Master Synthesizer] Plan generated successfully ({len(consolidated_plan.split())} words).")
    return state