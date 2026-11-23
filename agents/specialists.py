"""
Specialized Diagnostic Sub-Agents
1. Electrical Specialist Agent: Wiring harnesses, voltages, grounds, sensor resistances, CAN bus.
2. Mechanical Specialist Agent: Compression, vacuum leaks, fuel pressure, physical wear, torque specs.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import ChatPromptTemplate
from agents.planner import get_llm, update_token_metrics
from agents.thought_parser import parse_thought_and_content
from infra.factory import factory

def electrical_specialist_agent(state: dict) -> dict:
    """
    Sub-Agent Node: Focuses strictly on electrical diagnostics, sensors, 
    reference voltages, grounds, and signal patterns.
    """
    print("--- [Electrical Specialist] Analyzing Sensor, Wiring & Voltage Context ---")

    if "reasoning_traces" not in state or state["reasoning_traces"] is None:
        state["reasoning_traces"] = {}

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Automotive Electrical Diagnostic Specialist.
        Your expertise covers: sensor resistances, 5V reference circuits, ground voltage drops (<0.1V), wiring harnesses, ECU pinouts, and CAN/LIN bus communications.
        MANDATORY INSTRUCTIONS:
        1. Begin your response with a private <thought>...</thought> block where you reason step-by-step:
        - Identify electrical sensors/actuators related to this DTC.
        - Plan specific multimeter (DMM) and oscilloscope measurement points (expected Volts, Ohms, Amps).
        - Filter out purely mechanical steps.
        2. After the </thought> tag, provide a concise, numbered electrical diagnostic procedure based strictly on the provided manuals and topology.
        3. Incorporate any previous critique feedback if provided."""),
                ("human", """Vehicle: {vehicle}
        DTC Trouble Code: {dtc}
        Customer Symptoms: {symptoms}
        Neo4j Parts Topology:
        {topology}
        OEM Workshop Manuals:
        {manuals}
        Previous Critique Feedback (if any):
        {feedback}
        Execute electrical diagnostic investigation:""")
    ])
    llm = get_llm(tier="reasoning")
    chain = prompt | llm
    response = chain.invoke({
        "vehicle": state.get("vehicle_info", ""),
        "dtc": state.get("dtc_code", ""),
        "symptoms": state.get("symptoms", ""),
        "topology": state.get("topology_nodes", []),
        "manuals": state.get("retrieved_manuals", []),
        "feedback": state.get("critique_feedback", "None")
    })
    # Track tokens
    llm_config = factory.get_llm_config(tier="reasoning")
    update_token_metrics(state, response, llm_config.get("model_name", "unknown"))
    # Extract <thought> reasoning block
    thought_trace, report = parse_thought_and_content(response.content)
    state["reasoning_traces"]["electrical"] = thought_trace
    state["electrical_report"] = report
    print(f"-> [Electrical Specialist] Completed electrical evaluation with {len(report.split())} words.")
    return state

def mechanical_specialist_agent(state: dict) -> dict:
    """
    Sub-Agent Node: Focuses strictly on mechanical wear, fuel rail pressure,
    vacuum leaks, cylinder compression, and physical torque specifications.
    """
    print("--- [Mechanical Specialist] Analyzing Powertrain, Vacuum & Torque Context ---")
    if "reasoning_traces" not in state or state["reasoning_traces"] is None:
        state["reasoning_traces"] = {}
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Master Powertrain & Mechanical Diagnostic Specialist.
        Your expertise covers: cylinder compression tests, vacuum leak smoke testing, fuel rail pressure, mechanical clearances, valve timing, and precise torque specifications.
        MANDATORY INSTRUCTIONS:
        1. Begin your response with a private <thought>...</thought> block where you reason step-by-step:
        - Identify mechanical failure modes that could trigger this DTC.
        - Review manual chunks for exact torque values (ft-lbs or Nm), pressure thresholds (PSI/bar), and mechanical inspection steps.
        - Filter out purely electrical wiring steps.
        2. After the </thought> tag, provide a concise, numbered mechanical inspection and repair procedure based strictly on the provided manuals and topology.
        3. Incorporate any previous critique feedback if provided."""),
                ("human", """Vehicle: {vehicle}
        DTC Trouble Code: {dtc}
        Customer Symptoms: {symptoms}
        Neo4j Parts Topology:
        {topology}
        OEM Workshop Manuals:
        {manuals}
        Previous Critique Feedback (if any):
        {feedback}
        Execute mechanical diagnostic investigation:""")
    ])
    llm = get_llm(tier="reasoning")
    chain = prompt | llm
    response = chain.invoke({
        "vehicle": state.get("vehicle_info", ""),
        "dtc": state.get("dtc_code", ""),
        "symptoms": state.get("symptoms", ""),
        "topology": state.get("topology_nodes", []),
        "manuals": state.get("retrieved_manuals", []),
        "feedback": state.get("critique_feedback", "None")
    })
    # Track tokens
    llm_config = factory.get_llm_config(tier="reasoning")
    update_token_metrics(state, response, llm_config.get("model_name", "unknown"))
    # Extract <thought> reasoning block
    thought_trace, report = parse_thought_and_content(response.content)
    state["reasoning_traces"]["mechanical"] = thought_trace
    state["mechanical_report"] = report
    print(f"-> [Mechanical Specialist] Completed mechanical evaluation with {len(report.split())} words.")
    return state

    