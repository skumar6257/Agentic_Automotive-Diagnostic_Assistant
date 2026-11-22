"""
HyDE (Hypothetical Document Embeddings) Agent
Generates an artificial OEM technical repair paragraph to bridge the semantic 
gap between informal user symptoms and technical workshop manuals.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import ChatPromptTemplate
from agents.planner import get_llm, update_token_metrics
from infra.factory import factory

def hyde_expansion_agent(state: dict) -> dict:
    """
    Agent Node: Expands user symptoms into a technical diagnostic excerpt 
    before performing vector similarity search.
    """
    print("--- [HyDE Agent] Synthesizing Hypothetical Technical Excerpt ---")
    vehicle = state.get("vehicle_info", "")
    dtc = state.get("dtc_code", "")
    symptoms = state.get("symptoms", "")
    topology = state.get("topology_nodes", [])
    # Extract related parts discovered by graph retriever
    parts = [node["Part"] for node in topology if isinstance(node, dict) and "Part" in node]
    parts_str = ", ".join(parts) if parts else "Unknown Components"
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an OEM Automotive Technical Writer. 
Given a vehicle, DTC trouble code, and observed symptoms, write a hypothetical 1-to-2 paragraph excerpt from an official factory repair manual.
Use precise technical terminology (e.g., sensor resistances, wiring harnesses, signal voltages, mechanical tolerances, component inspection procedures).
Do NOT include greetings or conversational text. Output ONLY the technical service manual excerpt."""),
        ("human", """Vehicle: {vehicle}
DTC Trouble Code: {dtc}
Customer Symptoms: {symptoms}
Identified Subsystem Parts: {parts}
Factory Manual Technical Excerpt:""")
    ])

    try:
        llm = get_llm(tier="fast")
        chain = prompt | llm
        response = chain.invoke({
            "vehicle": vehicle,
            "dtc": dtc,
            "symptoms": symptoms,
            "parts": parts_str
        })
        hyde_text = response.content.strip()
        state["hyde_doc"] = hyde_text
        # Track token usage
        llm_config = factory.get_llm_config()
        update_token_metrics(state, response, llm_config.get("model_name", "unknown"))
        print(f"-> [HyDE] Generated {len(hyde_text.split())} words of technical diagnostic context.")
    except Exception as e:
        print(f"-> [HyDE] Expansion error: {e}. Falling back to raw symptoms.")
        state["hyde_doc"] = f"{symptoms} DTC: {dtc} Parts: {parts_str}"
    return state