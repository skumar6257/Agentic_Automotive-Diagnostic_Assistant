"""
Semantic Routing Agent
Classifies user query complexity to dynamically route between:
- SIMPLE: Ultra-fast informational answers (Gemini 1.5 Flash, bypass RAG)
- COMPLEX: Multi-step diagnostic investigations (Gemini 2.5 Pro, full Graph + Vector RAG)
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_core.prompts import ChatPromptTemplate
from agents.planner import get_llm, update_token_metrics
from infra.factory import factory

def semantic_router_agent(state: dict) -> dict:
    """
    Agent Node: Inspects DTC code and symptoms to classify query complexity.
    """
    print("--- [Semantic Router] Analyzing Query Complexity ---")
    symptoms = state.get("symptoms", "").strip()
    dtc_code = state.get("dtc_code", "").strip()
    vehicle = state.get("vehicle_info", "").strip()

    # If no symptoms or purely informational query keywords
    prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classification router for an automotive diagnostic system.
            Analyze the user's input and classify it into EXACTLY one category:
            - 'SIMPLE': The user is asking a basic informational, conceptual, or definition question (e.g., 'What does OBD stand for?', 'What is a MAF sensor?', 'Where is the OBD port?'). There is no vehicle troubleshooting or fault diagnosis required.
            - 'COMPLEX': The user has provided a Diagnostic Trouble Code (DTC), fault symptoms, vehicle issues, or is requesting repair/diagnostic procedures (e.g., 'Engine misfire P0300', 'Brakes squeaking', 'Car won't start').
            Respond with ONLY the single word: SIMPLE or COMPLEX."""),
                    ("human", "Vehicle: {vehicle}\nDTC Code: {dtc}\nSymptoms: {symptoms}\n\nClassification:")
                ])

    try:
        # Use fast model for routing classification
        llm = get_llm(tier="fast")
        chain = prompt | llm
        response = chain.invoke({
            "vehicle": vehicle,
            "dtc": dtc_code,
            "symptoms": symptoms
        })
        llm_config = factory.get_llm_config()
        update_token_metrics(state, response, llm_config.get("model_name", "unknown"))
        classification = response.content.strip().upper()
        if "SIMPLE" in classification:
            state["query_complexity"] = "SIMPLE"
            print("-> [Router] Classification: SIMPLE (Routing to fast-path informational response)")
        else:
            state["query_complexity"] = "COMPLEX"
            print("-> [Router] Classification: COMPLEX (Routing to full Hybrid RAG pipeline)")
    except Exception as e:
        # Failsafe: Default to COMPLEX if router encounters an issue
        print(f"-> [Router] Routing error: {e}. Defaulting to COMPLEX.")
        state["query_complexity"] = "COMPLEX"
    return state

def fast_informational_agent(state: dict) -> dict:
    """
    Fast-path Node: Directly answers simple informational questions 
    without expensive database traversals.
    """
    print("--- [Fast-Path Agent] Generating Direct Informational Answer ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Automotive Assistant. Provide a concise, accurate, and direct technical explanation to the user's automotive question."),
        ("human", "Question: {symptoms}\nVehicle: {vehicle_info}\nDTC Code: {dtc_code}")
    ])
    llm = get_llm(tier="fast")
    chain = prompt | llm
    response = chain.invoke({
        "symptoms": state.get("symptoms", ""),
        "vehicle_info": state.get("vehicle_info", ""),
        "dtc_code": state.get("dtc_code", "")
    })
    llm_config = factory.get_llm_config()
    update_token_metrics(state, response, llm_config.get("model_name", "unknown"))
    state["verified_plan"] = response.content
    state["is_approved"] = True
    return state



    