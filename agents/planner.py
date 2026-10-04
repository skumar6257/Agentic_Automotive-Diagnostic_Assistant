'''
Agent that plans the diagnostic process
'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import ChatPromptTemplate
from infra.factory import factory

from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

def get_llm(cache_id=None):
    if factory.deployment_mode == 'CLOUD':
        # Base configuration for Gemini
        kwargs = {
            "model_name": config["model_name"],
            "project": config["project_id"],
            "location": config["region"],
            "temperature": 0.2
        }
        
        # If a KV Cache ID was provided, inject it!
        if cache_id:
            kwargs["cached_content"] = cache_id
            print(f"-> Injecting Vertex KV Cache: {cache_id}")
        
        return ChatVertexAI(**kwargs)
    else:
        # Local models (vLLM, Ollama, LMStudio) expose an OpenAI-compatible API
        config = factory.get_llm_config()
        return ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model_name"],
            temperature=0.2
        )

def diagnostic_planner_agent(state: dict):
    """
    Agent Node: Synthesizes Graph topology and Vector manuals into a step-by-step plan.
    """
    print("--- [Planner Agent] Synthesizing Diagnostic Plan ---")
    
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Automotive Diagnostic AI. Create a step-by-step repair plan based ONLY on the provided context. If the context lacks sufficient data, provide a very general diagnostic approach and state that official manuals are required for safety. YOU MUST INCORPORATE PREVIOUS CRITIQUE FEEDBACK if provided!"),
        ("human", """
        Vehicle: {vehicle_info}
        DTC Code: {dtc_code}
        Symptoms: {symptoms}
        
        Graph Database Parts Topology:
        {topology}
        
        Vector Database Repair Manuals:
        {manuals}

        Web Search Fallback Data:
        {web_results}

        Previous Critique (Fix these issues if provided):
        {feedback}
        
        Draft a preliminary step-by-step diagnostic and repair plan. 
        """)
    ])

    chain = prompt | llm

    response = chain.invoke({
        "vehicle_info": state["vehicle_info"],
        "dtc_code": state["dtc_code"],
        "symptoms": state["symptoms"],
        "topology": state["topology_nodes"],
        "manuals": state["retrieved_manuals"],
        "web_results": state.get("web_search_results", "None"),
        "feedback": state.get("critique_feedback", "None")
    })

    state["preliminary_plan"] = response.content
    return state


def critique_agent(state: dict):
    """
    Agent Node: Reviews the preliminary plan against the manuals to ensure no steps 
    or torque specs were hallucinated.
    """
    print("--- [Critique Agent] Verifying Plan against Source Manuals ---")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        # ("system", "You are a Senior Master Technician. Review the proposed repair plan against the official manuals. Reply 'APPROVED' if the plan strictly follows the manual. If it hallucinates or misses critical safety specs, reply 'REJECTED' and explain why."),
        ("system", "You are a Senior Master Technician reviewing a proposed diagnostic plan against official manuals. \n\n1. If the plan hallucinated torque specs or steps not in the manual, reply 'REJECTED' and explain why.\n2. If the plan safely states that more official manuals are required, reply 'APPROVED'.\n3. If the plan strictly follows the provided manual, reply 'APPROVED'."),
        ("human", """
        Official Manuals: {manuals}
        
        Proposed Plan:
        {plan}
        
        Review the plan.
        """)
    ])

    chain = prompt | llm

    response = chain.invoke({
        "manuals": state.get("retrieved_manuals", []),
        "plan": state.get("preliminary_plan")
    })

    feedback = response.content
    state["critique_feedback"] = feedback

    # Simple routing logic
    if "APPROVED" in feedback.upper():
        state["safety_cleared"] = True
        state["verified_plan"] = state["preliminary_plan"]
    else:
        state["safety_cleared"] = False
        state["requires_more_info"] = True

    return state