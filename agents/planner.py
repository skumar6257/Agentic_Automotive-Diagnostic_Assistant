'''
Agent that plans the diagnostic process
'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import ChatPromptTemplate
from infra.factory import factory
from infra.cost_calculator import calculate_token_cost

from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

def get_llm(cache_id=None):
    config = factory.get_llm_config() 
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
        return ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model_name"],
            temperature=0.2
        )

def update_token_metrics(state: dict, response, model_name: str):
    """
    Extracts usage_metadata from LangChain AIMessage response and 
    accumulates prompt, candidate, and total tokens in state['token_usage'].
    """

    if "token_usage" not in state or state["token_usage"] is None:
        state["token_usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "model_name": model_name
        }

    # Extract usage_metadata (Standard LangChain 0.2+ / 1.0+ structure)
    usage = getattr(response, "usage_metadata", None)

    prompt_tokens = 0
    completion_tokens = 0

    if usage:
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
    elif hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
        # Fallback for OpenAI / Ollama compatible metadata
        tok = response.response_metadata["token_usage"]
        prompt_tokens = tok.get("prompt_tokens", 0)
        completion_tokens = tok.get("completion_tokens", 0)
    else:
        # Fallback approximation (4 chars ~= 1 token) if metadata is omitted by provider
        completion_tokens = len(str(response.content)) // 4
    
    # Accumulate tokens across all agent steps
    state["token_usage"]["prompt_tokens"] += prompt_tokens
    state["token_usage"]["completion_tokens"] += completion_tokens
    state["token_usage"]["total_tokens"] = (
        state["token_usage"]["prompt_tokens"] + state["token_usage"]["completion_tokens"]
    )
    state["token_usage"]["model_name"] = model_name
    # Calculate real-time estimated cost
    state["token_usage"]["estimated_cost_usd"] = calculate_token_cost(
        model_name=model_name,
        deployment_mode=factory.deployment_mode,
        prompt_tokens=state["token_usage"]["prompt_tokens"],
        completion_tokens=state["token_usage"]["completion_tokens"]
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

    # Intercept token usage from Planner LLM
    llm_config = factory.get_llm_config()
    update_token_metrics(state, response, llm_config.get("model_name", "unknown"))

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

    # Intercept token usage from Critique LLM
    llm_config = factory.get_llm_config()
    update_token_metrics(state, response, llm_config.get("model_name", "unknown"))

    feedback = response.content
    state["critique_feedback"] = feedback

    # Simple routing logic
    if "APPROVED" in feedback.upper():
        state["is_approved"] = True
        state["requires_more_info"] = False
        state["verified_plan"] = state["preliminary_plan"]
    else:
        state["is_approved"] = False
        state["requires_more_info"] = True

    return state