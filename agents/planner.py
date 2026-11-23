'''
Agent that plans the diagnostic process
'''

import os
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import ChatPromptTemplate
from infra.factory import factory
from infra.cost_calculator import calculate_token_cost
from agents.thought_parser import parse_thought_and_content

from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

def get_llm(cache_id=None, tier: str = "reasoning"):
    """
    Returns dynamically routed LLM instance based on deployment mode and tier.
    - tier='reasoning': Deep reasoning (Gemini 2.5 Pro / Llama 3.1)
    - tier='fast': High-speed routing & HyDE (Gemini 1.5 Flash / Llama 3.2)
    """
    config = factory.get_llm_config(tier=tier) 
    if factory.deployment_mode == 'CLOUD':
        kwargs = {
            "model_name": config["model_name"],
            "project": config["project_id"],
            "location": config["region"],
            "temperature": 0.2
        }
        if cache_id:
            kwargs["cached_content"] = cache_id
            print(f"-> Injecting Vertex KV Cache: {cache_id}")
        
        return ChatVertexAI(**kwargs)
    else:
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
    Agent Node: Rigorously verifies the synthesized plan against official manuals.
    Scores the plan from 1 to 10 for factual precision and safety.
    Approves ONLY if Score >= 8.
    """
    print("--- [Critique Agent] Scoring Plan Accuracy & Safety (1-10 Scale) ---")

    if "reasoning_traces" not in state or state["reasoning_traces"] is None:
        state["reasoning_traces"] = {}
    llm = get_llm(tier="reasoning")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Automotive Quality & Safety Inspector.
        Your responsibility is to mathematically evaluate the proposed diagnostic plan against the official OEM repair manuals.
        EVALUATION CRITERIA (1-10 Scale):
        - 9-10 (Flawless): Plan strictly matches official manual procedures, includes exact OEM torque specs (e.g. ft-lbs or Nm), proper safety pre-checks (e.g. battery disconnect), and has ZERO hallucinations.
        - 8 (Acceptable): Plan is factually accurate, safe, and logical. If official manuals are brief, it safely advises the technician without hallucinating specs.
        - 1-7 (Rejected): Plan hallucinated torque specs, missed critical safety warnings (e.g. high-voltage or fuel depressurization), or contradicts OEM manuals.
        MANDATORY INSTRUCTIONS:
        1. First, provide your reasoning inside private <thought>...</thought> tags:
        - Audit each step against the provided manual excerpts.
        - Point out any hallucinated torque specs or skipped safety steps.
        2. After </thought>, your output MUST follow this exact format:
        SCORE: <integer between 1 and 10>
        STATUS: <APPROVED if score >= 8 else REJECTED>
        FEEDBACK: <Actionable advice on what must be corrected or why it passed>"""),
        ("human", """Official OEM Manuals Context:
        {manuals}
        Proposed Diagnostic Plan:
        {plan}
        Audit and score the plan:""")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "manuals": state.get("retrieved_manuals", []),
        "plan": state.get("preliminary_plan")
    })

    # Intercept token usage from Critique LLM
    llm_config = factory.get_llm_config()
    update_token_metrics(state, response, llm_config.get("model_name", "unknown"))

    # Extract thought and cleaned feedback
    thought_trace, feedback = parse_thought_and_content(response.content)
    state["reasoning_traces"]["critique"] = thought_trace
    state["critique_feedback"] = feedback

    # Extract numeric score using regex
    score_match = re.search(r"SCORE:\s*(\d+)", feedback, re.IGNORECASE)
    critic_score = int(score_match.group(1)) if score_match else 7
    state["critic_score"] = critic_score
    print(f"-> [Critique Agent] Quality Score: {critic_score}/10")
    # Routing evaluation: Threshold is 8/10
    if critic_score >= 8 or "STATUS: APPROVED" in feedback.upper():
        state["is_approved"] = True
        state["requires_more_info"] = False
        # Append verified badge with score to final plan
        state["verified_plan"] = f"### 🛡️ Verified Diagnostic Plan (Quality Score: {critic_score}/10)\n\n" + state["preliminary_plan"]
        print(f"-> [Critique Agent] Plan PASSED with score {critic_score}/10.")
    else:
        state["is_approved"] = False
        state["requires_more_info"] = True
        print(f"-> [Critique Agent] Plan REJECTED (Score: {critic_score}/10). Initiating replanning loop.")
    return state