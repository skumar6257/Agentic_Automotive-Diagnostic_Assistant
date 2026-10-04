'''
Agent to implement required guardrails
'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nemoguardrails import LLMRails, RailsConfig
from agents.planner import get_llm  # Import your dynamic factory LLM!


def safety_guardrail_agent(state: dict):
    """
    Agent Node: Intercepts the user input before any diagnostic work begins.
    If NeMo Guardrails detects a dangerous topic, it halts the graph.
    """
    print("--- [Safety Guardrail] Checking user input for enterprise compliance ---")

    # Force NeMo's internal embedding model to save in our local models/ folder
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.environ["FASTEMBED_CACHE_PATH"] = cache_dir

    # Load the NeMo config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'guardrails')
    config = RailsConfig.from_path(config_path)
    llm = get_llm()
    rails = LLMRails(config, llm=llm)

    user_input = str(state.get('symptoms', ''))

    # Pass input through NVIDIA Guardrails
    response = rails.generate(messages=[{"role": "user", "content": user_input}])
    bot_reply = response["content"]

    # If the response contains any of our refusal strings, the guardrail was triggered
    if "For your safety and compliance" in bot_reply or "I cannot provide instructions" in bot_reply or "I cannot assist with modifications" in bot_reply:
        state["safety_cleared"] = False
        state["verified_plan"] = f"⚠️ SAFETY GUARDRAIL TRIGGERED ⚠️\n{bot_reply}"
        print("🚨 Unsafe topic detected! Halting workflow. 🚨")
    else:
        # It's safe to proceed
        state["safety_cleared"] = True
        print("✅ Input is safe.")

    return state
