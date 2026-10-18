"""
Token Economics & Cost Calculator
Calculates real-world inference costs based on active provider and model.
"""

from typing import Dict, Any

# Standard Vertex AI Pricing per 1,000,000 tokens (as of 2026/latest tier)
PRICING_PER_MILLION = {
    # Gemini 1.5 Flash / Flash-Lite
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-001": {"input": 0.075, "output": 0.30},
    # Gemini 2.5 Pro / 1.5 Pro
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro-001": {"input": 1.25, "output": 5.00},
    # Default fallback for Gemini models
    "default_vertex": {"input": 0.15, "output": 0.60}
}

def calculate_token_cost(
    model_name: str,
    deployment_mode: str,
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    """
    Computes dollar cost for inference.
    - If OFFLINE: Returns 0.0 (local hardware compute is free).
    - If CLOUD: Calculates exact dollar amount based on Vertex AI model tiers.
    """

    if deployment_mode.upper() == "OFFLINE":
        return 0.0

    # Match model pricing
    clean_name = model_name.lower().strip() if model_name else "default_vertex"
    rates = PRICING_PER_MILLION.get(clean_name, PRICING_PER_MILLION["default_vertex"])

    input_cost = (prompt_tokens / 1_000_000) * rates["input"]
    output_cost = (completion_tokens / 1_000_000) * rates["output"]
    total_cost = round(input_cost + output_cost, 6)
    return total_cost