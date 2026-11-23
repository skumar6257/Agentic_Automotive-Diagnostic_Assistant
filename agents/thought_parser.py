"""
Reasoning Harness Utility
Extracts and isolates structured <thought>...</thought> Chain-of-Thought blocks 
from agent responses for transparency, auditing, and clean UI display.
"""

import re
from typing import Tuple
def parse_thought_and_content(raw_text: str) -> Tuple[str, str]:
    """
    Parses an LLM response containing <thought>...</thought> tags.
    Returns:
        thought_trace (str): The internal deliberation/planning text.
        cleaned_content (str): The user-facing output with <thought> tags removed.
    """
    thought_match = re.search(r"<thought>(.*?)</thought>", raw_text, re.DOTALL | re.IGNORECASE)

    if thought_match:
        thought_trace = thought_match.group(1).strip()
        cleaned_content = re.sub(r"<thought>.*?</thought>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    else:
        thought_trace = "Direct response generated without explicit thought block."
        cleaned_content = raw_text.strip()
    return thought_trace, cleaned_content

