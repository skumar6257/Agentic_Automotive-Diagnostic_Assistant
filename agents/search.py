"""
Agent Node: Web Search Fallback
Queries live automotive sources for missing specs, torque values, and procedures
when local vector database and graph context lack specific details.
"""
import os
import sys

def web_search_agent(state: dict) -> dict:
    """
    Searches the live web for missing diagnostic manuals, DTC specs, and torque ratings.
    Guaranteed zero-crash execution.
    """
    print("--- [Web Search Agent] Hunting the web for missing manuals ---")

    vehicle = state.get("vehicle_info", "")
    dtc = state.get("dtc_code", "")
    query = f"{vehicle} {dtc} diagnostic steps repair manual torque specs"

    print(f"Executing search: {query}")
    results = ""

    try:
        # Support both the new 'ddgs' package and classic 'duckduckgo_search'
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            search_hits = list(ddgs.text(query, max_results=3))
            if search_hits:
                formatted = []
                for hit in search_hits:
                    title = hit.get("title", "OEM Source")
                    body = hit.get("body", "")
                    href = hit.get("href", "")
                    formatted.append(f"Source: {title} ({href})\nExcerpt: {body}")
                results = "\n\n".join(formatted)
                print(f"-> [Web Search Agent] Successfully retrieved {len(search_hits)} external excerpts.")
            else:
                results = "No relevant online repair manuals found."
                print("-> [Web Search Agent] No direct web hits found.")

    except Exception as e:
        print(f"-> [Web Search Agent] Search notice: {e}. Proceeding with internal knowledge.")
        results = f"Web search notice: {e}. Diagnostic synthesis will proceed using internal OEM knowledge."

    state["web_search_results"] = results
    return state