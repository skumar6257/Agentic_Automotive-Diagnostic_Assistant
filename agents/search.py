from langchain_community.tools import DuckDuckGoSearchRun

def web_search_agent(state: dict):
    """
    Agent Node: If local databases lack info, searches the live internet 
    for diagnostic steps and specs.
    """
    print("--- [Web Search Agent] Hunting the web for missing manuals ---")

    vehicle = state.get("vehicle_info", "")
    dtc = state.get("dtc_code", "")

    # Initialize the free DuckDuckGo tool
    search = DuckDuckGoSearchRun()
    query = f"{vehicle} {dtc} diagnostic steps repair manual torque specs"

    print(f"Executing search: {query}")
    try:
        results = search.invoke(query)
    except Exception as e:
        results = f"Search failed: {e}"
        
    state["web_search_results"] = results
    return state