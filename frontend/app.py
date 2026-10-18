import streamlit as st
import requests
import os
from dotenv import load_dotenv
                
load_dotenv()

st.set_page_config(page_title="AI Diagnostic Assistant", page_icon="🚗", layout="wide")

st.title("🚗 Agentic Automotive Diagnostic Assistant")
st.markdown("Enter the vehicle information and OBD-II codes below to generate an enterprise-grade diagnostic plan.")

# Initialize session state for telemetry and plan persistence
if "last_telemetry" not in st.session_state:
    st.session_state["last_telemetry"] = None
if "last_plan" not in st.session_state:
    st.session_state["last_plan"] = None


# ==========================================
# SIDEBAR: Architecture & Observability
# ==========================================
with st.sidebar:
    st.header("⚙️ Architecture Settings")
    st.markdown("Switch between Cloud AI and Local Edge Computing.")

    deployment_mode = st.radio(
        "Deployment Mode:",
        ("OFFLINE", "CLOUD"),
        index=0,
        help="OFFLINE uses local Llama 3.1 & Neo4j. CLOUD uses Gemini 1.5 & Vertex AI."
    )

    if deployment_mode == "OFFLINE":
        st.success("🔒 Processing securely on local Edge device (vLLM/Ollama).")
        model_name = None
    else:
        st.info("☁️ Processing in Cloud (Google Vertex AI) with KV Caching.")
        model_name = st.selectbox(
            "Select Vertex AI Model:",
            ("gemini-2.5-pro", "gemini-1.5-flash-001", "gemini-1.0-pro"),
            index=0
        )

    # Observability & Token Economics Panel
    st.divider()
    st.header("📊 Token Economics & Telemetry")

    telemetry = st.session_state["last_telemetry"]
    if telemetry:
        token_usage = telemetry.get("token_usage", {})
        total_tokens = token_usage.get("total_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        cost = token_usage.get("estimated_cost_usd", 0.0)
        total_latency_ms = telemetry.get("total_latency_ms", 0.0)
        # Metric Cards
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Tokens Used", f"{total_tokens:,}")
        with col_m2:
            if deployment_mode == "OFFLINE":
                st.metric("Est. Cost", "$0.0000", help="Local hardware execution has zero API cost.")
            else:
                st.metric("Est. Cost", f"${cost:.5f}", help=f"Calculated using Google Vertex AI rates for {token_usage.get('model_name')}.")
        st.caption(f"Prompt Tokens: {prompt_tokens:,} | Output Tokens: {completion_tokens:,}")
        st.metric("Total Latency", f"{total_latency_ms / 1000:.2f} s", f"{total_latency_ms:.0f} ms")
        # Node-by-Node Latency Breakdown
        node_latencies = telemetry.get("node_latencies", {})
        if node_latencies:
            with st.expander("⏱️ Node Latency Breakdown", expanded=False):
                for node, latency in node_latencies.items():
                    st.write(f"**{node}**: `{latency:.1f} ms`")
    else:
        st.info("Submit a diagnostic query to view real-time token usage, cost, and node latencies.")

# Main Form
with st.form("diagnostic_form"):
    col1, col2 = st.columns(2)
    with col1:
        vehicle_info = st.text_input("Vehicle Info", value="2018 Toyota Camry", placeholder="e.g. 2018 Toyota Camry")
    
    with col2:
        dtc_code = st.text_input("DTC Code", value="P0300", placeholder="e.g. P0300")
    
    symptoms = st.text_area(
        "Symptoms & Customer Notes", 
        value="Engine misfire, rough idle, check engine light flashing.",
        placeholder="e.g. Engine misfire, rough idle..."
    )    
    
    submit = st.form_submit_button("Generate Diagnostic Plan")

if submit:
    if not vehicle_info or not dtc_code or not symptoms:
        st.warning("Please fill out all fields.")
    else:
        with st.spinner("Agents are traversing Knowledge Graphs and Manuals... (This may take a minute on CPU)"):
            try:
                payload = {
                    "vehicle_info": vehicle_info,
                    "dtc_code": dtc_code,
                    "symptoms": symptoms,
                    "deployment_mode": deployment_mode,
                    "model_name": model_name
                }

                # Call our FastAPI backend
                # response = requests.post("http://localhost:8000/diagnose", json=payload)
                # response = requests.post("https://agentic-diagnostic-api-qlbqtryffq-uc.a.run.app/diagnose", json=payload)

                # 1. Dynamically select the API URL based on the UI toggle!
                if deployment_mode == "CLOUD":
                    api_url = os.getenv("CLOUD_API_URL")
                    # api_url = os.getenv("LOCAL_API_URL", "http://localhost:8000/diagnose")
                else:
                    api_url = os.getenv("LOCAL_API_URL", "http://localhost:8000/diagnose")
                
                # 2. Call the dynamically selected backend
                response = requests.post(api_url, json=payload)


                if response.status_code == 200:
                    data = response.json()

                    # Save telemetry to session state for sidebar persistence
                    st.session_state["last_telemetry"] = data.get("telemetry")
                    st.session_state["last_plan"] = data.get("diagnostic_plan", "No plan returned.")
                    st.rerun()

                    st.subheader("📋 Verified Diagnostic Plan")
                    st.markdown(data.get("diagnostic_plan", "No plan returned."))

                    # # Quick Summary Metrics above feedback
                    # t = data.get("telemetry", {})
                    # u = t.get("token_usage", {})
                    # c1, c2, c3 = st.columns(3)
                    # c1.info(f"🪙 **Tokens**: {u.get('total_tokens', 0):,}")
                    # c2.info(f"💰 **Est. Cost**: ${u.get('estimated_cost_usd', 0.0):.5f}" if deployment_mode == "CLOUD" else "💰 **Est. Cost**: $0.0000 (Local)")
                    # c3.info(f"⚡ **Total Time**: {t.get('total_latency_ms', 0):.0f} ms")

                else:
                    st.error(f"Backend Error: {response.text}")

            except Exception as e:
                st.error(f"Failed to connect to backend. True Error: {str(e)}")

# Display Verified Plan and Telemetry Banner (persists across reruns)
if st.session_state.get("last_plan"):
    st.subheader("📋 Verified Diagnostic Plan")
    st.markdown(st.session_state["last_plan"])

    t = st.session_state.get("last_telemetry", {})
    u = t.get("token_usage", {})
    c1, c2, c3 = st.columns(3)
    c1.info(f"🪙 **Tokens**: {u.get('total_tokens', 0):,}")
    c2.info(f"💰 **Est. Cost**: ${u.get('estimated_cost_usd', 0.0):.5f}" if deployment_mode == "CLOUD" else "💰 **Est. Cost**: $0.0000 (Local)")
    c3.info(f"⚡ **Total Time**: {t.get('total_latency_ms', 0):.0f} ms")


                