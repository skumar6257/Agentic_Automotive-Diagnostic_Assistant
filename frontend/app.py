import streamlit as st
import requests

st.set_page_config(page_title="AI Diagnostic Assistant", page_icon="🚗", layout="wide")

st.title("🚗 Agentic Automotive Diagnostic Assistant")
st.markdown("Enter the vehicle information and OBD-II codes below to generate an enterprise-grade diagnostic plan.")

# Sidebar for Architecture Toggle
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
    else:
        st.info("☁️ Processing in Cloud (Google Vertex AI) with KV Caching.")

# Main Form
with st.form("diagnostic_form"):
    col1, col2 = st.columns(2)
    with col1:
        vehicle_info = st.text_input("Vehicle Info", placeholder="e.g. 2018 Toyota Camry")
    with col2:
        dtc_code = st.text_input("DTC Code", placeholder="e.g. P0300")

    symptoms = st.text_area("Symptoms & Customer Notes", placeholder="e.g. Engine misfire, rough idle...")
    
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
                    "deployment_mode": deployment_mode
                }

                # Call our FastAPI backend
                response = requests.post("http://localhost:8000/diagnose", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    st.subheader("📋 Verified Diagnostic Plan")
                    st.markdown(data["diagnostic_plan"])
                else:
                    st.error(f"Backend Error: {response.text}")

            except Exception as e:
                st.error(f"Failed to connect to backend. Is the FastAPI server running?")

                