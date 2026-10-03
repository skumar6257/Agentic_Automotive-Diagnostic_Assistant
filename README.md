# Agentic Automotive Diagnostic Assistant

An enterprise-grade, agentic AI workflow designed to diagnose automotive faults. This system utilizes a **Hybrid GraphRAG** architecture, combining the topological relationship mapping of a Graph Database (Neo4j) with the semantic search capabilities of a Vector Database (Qdrant/Vertex AI) to parse OBD-II Diagnostic Trouble Codes (DTCs), OEM repair manuals, and Technical Service Bulletins (TSBs).

## 🚀 Key Features

*   **Cyclic Agentic Reasoning:** Uses LangGraph to orchestrate reasoning, retrieval, critique, and reflection loops rather than a brittle linear pipeline.
*   **Hybrid RAG (Graph + Vector):** Combines LlamaIndex `PropertyGraphIndex` (Neo4j) for traversing multi-hop part relationships with Vertex AI Vector Search for retrieving dense unstructured manual steps.
*   **Cost-Optimized Context:** Utilizes Gemini 1.5 Pro with Vertex AI Context (KV) Caching to load massive OEM workshop manuals into memory once, drastically reducing per-query token costs and latency.
*   **Enterprise Safety Guardrails:** Integrated NeMo Guardrails intercept and block unsafe DIY advice (e.g., high-voltage EV battery handling without proper PPE).
*   **LLMOps & Evaluation:** Offline CI/CD evaluation pipeline using Ragas to measure Context Precision, Faithfulness, and Answer Relevance.

## 🏗 Architecture (Dual-Mode)

This application is designed with an `InfraFactory` that allows zero-downtime switching between two deployment modes via a single environment variable (`DEPLOYMENT_MODE`):

1. **CLOUD Mode (Managed):** Uses Google Vertex AI Embeddings and Gemini 1.5 Pro inference. Designed for high scalability and speed.

2. **OFFLINE Mode (Edge / Air-Gapped):** Uses local HuggingFace CPU-friendly embeddings (`BAAI/bge-small-en-v1.5`), local Qdrant, and local Neo4j Docker containers. Designed for maximum privacy and air-gapped shop floor environments.
---

## 🛠️ Setup & Installation
### Prerequisites
- Python 3.10+
- Docker Desktop (Required for Offline Graph Database)
### 1. Project Initialization
Clone the repository:
```bash
git clone https://github.com/skumar6257/Agentic_Automotive-Diagnostic-_Assistant
cd Agentic_Automotive-Diagnostic-_Assistant
```
Set up a Python Virtual Environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
### 2. Environment Configuration
Copy the environment template and configure your deployment mode:
```bash
cp .env.example .env
```
*Make sure `DEPLOYMENT_MODE=OFFLINE` is set in `.env` if you are testing locally.*
### 3. Start Local Databases (Offline Mode)
Spin up the local Neo4j graph database using Docker Compose:
```bash
docker-compose up -d
```
### 4. Data Ingestion Pipeline (Phase 2)
Before the AI Agents can run, we must fetch real-world automotive data and ingest it into our databases.
**Step 4a: Fetch Raw Data**
Downloads real OBD-II DTC codes and TSBs from the NHTSA API.
```bash
python data/fetch_real_data.py
```
**Step 4b: Vector Database Ingestion (Unstructured Text)**
Chunks the repair manuals and TSBs, generates embeddings, and saves the Vector DB to disk.
```bash
python index/vector_ingest.py
```
**Step 4c: Graph Database Ingestion (Topological Data)**
Parses the structured DTC codes into Neo4j nodes (DTC -> Part -> Subsystem).
```bash
python index/graph_ingest.py
```
---

## 🔄 Workflow Architecture

```mermaid
graph TD
    %% User Interaction
    Client[Client App / Technician] --> |VIN & OBD-II Code| API[Cloud Run API Gateway]
    
    %% Orchestration & Agentic Flow (LangGraph)
    subgraph Agentic Orchestration [LangGraph State Machine]
        API --> RouterNode{Query Router Agent}
        RouterNode --> |Requires Part Relations| GraphRAG[Knowledge Graph Agent]
        RouterNode --> |Requires Manuals/TSBs| VectorRAG[Document Retrieval Agent]
        RouterNode --> |Diagnostic Reasoning| Reasoning[Diagnostic Planner Agent]
        
        GraphRAG --> Reasoning
        VectorRAG --> Reasoning
        
        Reasoning --> Critique[Critique & Reflection Node]
        Critique --> |Refine| VectorRAG
        Critique --> |Pass| Guardrails[NeMo Guardrails / Safety Node]
    end
    
    %% Retrieval Engine (LlamaIndex)
    subgraph LlamaIndex RAG Engine
        VectorRAG --> VectorStore[(Vertex AI Vector Search)]
        GraphRAG --> GraphStore[(Neo4j / Memgraph on GKE)]
    end
    
    %% LLM & Memory
    subgraph Google Cloud & LLM
        Reasoning --> Gemini[Vertex AI: Gemini 1.5 Pro]
        Gemini --> KVCache[Vertex AI Context Caching]
    end
    
    %% Safety & Evals
    Guardrails --> |Approved Output| API
    Guardrails --> |Rejected/Unsafe| Fallback[Safe Fallback Response]
    
    %% Offline CI/CD
    subgraph CI/CD & LLMOps
        Telemetry[Traces & Logs] --> LangSmith[LangSmith / Vertex ML Metadata]
        EvalPipeline[Ragas Evaluation Pipeline] -.-> |Evaluates| Telemetry
    end
