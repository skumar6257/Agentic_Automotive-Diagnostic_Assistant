# Agentic_Automotive-Diagnostic_Assistant
# Enterprise Agentic Assistant for Automotive Diagnostics 🚗🧠

An enterprise-grade, agentic AI diagnostic system built to assist dealership technicians and fleet mechanics. This system intelligently cross-references structured diagnostic data (OBD-II fault codes) with deeply technical, unstructured documents (NHTSA Recalls, OEM Workshop Manuals, Technical Service Bulletins).

By leveraging a cyclic agentic state machine, hybrid Graph+Vector retrieval, and advanced context caching, AutoDiag iteratively reasons through complex mechanical issues while strictly enforcing safety guardrails.

## 🚀 Key Features

*   **Cyclic Agentic Reasoning:** Uses LangGraph to orchestrate reasoning, retrieval, critique, and reflection loops rather than a brittle linear pipeline.
*   **Hybrid RAG (Graph + Vector):** Combines LlamaIndex `PropertyGraphIndex` (Neo4j) for traversing multi-hop part relationships with Vertex AI Vector Search for retrieving dense unstructured manual steps.
*   **Cost-Optimized Context:** Utilizes Gemini 1.5 Pro with Vertex AI Context (KV) Caching to load massive OEM workshop manuals into memory once, drastically reducing per-query token costs and latency.
*   **Enterprise Safety Guardrails:** Integrated NeMo Guardrails intercept and block unsafe DIY advice (e.g., high-voltage EV battery handling without proper PPE).
*   **LLMOps & Evaluation:** Offline CI/CD evaluation pipeline using Ragas to measure Context Precision, Faithfulness, and Answer Relevance.

## 🏗 Architecture

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
