'''
Agent that retrieves relevant information from the Neo4j database and Qdrant vector store.
'''

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from neo4j import GraphDatabase
from langchain_qdrant import QdrantVectorStore
from infra.factory import factory
import torch

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_vector_store():
    # 3. Dual-Mode Embeddings
    if factory.deployment_mode == "CLOUD":
        # NOTE: This requires GCP credentials to be set up!
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004"
        )
        config = factory.get_vector_db_config()
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name="repair_manuals",
            url=config["url"],
            api_key=config["api_key"]
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {'device': device}
        model_name = "BAAI/bge-small-en-v1.5" if device == 'cpu' else "BAAI/bge-large-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True}
        
        # Define a local folder inside your project to hold the heavy weights
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name="repair_manuals",
            path="data/processed/qdrant_db"
        )

def get_property_graph_store() -> Neo4jPropertyGraphStore:
    """
    Initializes LlamaIndex Neo4j PropertyGraphStore using factory credentials.
    """
    config = factory.get_graph_db_config()
    return Neo4jPropertyGraphStore(
        username=config["username"],
        password=config["password"],
        url=config["uri"]
    )

def query_graph_agent(state: dict):
    """
    Agent Node: Uses LlamaIndex Property Graph Store to extract the part topology
    and subsystem relationships for the given DTC trouble code.
    """
    dtc_code = state.get("dtc_code", "").strip()
    print(f"--- [LlamaIndex Graph Agent] Extracting Topology for {dtc_code} ---")

    results = []
    try:
        graph_store = get_property_graph_store()

        # Cypher query executed via LlamaIndex Neo4j Store session
        cypher_query = """
        MATCH (d:DTC {code: $code})-[:INDICATES_ISSUE_IN]->(p:Part)-[:LOCATED_IN]->(s:Subsystem)
        RETURN p.name AS Part, s.name AS Subsystem, d.description AS Description
        """
        
        records = graph_store.structured_query(cypher_query, param_map={"code": dtc_code})
        
        for record in records:
            results.append({
                "Part": record.get("Part", "Unknown Part"),
                "Subsystem": record.get("Subsystem", "Unknown Subsystem"),
                "Description": record.get("Description", "")
            })
            
        print(f"-> [LlamaIndex Graph] Extracted {len(results)} topological relationships.")

    except Exception as e:
        print(f"-> [LlamaIndex Graph] Graph traversal notice: {e}. Falling back to direct connection.")
        # Graceful fallback if LlamaIndex driver meets an environment quirk
        try:
            config = factory.get_graph_db_config()
            driver = GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"]))
            with driver.session() as session:
                records = session.run(
                    "MATCH (d:DTC {code: $code})-[:INDICATES_ISSUE_IN]->(p:Part)-[:LOCATED_IN]->(s:Subsystem) RETURN p.name AS Part, s.name AS Subsystem, d.description AS Description",
                    code=dtc_code
                )
                for r in records:
                    results.append({"Part": r["Part"], "Subsystem": r["Subsystem"], "Description": r["Description"]})
            driver.close()
        except Exception as fallback_err:
            print(f"-> [Graph Agent] Fallback failed: {fallback_err}")

    state["topology_nodes"] = results
    return state

def query_vector_agent(state: dict):
    """
    Agent Node: Queries Qdrant for repair manuals based on symptoms and parts.
    Updates the 'retrieved_manuals' in the state.
    """
    print("--- [Vector Agent] Fetching unstructured manuals ---")

    # Build a search query combining the symptom and the parts we found in the graph

     # Use HyDE expanded document if available, else fallback to raw symptoms
    if state.get("hyde_doc"):
        search_query = state["hyde_doc"]
        print("-> [Vector Agent] Searching Qdrant using HyDE dense representation.")
    else:
        parts = [node["Part"] for node in state.get("topology_nodes", []) if isinstance(node, dict) and "Part" in node]
        search_query = f"Symptoms: {state.get('symptoms')}. Related parts: {', '.join(parts)}"
        print("-> [Vector Agent] Searching Qdrant using fallback symptom query.")

    vector_store = get_vector_store()

    # Stage 1: High Recall - Retrieve up to 50 candidate chunks
    docs = vector_store.similarity_search(search_query, k=50)
    raw_candidates = [doc.page_content for doc in docs]
    state["raw_manual_candidates"] = raw_candidates
    # Temporary fallback in case reranker node is skipped
    state["retrieved_manuals"] = raw_candidates[:3]
    
    print(f"-> [Vector Agent] Stage 1 Recall: Pulled {len(raw_candidates)} candidate chunks from Qdrant.")
    return state

    
