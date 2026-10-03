'''
Agent that retrieves relevant information from the Neo4j database and Qdrant vector store.
'''

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def query_graph_agent(state: dict):
    """
    Agent Node: Queries Neo4j for the part topology related to the DTC code.
    Updates the 'topology_nodes' in the state.
    """
    dtc_code = state.get("dtc_code","")
    print(f"--- [Graph Agent] Fetching topology for {dtc_code} ---")

    config = factory.get_graph_db_config()
    uri = config["uri"]
    user = config["username"]
    password = config["password"]

    driver = GraphDatabase.driver(uri, auth=(user, password))

    cypher_query = """
    MATCH (d:DTC {code: $code})-[:INDICATES_ISSUE_IN]->(p:Part)-[:LOCATED_IN]->(s:Subsystem)
    RETURN p.name AS Part, s.name AS Subsystem, d.description AS Description
    """

    results=[]
    try:
        with driver.session() as session:
            records = session.run(cypher_query, code=dtc_code)
            for record in records:
                results.append({
                    "Part": record["Part"],
                    "Subsystem": record["Subsystem"],
                    "Description": record["Description"]
                })
    except Exception as e:
        print(f"Graph retrieval error: {e}")
    finally:
        driver.close()

    state["topology_nodes"] = results
    return state

def query_vector_agent(state: dict):
    """
    Agent Node: Queries Qdrant for repair manuals based on symptoms and parts.
    Updates the 'retrieved_manuals' in the state.
    """
    print("--- [Vector Agent] Fetching unstructured manuals ---")

    # Build a search query combining the symptom and the parts we found in the graph

    parts = [node["Part"] for node in state.get("topology_nodes", [])]
    search_query = f"Symptoms: {state.get('symptoms')}. Related parts: {', '.join(parts)}"

    vector_store = get_vector_store()

    # Retrieve top 3 most relevant chunks
    docs = vector_store.similarity_search(search_query, k=3)

    manuals = [doc.page_content for doc in docs]
    state["retrieved_manuals"] = manuals
    
    return state

    
