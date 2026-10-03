import os
from typing import Dict, Any
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

class InfraFactory:
    """
    Provider Factory Engine to route configurations and clients 
    based on the active DEPLOYMENT_MODE (CLOUD or OFFLINE).
    """

    def __init__(self):
        # Deployment Router
        self.deployment_mode = os.getenv("DEPLOYMENT_MODE", "CLOUD").upper() 
        if self.deployment_mode not in ["CLOUD", "OFFLINE"]: 
            raise ValueError(f"Invalid DEPLOYMENT_MODE: {self.deployment_mode}. Must be CLOUD or OFFLINE.")
        else:
            print(f"--- Initializing {self.deployment_mode} Mode ---")
        
    def get_llm_config(self) -> Dict[str, Any]:
        """Return LLM configuration based on deployment mode."""
        if self.deployment_mode == "CLOUD":
            return {
                "provider": "vertex_ai",
                "model_name": "gemini-1.5-pro",
                "project_id": os.getenv("GCP_PROJECT_ID"),
                "region": os.getenv("GCP_REGION")
            }
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"--- Using Device: {device} ---")
            if device =='cuda':
                return {
                    "provider": "vllm",
                    "base_url": os.getenv("VLLM_API_BASE_URL", "http://localhost:8000/v1"),
                    "model_name": os.getenv("VLLM_LOCAL_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
                    "api_key": os.getenv("VLLM_API_KEY", "local")
                }
            else:
                return {
                    "provider": "ollama",
                    "base_url": os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434/v1"),
                    "model_name": os.getenv("OLLAMA_LOCAL_MODEL_NAME", "llama3.2"),
                    "api_key": os.getenv("OLLAMA_API_KEY", "ollama")
                }
    
    def get_vector_db_config(self) -> Dict[str, str]:
        """Return vector DB configuration based on deployment mode."""
        if self.deployment_mode == "CLOUD":
            return {
                "provider": "vertex_vector_search",
                "index_id": os.getenv("VERTEX_VECTOR_INDEX_ID", ""),
                "endpoint_id": os.getenv("VERTEX_VECTOR_ENDPOINT_ID", ""),
            }
        else:
            return {
                "provider": "qdrant",
                "url": os.getenv("LOCAL_VECTOR_URL", "http://localhost:6333"),
                "api_key": os.getenv("LOCAL_VECTOR_API_KEY", ""),
            }

    def get_graph_db_config(self) -> Dict[str, str]:
        """Returns the configuration for the Neo4j Graph Database."""
        if self.deployment_mode == "CLOUD":
            return {
                "uri": os.getenv("AURA_DB_URI", ""),
                "username": os.getenv("AURA_DB_USER", "neo4j"),
                "password": os.getenv("AURA_DB_PASSWORD", ""),
            }
        else:
            return {
                "uri": os.getenv("LOCAL_GRAPH_URI", "bolt://localhost:7687"),
                "username": os.getenv("LOCAL_GRAPH_USER", "neo4j"),
                "password": os.getenv("LOCAL_GRAPH_PASSWORD", "localpassword"),
            }
    
    def get_data_path(self) -> str:
        """Return data path based on deployment mode."""
        if self.deployment_mode == "CLOUD":
            return os.getenv("DATA_PATH", "data/documents")
        else:
            return os.getenv("DATA_PATH", "data/documents")

# Singleton instance to be imported across the application
factory = InfraFactory()
if __name__ == "__main__":
    # Simple test to verify routing
    print("\n--- Testing InfraFactory ---")
    print("LLM Config:", factory.get_llm_config())
    print("Graph Config:", factory.get_graph_db_config())
    print("Vector Config:", factory.get_vector_db_config())
    print('Data Path:', factory.get_data_path())
        