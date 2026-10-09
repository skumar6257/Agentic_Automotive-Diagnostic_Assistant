import os
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from infra.factory import factory
import torch

def ingest_manuals():
    print(f"--- Starting Vector DB Ingestion ({factory.deployment_mode} Mode) ---")

    # 1. Dynamically Load all .txt files from both directories
    all_files = []
    all_files.extend(glob.glob("data/raw/manuals/*.txt"))

    if not all_files:
        print("No text files found in data/raw/manuals/ or data/raw/tsbs/")
        return

    documents =[]
    for file_path in all_files:
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents.")

    # 2. Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(all_files)} files.")

    # 3. Dual-Mode Embeddings
    if factory.deployment_mode == "CLOUD":
        print("Using Cloud: Vertex AI Embeddings")
        # NOTE: This requires GCP credentials to be set up!
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004"
        )
    else:
        print("Using Offline: HuggingFace BGE Small (CPU Friendly)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {'device': device}
        model_name = "BAAI/bge-small-en-v1.5" if device == 'cpu' else "BAAI/bge-large-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True}
        
        # Define a local folder inside your project to hold the heavy weights
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(cache_dir, exist_ok=True)
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    # 4. Ingest into Qdrant
    print("Generating embeddings and saving to Qdrant...")
    if factory.deployment_mode == "CLOUD":
        config = factory.get_vector_db_config()
        qdrant = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            url=config["url"],
            api_key=config["api_key"],
            collection_name="repair_manuals",
        )
    else:
        os.makedirs("data/processed/qdrant_db", exist_ok=True)
        qdrant = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            path="data/processed/qdrant_db", # Saves DB to disk so we don't lose it!
            collection_name="repair_manuals",
        )
    
    print("Vector Ingestion Complete!")
    return qdrant

if __name__ == "__main__":
    # Ensure you are in CLOUD mode to use Vertex Embeddings
    if factory.deployment_mode != "CLOUD":
        print("WARNING: Please set DEPLOYMENT_MODE=CLOUD in .env for embeddings.")
    
    ingest_manuals()