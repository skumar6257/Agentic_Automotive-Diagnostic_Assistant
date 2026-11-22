"""
Cross-Encoder Re-Ranking Agent
Takes candidate chunks retrieved by Qdrant (high recall), performs full 
cross-attention scoring against the diagnostic query, and selects the Top 3 (high precision).
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Tuple
from sentence_transformers import CrossEncoder
# Global singleton so model weights stay warm in memory
_RERANKER_MODEL = None

def get_reranker_model() -> CrossEncoder:
    """
    Loads a lightweight, high-speed Cross-Encoder model.
    Runs locally on CPU in ~30ms, completely offline with zero API cost.
    """
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        os.makedirs(cache_dir, exist_ok=True)
        print("-> [Re-Ranker] Initializing Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
        _RERANKER_MODEL = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            automodel_args={"cache_dir": cache_dir}
        )
    return _RERANKER_MODEL

def cross_encoder_rerank_agent(state: dict) -> dict:
    """
    Agent Node: Re-ranks raw manual candidates from Qdrant and picks the Top 3.
    """
    print("--- [Re-Ranker Agent] Scoring and Filtering Context Chunks ---")
    candidates: List[str] = state.get("raw_manual_candidates", [])
    if not candidates:
        print("-> [Re-Ranker] No candidates found in state. Keeping empty.")
        state["retrieved_manuals"] = []
        state["rerank_scores"] = []
        return state

    # Use HyDE technical text as query if available, otherwise raw symptoms
    query = state.get("hyde_doc") or state.get("symptoms", "")
    print(f"-> [Re-Ranker] Evaluating {len(candidates)} candidate chunks against query...")
    # Build (query, chunk) pairs for cross-attention
    pairs = [[query, chunk] for chunk in candidates]
    try:
        model = get_reranker_model()
        scores = model.predict(pairs)
        # Pair each chunk with its score and sort descending
        scored_chunks: List[Tuple[str, float]] = list(zip(candidates, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        # Select Top 3 most relevant chunks
        top_k = min(3, len(scored_chunks))
        top_results = scored_chunks[:top_k]
        state["retrieved_manuals"] = [chunk for chunk, _ in top_results]
        state["rerank_scores"] = [round(float(score), 4) for _, score in top_results]
        print(f"-> [Re-Ranker] Successfully selected Top {top_k} chunks (Top score: {state['rerank_scores'][0] if state['rerank_scores'] else 'N/A'})")
    except Exception as e:
        print(f"-> [Re-Ranker] Error during re-ranking: {e}. Falling back to first 3 chunks.")
        state["retrieved_manuals"] = candidates[:3]
        state["rerank_scores"] = []
    return state