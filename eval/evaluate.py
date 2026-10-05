import os
import sys
# Ensure we can import from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datasets import Dataset

# --- HACK TO FIX RAGAS LEGACY LANGCHAIN IMPORTS ---
import types
import pydantic.v1
cm = types.ModuleType('langchain_community.chat_models')
cm.ChatVertexAI = object
sys.modules['langchain_community.chat_models'] = cm
sys.modules['langchain_core.pydantic_v1'] = pydantic.v1
sys.modules['langchain.pydantic_v1'] = pydantic.v1       # <-- ADD THIS LINE
# --------------------------------------------------

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from graph.workflow import build_workflow
from agents.planner import get_llm
from langchain_community.embeddings import FastEmbedEmbeddings

def run_evaluation():
    print("--- [Ragas] Starting Offline LLMOps Evaluation Pipeline ---")
    
    # 1. Define our synthetic test cases (Ground Truth)
    questions = [
        "Engine misfire, rough idle, check engine light flashing.",
    ]
    ground_truths = [
        "A flashing check engine light with a rough idle typically indicates a severe misfire (P0300). Immediate action is required to prevent catalytic converter damage. Check spark plugs and ignition coils."
    ]

    # 2. Run the LangGraph Workflow to generate answers
    app = build_workflow()
    answers = []
    contexts = []

    for q in questions:
        print(f"\nEvaluating Query: {q}")
        initial_state = {
            "vehicle_info": "2018 Toyota Camry",
            "dtc_code": "P0300",
            "symptoms": q
        }

        # Execute the graph
        config = {"recursion_limit": 10}
        try:
            final_state = app.invoke(initial_state, config)
            ans = final_state.get("verified_plan", "No plan generated")
            manuals = final_state.get("retrieved_manuals", [])
            ctx = "\n".join(manuals) if manuals else "No context retrieved"

        except Exception as e:
            ans = f"Error: {e}"
            ctx = ""
            
        answers.append(ans)
        # Ragas expects a list of context strings per answer
        contexts.append([ctx]) 

        # 3. Format the data for Ragas
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        dataset = Dataset.from_dict(data)

    # 4. Initialize our Local LLM and Embeddings to act as the "Judge"
    print("\n--- [Ragas] Initializing the AI Judge ---")
    judge_llm = get_llm()
    judge_embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5. Run the Evaluation
    print("--- [Ragas] Grading the Agent's responses... (This may take a few minutes) ---")
    result = evaluate(
        dataset = dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    print("\n" + "="*40)
    print("RAGAS EVALUATION RESULTS 🏆")
    print("="*40)
    print(result)

# Save results for CI/CD tracking
    df = result.to_pandas()
    df.to_csv("eval_results.csv", index=False)
    print("Results saved to eval_results.csv")
if __name__ == "__main__":
    run_evaluation()