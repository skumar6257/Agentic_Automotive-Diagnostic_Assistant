import os
import sys
# Ensure project root is always in sys.path regardless of execution directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from graph.workflow import build_workflow
from infra.factory import factory

app = FastAPI(
    title="Agentic Automotive Diagnostic API",
    description="Enterprise Multi-Agent Diagnostic Engine with Real-Time Observability & Token Economics"
)

class DiagnosticRequest(BaseModel):
    vehicle_info: str = Field(..., example="2018 Toyota Camry")
    dtc_code: str = Field(..., example="P0300")
    symptoms: str = Field(..., example="Engine misfire, rough idle, check engine light flashing.")
    deployment_mode: str = Field("CLOUD", example="CLOUD")  # "CLOUD" or "OFFLINE"
    model_name: Optional[str] = Field(None, example="gemini-2.5-pro")  # UI override

class TokenUsageSchema(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model_name: str = "unknown"

class TelemetrySchema(BaseModel):
    node_latencies: Dict[str, float] = {}
    total_latency_ms: float = 0.0
    token_usage: TokenUsageSchema

class DiagnosticResponse(BaseModel):
    status: str
    deployment_mode: str
    diagnostic_plan: str
    telemetry: TelemetrySchema

# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/diagnose", response_model=DiagnosticResponse)
def diagnose(req: DiagnosticRequest):
    """
    Executes the multi-agent diagnostic graph and returns the verified plan
    along with full execution latency and token economics telemetry.
    """
    try:
        # Dynamically override the factory's deployment mode for this request!
        factory.deployment_mode = req.deployment_mode
        factory.model_override = req.model_name
        os.environ["DEPLOYMENT_MODE"] = req.deployment_mode

        # Build the graph pipeline
        workflow = build_workflow()
        
        initial_state = {
            "vehicle_info": req.vehicle_info,
            "dtc_code": req.dtc_code,
            "symptoms": req.symptoms
        }
        
        config = {"recursion_limit": 10}
        
        # Execute the agentic workflow
        final_state = workflow.invoke(initial_state, config)
        
        # Extract the verified plan
        plan = final_state.get("verified_plan", "No plan generated.")
        
        # Extract Telemetry (with safe fallbacks)
        node_latencies = final_state.get("node_latencies") or {}
        total_latency_ms = final_state.get("total_latency_ms", 0.0)

        raw_token_usage = final_state.get("token_usage") or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "model_name": req.model_name or "none"
        }

        telemetry = TelemetrySchema(
            node_latencies=node_latencies,
            total_latency_ms=total_latency_ms,
            token_usage=TokenUsageSchema(
                prompt_tokens=raw_token_usage.get("prompt_tokens", 0),
                completion_tokens=raw_token_usage.get("completion_tokens", 0),
                total_tokens=raw_token_usage.get("total_tokens", 0),
                estimated_cost_usd=raw_token_usage.get("estimated_cost_usd", 0.0),
                model_name=raw_token_usage.get("model_name", req.model_name or "unknown")
            )
        )

        return DiagnosticResponse(
            status="success",
            deployment_mode=req.deployment_mode,
            diagnostic_plan=plan,
            telemetry=telemetry
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)} || Traceback: {error_details}")

