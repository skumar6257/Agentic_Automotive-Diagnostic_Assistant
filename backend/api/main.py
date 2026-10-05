import os
import sys
# Ensure we can import from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph.workflow import build_workflow
from infra.factory import factory

app = FastAPI(title="Agentic Diagnostic API")

class DiagnosticRequest(BaseModel):
    vehicle_info: str
    dtc_code: str
    symptoms: str
    deployment_mode: str = "OFFLINE" # "CLOUD" or "OFFLINE"

@app.post("/diagnose")
def diagnose(req: DiagnosticRequest):
    try:
        # Dynamically override the factory's deployment mode for this request!
        factory.deployment_mode = req.deployment_mode
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
        
        plan = final_state.get("verified_plan", "No plan generated.")

        return {
            "status": "success",
            "deployment_mode": req.deployment_mode,
            "diagnostic_plan": plan
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
