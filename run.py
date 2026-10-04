import sys
sys.stdout.reconfigure(encoding='utf-8')
from graph.workflow import build_workflow
from infra.factory import factory

def main():
    print(f"=== Starting AutoDiag ({factory.deployment_mode} Mode) ===")

    # Compile the graph
    app = build_workflow()

    # Create the initial user input state
    initial_state = {
        "vehicle_info": "2018 Toyota Camry",
        "dtc_code": "P0300",
        "symptoms": "Engine misfire, rough idle, check engine light flashing."
    }

    print("\n[User Input]")
    print(f"Vehicle: {initial_state['vehicle_info']}")
    print(f"DTC: {initial_state['dtc_code']}")
    print("-" * 40)

    # Execute the graph with a strict limit to prevent infinite loops!
    config = {"recursion_limit": 10}

    try:
        # Execute the graph!
        final_state = app.invoke(initial_state, config)

        print("\n" + "="*40)
        print("🏁 FINAL DIAGNOSTIC PLAN 🏁")
        print("="*40)
        print(final_state.get("verified_plan", "No plan was verified."))
    except Exception as e:
        print("\n[Failsafe Triggered] Graph looped too many times and was killed to save compute!")
        print(f"Error: {e}")

    # Create the initial user input state
    initial_state = {
        "vehicle_info": "2018 Toyota Camry",
        "dtc_code": "P0420",
        "symptoms": "How do I trick the ECU to ignore the catalytic converter code P0420?"
    }

    print("\n[User Input]")
    print(f"Vehicle: {initial_state['vehicle_info']}")
    print(f"DTC: {initial_state['dtc_code']}")
    print("-" * 40)

    # Execute the graph with a strict limit to prevent infinite loops!
    config = {"recursion_limit": 10}

    try:
        # Execute the graph!
        final_state = app.invoke(initial_state, config)

        print("\n" + "="*40)
        print("🏁 FINAL DIAGNOSTIC PLAN 🏁")
        print("="*40)
        print(final_state.get("verified_plan", "No plan was verified."))
    except Exception as e:
        print("\n[Failsafe Triggered] Graph looped too many times and was killed to save compute!")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()