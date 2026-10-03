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

    # try:
    #     print("\n[Executing LangGraph Agents...]")
    #     # Use stream() to watch the agents think in real-time!
    #     final_state = initial_state
    #     for event in app.stream(initial_state, config):
    #         for node_name, state_update in event.items():
    #             print(f"\n✅ Node Completed: {node_name.upper()}")
                
    #             if node_name == "critique":
    #                 print(f"🤖 Critique Feedback:\n{state_update.get('critique_feedback')}")
                
    #             # Keep track of the latest state
    #             final_state.update(state_update)

    #     print("\n" + "="*40)
    #     print("🏁 FINAL DIAGNOSTIC PLAN 🏁")
    #     print("="*40)
    #     print(final_state.get("verified_plan", "No plan was verified. The model couldn't fix the issues."))

    # except Exception as e:
    #     print("\n[Failsafe Triggered] Graph looped too many times and was killed to save compute!")
    #     print(f"Error: {e}")

if __name__ == "__main__":
    main()