from langgraph.graph import StateGraph

class AppState(TypedDict):
    input: dict
    subtasks: list
    retrieved: dict
    answer: str
    confidence: float

# Add nodes: preprocess -> planner -> retriever -> orchestrator -> verifier -> format
workflow = StateGraph(AppState)
# ... wire nodes as per steps 1-6
app = workflow.compile()
