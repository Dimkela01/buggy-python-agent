from __future__ import annotations
from typing import Optional, TypedDict, Callable, Tuple
from langgraph.graph import StateGraph, END


#State passed between nodes
class State(TypedDict, total=False):
    task_id: str
    spec: str
    buggy_code: str
    entry_point: str
    tests: str

    candidate_code: str
    passed: bool
    error: Optional[str]


#Type hints matching existing components
LLMFixer = Callable[[str, str, str | None], str]
SandboxEval = Callable[[str, str, str], Tuple[bool, Optional[str]]]


def build_graph(fixer: LLMFixer, evaluator: SandboxEval):
    """
    Build a two-node LangGraph:
        fix (LLM) -> run (sandbox) -> END
    """

    def node_fix(state: State) -> State:
        code = fixer(state["spec"], state["buggy_code"], state.get("entry_point"))
        state["candidate_code"] = code
        return state

    def node_run(state: State) -> State:
        code = state.get("candidate_code", "")
        passed, err = evaluator(code, state["tests"], state["entry_point"])
        state["passed"] = passed
        state["error"] = err
        return state

    g = StateGraph(State)
    g.add_node("fix", node_fix)
    g.add_node("run", node_run)
    g.set_entry_point("fix")
    g.add_edge("fix", "run")
    g.add_edge("run", END)
    return g.compile()
