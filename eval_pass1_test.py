from __future__ import annotations
from datasets import load_dataset
from qwen_fixer import QwenFixer
from sandbox import evaluate_candidate
from graph_agent import build_graph, State
import json


def main():
    ds = load_dataset("bigcode/humanevalpack", "python")
    test_split = ds["test"]
    n_total = len(test_split)
    start = 20  # first 20 samples
    end = 140  # limited capacity on CPU use n_total with enogh RAM and CPU
    assert start < n_total

    fixer = QwenFixer("Qwen/Qwen2.5-0.5B-Instruct") #small model
    app = build_graph(fixer, evaluate_candidate) #building langgraph model

    # Warm up sandbox once to avoid first-task timeouts
    warmup_code = "def __warmup__():\n    return None\n"
    warmup_tests = "def check(__warmup__):\n    pass\ncheck(__warmup__)\n"
    evaluate_candidate(warmup_code, warmup_tests, "__warmup__")

    passed = 0
    results = []

    #test loop
    for idx, ex in enumerate(test_split.select(range(start, end)), start=start):
        task_id = ex["task_id"]
        spec = ex.get("prompt") or ex.get("declaration", "")
        buggy_code = ex["buggy_solution"]
        entry_point = ex["entry_point"]
        tests = ex["test"]

        state: State = {
            "task_id": task_id,
            "spec": spec,
            "buggy_code": buggy_code,
            "entry_point": entry_point,
            "tests": tests,
        }

        try:
            result = app.invoke(state)
        except Exception as e:
            # Keep going even if a single task blows up
            result = {"passed": False, "error": f"RunnerError: {type(e).__name__}: {e}", "candidate_code": ""}

        result.update({
            "task_id": task_id,
            "candidate_code": result.get("candidate_code", ""),
            "error": result.get("error"),
        })
        results.append(result)

        if result.get("passed"):
            passed += 1
        print(f"[{idx+1}/{n_total}] {task_id} -> passed={result['passed']}")

    evaluated = len(results)
    pass_at_1 = passed / evaluated if evaluated else 0.0

    print("\n--- SUMMARY (final slice) ---")
    print(f"Evaluated: {evaluated}")
    print(f"Passed   : {passed}")
    print(f"pass@1   : {pass_at_1:.3f}")

    with open("results_final.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved detailed results to results_final.json")


if __name__ == "__main__":
    # Windows-friendly startup for multiprocessing
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main()
