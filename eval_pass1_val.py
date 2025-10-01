from __future__ import annotations
from datasets import load_dataset
from qwen_fixer import QwenFixer
from sandbox import evaluate_candidate
from graph_agent import build_graph, State
from prompt import build_repair_prompt
import json


def main():
    ds = load_dataset("bigcode/humanevalpack", "python")
    test_split = ds["test"]

    fixer = QwenFixer("Qwen/Qwen2.5-0.5B-Instruct")
    app = build_graph(fixer, evaluate_candidate)

    # Warm up sandbox process once to avoid first-task spawn timeouts
    warmup_code = "def __warmup__():\n    return None\n"
    warmup_tests = "def check(__warmup__):\n    pass\ncheck(__warmup__)\n"
    evaluate_candidate(warmup_code, warmup_tests, "__warmup__")

    #Evaluate first 20 tasks
    n_eval = 20
    passed_count = 0
    results = []

    for i, ex in enumerate(test_split.select(range(n_eval))):
        task_id = ex["task_id"]
        spec = ex.get("prompt") or ex.get("declaration", "")
        buggy_code = ex["buggy_solution"]
        entry_point = ex["entry_point"]
        tests = ex["test"]

        prompt = build_repair_prompt(spec, buggy_code, entry_point=entry_point, include_few_shot=False)
        state: State = {
            "task_id": task_id,
            "spec": spec,
            "buggy_code": buggy_code,
            "entry_point": entry_point,
            "tests": tests,
        }

        result = app.invoke(state)
        result.update({
            "task_id": task_id,
            "candidate_code": result.get("candidate_code", ""),
            "error": result.get("error"),
        })
        results.append(result)

        if result.get("passed"):
            passed_count += 1

        print(f"[{i+1}/{n_eval}] {task_id} -> passed={result['passed']}")

    #Summary
    pass_at_1 = passed_count / n_eval
    print("\n--- SUMMARY (dev slice) ---")
    print(f"Evaluated: {n_eval}")
    print(f"Passed   : {passed_count}")
    print(f"pass@1   : {pass_at_1:.3f}")

    with open("results_val.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved detailed results to results_val.json")


if __name__ == "__main__":
    main()
