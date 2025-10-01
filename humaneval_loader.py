from __future__ import annotations
from typing import Iterable, List, Optional
from datasets import load_dataset
from agent import RepairTask


def _is_repair(x) -> bool:
    return bool(x.get("buggy_solution")) and bool(x.get("prompt"))


def load_repair_tasks(limit: Optional[int] = None) -> List[RepairTask]:
    ds = load_dataset("bigcode/humanevalpack", "python")
    test = ds["test"].filter(_is_repair)

    tasks: List[RepairTask] = []
    for i, ex in enumerate(test):
        tasks.append(
            RepairTask(
                task_id=str(ex.get("task_id", f"hefix-{i}")),
                spec=ex["prompt"],                 # spec (signature + docstring)
                buggy_code=ex["buggy_solution"],   # code to fix
                entry_point=ex.get("entry_point", ""),   # function name
                tests=ex.get("test", ""),                # unit test code
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks
