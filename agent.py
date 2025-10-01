from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class RepairTask:
    task_id: str
    spec: str            # e.g., dataset["prompt"]
    buggy_code: str      # e.g., dataset["buggy_solution"]
    entry_point: str     # e.g., dataset["entry_point"]
    tests: str           # e.g., dataset["test"]


@dataclass
class RepairResult:
    task_id: str
    candidate_code: str
    passed: bool
    error: Optional[str] = None

#Interfaces the agent will use

# 1) LLM "fixer": spec + buggy -> fixed function (as text)
LLMFixer = Callable[[str, str, str | None], str]

# 2) Runner/Evaluator: execute candidate + tests in a sandbox and return pass/fail + error
SandboxEval = Callable[[str, str, str], tuple[bool, Optional[str]]]
# signature: (candidate_code, tests, entry_point) -> (passed, error_message)


class RepairAgent:
    """
    Minimal orchestrator:
    - builds a prompt
    - calls an LLM fixer
    - runs the test harness in a sandbox
    """
    def __init__(self, fixer: LLMFixer, evaluator: SandboxEval):
        self.fixer = fixer
        self.evaluator = evaluator

    def run_once(self, task: RepairTask) -> RepairResult:
        # Later: use your prompting.build_repair_prompt()
        fixed_code = self.fixer(task.spec, task.buggy_code, task.entry_point)
        passed, err = self.evaluator(fixed_code, task.tests, task.entry_point)
        return RepairResult(
            task_id=task.task_id,
            candidate_code=fixed_code,
            passed=passed,
            error=err
        )
    

