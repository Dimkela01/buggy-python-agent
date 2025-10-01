from __future__ import annotations
from textwrap import dedent
from typing import List, Dict, Sequence


SYSTEM_PROMPT = dedent("""
You are a Python bug-fixing assistant.
Given a specification (docstring + signature) and a buggy implementation,
produce a corrected function that satisfies the spec.

STRICT RULES:
- Keep the EXACT same function name and signature.
- Deterministic only: no prints, input(), randomness, file/network access, or global state.
- Do not import external modules (only built-ins implicitly available).
- Return a SINGLE self-contained function definition.
- Do NOT include explanations, comments, or backticks. Output ONLY valid Python code.
- **Ensure the function body is indented correctly according to Python standards** (indent with spaces, not tabs).
""").strip()


# Optional few-shot examples
_FEW_SHOT = [
    {
        "spec": "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"\n",
        "buggy": "def add(a: int, b: int) -> int:\n    return a - b\n",
        "fix":   "def add(a: int, b: int) -> int:\n    return a + b\n",
    },
    {
        "spec": "def is_palindrome(s: str) -> bool:\n"
                "    \"\"\"Return True if s reads the same forward and backward.\"\"\"\n",
        "buggy": "def is_palindrome(s: str) -> bool:\n"
                 "    return s == s[::-2]\n",
        "fix":   "def is_palindrome(s: str) -> bool:\n"
                 "    return s == s[::-1]\n",
    },
]

#Build the user prompt for a code-repair task

def build_repair_prompt(
    spec: str,
    buggy_code: str,
    entry_point: str | None = None,
    include_few_shot: bool = False,
    blocked_modules: Sequence[str] = ("os", "sys", "subprocess", "pathlib", "shutil", "socket", "pickle", "importlib", "ctypes", "multiprocessing", "signal", "resource"),
) -> str:
    """
    Build a strict user prompt for code repair.

    Notes for the model (summarized in REQUIREMENTS below):
    - Output ONLY a single, valid Python function definition (no markdown fences, no prose).
    - Keep EXACT same function name and signature (and parameter names/types).
    - Deterministic: no prints, input(), randomness, file/network IO, or global state.
    - If the buggy code already uses safe stdlib imports (e.g., math, itertools), KEEP them.
      Do NOT import any of: {blocked_modules}.
    - Do not add extra helper functions or classes unless strictly necessary.
    - Prefer simple clear logic; handle obvious edge cases from the spec/examples.
    """

    def _few_shot_block() -> str:
        examples: List[str] = []
        if include_few_shot:
            for ex in _FEW_SHOT:
                examples.append(
                    "SPECIFICATION:\n"
                    f"{ex['spec'].strip()}\n\n"
                    "BUGGY IMPLEMENTATION:\n"
                    f"{ex['buggy'].strip()}\n\n"
                    "CORRECTED FUNCTION:\n"
                    f"{ex['fix'].strip()}\n"
                )
        return "\n".join(examples)

    entry_line = f"- The function to return is `{entry_point}`.\n" if entry_point else ""
    blocked_line = ", ".join(blocked_modules)

    prompt = dedent(f"""
    {_few_shot_block()}
    SPECIFICATION:
    {spec.strip()}

    BUGGY IMPLEMENTATION:
    {buggy_code.strip()}

    REQUIREMENTS:
    - Keep the EXACT same function name and signature. {entry_line}- Keep parameter names unchanged.
    - Deterministic; no prints, input(), randomness, file/network access, or global state.
    - **Ensure that the function body is indented correctly with spaces. Do not mix tabs and spaces.**
    - If the buggy code already imports safe stdlib modules (e.g., math, itertools), KEEP those imports.
      Do NOT import any of: {blocked_line}.
    - If the fix needs a safe stdlib helper (e.g., re, math, collections), add the import directly above the function.
    - Do not add extra helper functions or classes unless strictly necessary.
    - Do not change behavior beyond the specification and examples; fix the logic so the function actually satisfies the spec.
    - Avoid adding runtime assertions or type checks that rely on typing generics; focus on correct logic instead.
    - Output ONLY the corrected function as valid Python code (no comments, no backticks, no prose).

    CORRECTED FUNCTION:
    """).strip()

    return prompt


def get_chat_messages(
    spec: str,
    buggy_code: str,
    entry_point: str | None = None,
    include_few_shot: bool = False,
) -> List[Dict[str, str]]:
    """
    Return a messages list suitable for chat LLMs.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_repair_prompt(spec, buggy_code, entry_point=entry_point, include_few_shot=include_few_shot)},
    ]
