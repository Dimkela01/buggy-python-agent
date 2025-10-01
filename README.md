
# Buggy Python Agent

LLM-powered repair agent for the [HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack) dataset. The project stitches together a lightweight prompting layer, a quantized Qwen model, and a sandboxed runner so you can evaluate pass@1 performance on defective reference implementations.

## Features

- 🔧 **LLM fixer** – Uses `Qwen/Qwen2.5-0.5B-Instruct` to propose deterministic patches that respect the original signature.
- 🧪 **Secure sandbox** – Executes candidates in an isolated process with a locked-down builtins table and configurable timeout.
- 🔁 **LangGraph workflow** – Two-node graph (`fix → run`) orchestrates the repair cycle while keeping state traceable.
- 📊 **Eval scripts** – Run quick development slices or full test-suite sweeps and collect JSON summaries for analysis.

## Project Layout

agent.py # Minimal orchestration primitives (tasks, results, agent class)
eval_pass1_val.py # Dev-sized evaluation over the first 20 HumanEvalPack samples
eval_pass1_test.py # Extended evaluation across the full test split
graph_agent.py # LangGraph state machine wiring fixer → sandbox runner
prompt.py # Strict system/user prompts to keep generations compliant
qwen_fixer.py # Qwen model wrapper and output post-processing helpers
sandbox.py # Hardened execution harness using ProcessPoolExecutor
requirements.txt # Python package dependencies
results_val.json # Example evaluation output (dev slice)
results_final.json # Example evaluation output (full sweep)


## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) Git with LFS if you plan to pull larger checkpoints locally
- Access to Hugging Face Hub (the dataset and model live there)

### Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
Tip: Run huggingface-cli login ahead of time if the dataset/model require authentication.

Configuration
The scripts default to CPU execution. Expect ~2 GB RAM usage for the Qwen 0.5B model plus additional headroom for the sandbox worker.
Adjust HF_HOME or TRANSFORMERS_CACHE if you want to control the cache location for models/datasets.
Running Evaluations
Quick Development Slice
python eval_pass1_val.py
Processes the first 20 tasks.
Prints per-task pass/fail and writes detailed records to results_val.json.
Full Test Sweep
python eval_pass1_test.py
Iterates over the remaining tasks (start=20 onward).
Saves cumulative results to results_final.json.
Resource note: The full sweep is CPU-heavy. Consider breaking the loop into chunks or running on a machine with ample RAM/CPU time to avoid OS watchdog resets.

Inspecting Results
Each entry in the results JSON files includes:

{
  "task_id": "HumanEval/__some_id",
  "candidate_code": "def foo(...): ...",
  "passed": true,
  "error": null
}
Aggregate stats (pass@1, evaluated count) are printed to stdout at the end of each run.

Customization
Swap in a different fixer by modifying QwenFixer in eval_pass1_val.py / eval_pass1_test.py.
Tweak prompt behavior (prompt.py) or add few-shot examples by toggling include_few_shot.
Adjust sandbox timeout via the timeout_sec parameter in sandbox.evaluate_candidate.
Troubleshooting
Session resets / crashes: Reduce the number of samples per run or switch to a smaller/quantized model. Running on a GPU or server-class CPU helps.
Hugging Face download failures: Ensure huggingface_hub is logged in and network access is permitted.
Import blocked errors: The sandbox rejects imports from os, sys, subprocess, etc. Modify _BLOCKED_IMPORTS in sandbox.py only if you fully trust the generated code.
License
Add the appropriate license information here (e.g., MIT, Apache 2.0). If you borrowed code from other projects, ensure their licenses are compatible and acknowledged.

Natural next step: create `README.md` in the repo root, paste this content, and adjust the license section to match your project.
