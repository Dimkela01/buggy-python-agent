# Buggy Python Agent

LLM-powered repair agent for the [HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack) dataset. The project combines a lightweight prompting layer, a quantized Qwen model, and a sandboxed runner to evaluate **pass@1** performance on buggy reference implementations.

---

## Features

- 🔧 **LLM fixer** – Uses `Qwen/Qwen2.5-0.5B-Instruct` to propose deterministic patches that respect the original function signature.
- 🧪 **Secure sandbox** – Executes candidate code in an isolated process with a restricted builtins table and configurable timeout.
- 🔁 **LangGraph workflow** – Two-node graph (`fix → run`) orchestrates the repair cycle while keeping state traceable.
- 📊 **Evaluation scripts** – Run quick dev slices or full test-suite sweeps and collect JSON summaries for analysis.

---

## Project Layout

- `agent.py` – Minimal orchestration primitives (tasks, results, agent class)
- `eval_pass1_val.py` – Development-sized evaluation over the first 20 HumanEvalPack samples
- `eval_pass1_test.py` – Extended evaluation across the full test split
- `graph_agent.py` – LangGraph state machine wiring fixer → sandbox runner
- `prompt.py` – Strict system/user prompts to keep model generations compliant
- `qwen_fixer.py` – Qwen model wrapper and output post-processing helpers
- `sandbox.py` – Hardened execution harness using `ProcessPoolExecutor`
- `requirements.txt` – Python package dependencies
- `results_val.json` – Example evaluation output (dev slice)
- `results_final.json` – Example evaluation output (full sweep)

---

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) Git with LFS if you plan to pull larger checkpoints locally
- Access to Hugging Face Hub (dataset and model)

### Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Tip: Run huggingface-cli login if the dataset or model requires authentication.

### Configuration

By default, scripts run on CPU. Expect ~2 GB RAM usage for the Qwen 0.5B model plus extra headroom for the sandbox worker.
Optionally, set HF_HOME or TRANSFORMERS_CACHE to control cache location for models/datasets.

## Running Evaluations
### Quick Development Slice 
```
python eval_pass1_val.py
```

- Processes the first 20 tasks (dev slice)
- Prints per-task pass/fail
- Writes detailed records to results_val.json

Full Test Sweep
```
python eval_pass1_test.py
```

- Iterates over remaining tasks (start=20 onward)
- Saves cumulative results to results_final.json

Resource note: The full sweep is CPU-heavy. Consider running in smaller chunks or on a machine with sufficient RAM/CPU.

## Inspecting Results

Each entry in the results JSON files includes:

{
  "task_id": "HumanEval/__some_id",
  "candidate_code": "def foo(...): ...",
  "passed": true,
  "error": null
}


passed indicates whether the model successfully fixed the code.

candidate_code is the model-generated solution.

error contains any exception or timeout info.

Aggregate statistics like pass@1 and number of evaluated tasks are printed to stdout at the end of each run.

### Customization

Swap in a different fixer by modifying QwenFixer in eval_pass1_val.py or eval_pass1_test.py.

Tweak prompt behavior in prompt.py, or add few-shot examples via the include_few_shot flag.

Adjust sandbox timeout via the timeout_sec parameter in sandbox.evaluate_candidate.

### Troubleshooting

Session resets / crashes: Reduce the number of samples per run, or switch to a smaller/quantized model. Using GPU or server-class CPU is recommended.

Hugging Face download failures: Ensure you are logged in (huggingface-cli login) and have network access.

Import blocked errors: The sandbox rejects imports from os, sys, subprocess, etc. Only modify _BLOCKED_IMPORTS in sandbox.py if you fully trust the generated code.

## License

Add your license here (e.g., MIT, Apache 2.0). Ensure any code borrowed from other projects is compliant with their licenses.





