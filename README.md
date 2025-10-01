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

Configuration

By default, scripts run on CPU. Expect ~2 GB RAM usage for the Qwen 0.5B model plus extra headroom for the sandbox worker.
Optionally, set HF_HOME or TRANSFORMERS_CACHE to control cache location for models/datasets.

