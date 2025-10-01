from __future__ import annotations
import ast
import re
import textwrap
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import get_chat_messages


def _extract_function(text: str) -> str:
    """
    Clean Qwen output:
      - Strip <think> blocks
      - Remove code fences
      - Normalize indentation while keeping structure
      - Preserve any safe imports that precede the function definition
    """
    s = text.strip()
    # Remove Qwen reasoning traces
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    # Remove triple backticks
    s = re.sub(r"^```(?:python)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Normalize indentation: replace tabs with 4 spaces
    s = s.replace("\t", "    ")
    try:
        tree = ast.parse(s)
    except SyntaxError:
        return s
    func_node = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if func_node is None:
        return s
    lines = s.splitlines()
    def _slice(node: ast.AST) -> str:
        start_line = node.lineno - 1
        end_line = getattr(node, "end_lineno", None) or start_line + 1
        return "\n".join(lines[start_line:end_line]).strip()
    pieces: List[str] = []
    for node in tree.body:
        if node is func_node:
            break
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            pieces.append(_slice(node))
    start_idx = func_node.lineno - 1
    end_lineno = getattr(func_node, "end_lineno", None)
    if end_lineno is None:
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        end_idx = start_idx + 1
        for idx in range(start_idx + 1, len(lines)):
            line = lines[idx]
            if not line.strip():
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent:
                end_idx = idx
                break
        else:
            end_idx = len(lines)
    else:
        end_idx = end_lineno
    func_block = "\n".join(lines[start_idx:end_idx])
    pieces.append(textwrap.dedent(func_block).strip())
    return '\n\n'.join(piece for piece in pieces if piece)
_TypingNames = {"List", "Dict", "Set", "Tuple", "Optional", "Sequence", "Iterable"}
_SafeModules = {"re", "math", "collections"}
def _ensure_support_imports(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    existing_modules: set[str] = set()
    existing_from: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = (alias.name or "").split('.', 1)[0]
                existing_modules.add(top)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = existing_from.setdefault(node.module, set())
            for alias in node.names:
                mods.add(alias.name)
    typing_needed: set[str] = set()
    module_needed: set[str] = set()
    class UsageVisitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if node.id in _TypingNames:
                typing_needed.add(node.id)
            if node.id in _SafeModules:
                module_needed.add(node.id)
            self.generic_visit(node)
        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id in _SafeModules:
                module_needed.add(node.value.id)
            self.generic_visit(node)
    UsageVisitor().visit(tree)
    additions: list[str] = []
    if typing_needed:
        if 'typing' not in existing_modules:
            current = existing_from.get('typing', set())
            missing = sorted(typing_needed - current)
            if missing:
                additions.append(f"from typing import {', '.join(missing)}")
    for module in sorted(module_needed):
        if module in existing_modules:
            continue
        imported = existing_from.get(module, set())
        if imported:
            continue
        additions.append(f"import {module}")
    if not additions:
        return code
    lines = code.splitlines()
    return "\n".join(additions + [""] + lines)


def _strip_asserts(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class _AssertStripper(ast.NodeTransformer):
        def visit_Assert(self, node: ast.Assert):
            return None

    new_tree = _AssertStripper().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        return code

class QwenFixer:
    """
    CPU-friendly default. For speed on CPU, we use a small instruct model:
    - Qwen/Qwen2.5-0.5B-Instruct (recommended)
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",        # keep it explicit on CPU
            torch_dtype=torch.float32,  # stable on CPU
        )
    def __call__(self, spec: str, buggy_code: str, entry_point: str | None = None) -> str:
        messages: List[Dict[str, str]] = get_chat_messages(spec, buggy_code, entry_point=entry_point, include_few_shot=False)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        gen = self.model.generate(
            **inputs,
            max_new_tokens=384,          # keep this modest for speed
            do_sample=False,             # deterministic for pass@1
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        out = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cleaned = _extract_function(out)
        enriched = _ensure_support_imports(cleaned)

        return _strip_asserts(enriched)
