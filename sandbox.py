from __future__ import annotations
import ast
import builtins as _py_builtins
import multiprocessing as mp
import textwrap
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import atexit


_BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "pathlib", "shutil", "socket", "pickle",
    "importlib", "ctypes", "multiprocessing", "signal", "resource",
}

_orig_import = _py_builtins.__import__

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """A guarded __import__ for candidate code."""
    top = name.split(".", 1)[0]
    if top in _BLOCKED_IMPORTS:
        raise ImportError(f"Blocked import: {top}")
    return _orig_import(name, globals, locals, fromlist, level)


_ALLOWED_BUILTINS = {
    # types & basics
    "bool", "int", "float", "str", "list", "tuple", "dict", "set",
    # iter/sequence ops
    "len", "range", "enumerate", "zip", "reversed", "sorted",
    "sum", "min", "max", "all", "any", "map", "filter",
    # mathy helpers
    "abs", "pow", "round", "divmod",
    # conversions / chars
    "bin", "ord", "chr",
    # checks
    "isinstance", "type", "hash",
    # slicing helper
    "slice",
    # harmless; we instruct the LLM not to use prints anyway
    "print",
    # NOTE: __import__ is provided via _safe_import (below)
}


def _secure_builtins():
    """Return a restricted builtins dict for candidate code."""
    b = {name: getattr(_py_builtins, name) for name in _ALLOWED_BUILTINS}
    b["__import__"] = _safe_import
    return b


#AST screen for CANDIDATE code

def _reject_forbidden_nodes(code: str) -> Optional[str]:
    """
    Disallow obviously dangerous constructs in the candidate code:
      - Global / Nonlocal
      - Imports of explicitly blocked modules
    We *allow* stdlib imports (math, collections, itertools, etc.) via _safe_import.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return f"Forbidden construct: {type(node).__name__}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # If any alias targets a blocked top-level module, reject
            for alias in node.names:
                top = (alias.name or "").split(".", 1)[0]
                if top in _BLOCKED_IMPORTS:
                    return f"Forbidden import: {top}"
    return None


#global single-worker pool to avoid spawning per task on Windows
_EXECUTOR: ProcessPoolExecutor | None = None

def _get_executor() -> ProcessPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        ctx = mp.get_context("spawn")
        _EXECUTOR = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    return _EXECUTOR


def _shutdown_executor():
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(cancel_futures=True)
        _EXECUTOR = None


atexit.register(_shutdown_executor)

# 4) Subprocess worker
"""
def _worker(code: str, tests: str, entry_point: str, conn) -> None:
    
    #Child process:
      #1) Exec candidate under restricted builtins (with safe __import__)
      #2) Exec tests under FULL builtins (tests may import stdlib freely)
    #Sends (passed: bool, err: Optional[str]).
    
    try:
        # 1) Candidate env
        globals_dict = {"__builtins__": _secure_builtins()}
        locals_dict = {}
        exec(compile(code, "<candidate>", "exec"), globals_dict, locals_dict)

        # Ensure entry point present/callable
        fn = globals_dict.get(entry_point) or locals_dict.get(entry_point)
        if not callable(fn):
            conn.send((False, f"Entry point '{entry_point}' not found or not callable"))
            conn.close()
            return

        # 2) Tests with FULL builtins so they can import whatever they need
        test_globals = globals_dict.copy()
        test_globals["__builtins__"] = _py_builtins

        test_code = textwrap.dedent(tests)
        exec(compile(test_code, "<tests>", "exec"), test_globals, locals_dict)

        conn.send((True, None))
        conn.close()
    except Exception as e:
        conn.send((False, f"{type(e).__name__}: {e}"))
        conn.close()
"""

def _worker_func(code: str, tests: str, entry_point: str) -> Tuple[bool, Optional[str]]:
    """
    Worker body for ProcessPoolExecutor. This is the same as your _worker,
    but simplified because we don't need a Pipe anymore.
    """
    try:
        globals_dict = {"__builtins__": _secure_builtins()}
        locals_dict = {}
        exec(compile(code, "<candidate>", "exec"), globals_dict, locals_dict)

        fn = globals_dict.get(entry_point) or locals_dict.get(entry_point)
        if not callable(fn):
            return False, f"Entry point '{entry_point}' not found or not callable"

        test_globals = globals_dict.copy()
        test_globals["__builtins__"] = _py_builtins

        test_code = textwrap.dedent(tests)
        exec(compile(test_code, "<tests>", "exec"), test_globals, locals_dict)

        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

#Public API

def evaluate_candidate(
    candidate_code: str,
    tests: str,
    entry_point: str,
    timeout_sec: float = 10.0,
) -> Tuple[bool, Optional[str]]:
    """
    Execute candidate + tests in a separate, reusable process with a timeout.
    Reuses a singleton ProcessPoolExecutor to avoid repeated CreateProcess calls.
    """
    err = _reject_forbidden_nodes(candidate_code)
    if err:
        return False, err

    try:
        executor = _get_executor()
        future = executor.submit(_worker_func, candidate_code, tests, entry_point)
        return future.result(timeout=timeout_sec)
    except TimeoutError:
        return False, f"Timeout after {timeout_sec:.1f}s"
    except Exception as e:
        # Catch-all for executor errors (including broken pools)
        _shutdown_executor()
        executor = _get_executor()
        future = executor.submit(_worker_func, candidate_code, tests, entry_point)
        try:
            return future.result(timeout=timeout_sec)
        except TimeoutError:
            return False, f"Timeout after {timeout_sec:.1f}s"
        except Exception as e2:
            return False, f"ExecutorError: {type(e2).__name__}: {e2}"

