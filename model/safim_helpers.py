"""Vendored SAFIM evaluation logic for locLLM.

Adapted from gonglinyuan/safim (MIT):
  - prompt splitting / post-processing (prompt_utils.py)
  - tree-sitter AST visitor + syntax checks (ast_utils.py)
  - execution-based evaluation via the ExecEval daemon (exec_utils.py)

Only the parts needed by model/eval.py are included, with the tree-sitter
initialization switched to the modern `tree-sitter` + `tree-sitter-language-pack`
API (Language.build_library is gone in tree-sitter >= 0.22).
"""
from __future__ import annotations

import ast
import re
from typing import Iterable

import requests
from tree_sitter import Parser
from tree_sitter_language_pack import get_language

# ---------------------------------------------------------------------------
# Shared maps (SAFIM -> locLLM)
# ---------------------------------------------------------------------------

# Placeholder for the completion inside each SAFIM prompt, per language.
COMPLETION_PLACEHOLDER = {
    "python": "# TODO: Your code here",
    "java": "/* TODO: Your code here */",
    "cpp": "/* TODO: Your code here */",
    "csharp": "/* TODO: Your code here */",
}

# SAFIM language -> locLLM <lang> tag content. Matches normalize_lang() in
# main_big.py (LANG_ALIASES={"golang": "go", "cpp": "c++"}, lowercased).
LANG_MAP = {
    "python": "python",
    "java": "java",
    "cpp": "c++",
    "csharp": "c#",
}

# SAFIM language -> tree-sitter-language-pack name
_PACK_LANG = {
    "python": "python",
    "java": "java",
    "cpp": "cpp",
    "csharp": "c_sharp",
}

# Which post-processor SAFIM applies for each completion type.
POST_PROCESSORS = {
    "block": ["truncate_line_until_block"],
    "control": ["truncate_control"],
    "api": ["truncate_api_call"],
    "block_v2": ["truncate_line_until_block"],
}


# ---------------------------------------------------------------------------
# Prompt helpers (prompt_utils.py)
# ---------------------------------------------------------------------------

def get_infilling_parts(sample: dict):
    """Split sample['prompt'] on the completion placeholder -> (prefix, suffix)."""
    parts = sample["prompt"].split(COMPLETION_PLACEHOLDER[sample["lang"]])
    assert len(parts) == 2, f"placeholder split failed for {sample['task_id']}"
    return parts


def truncate_to_first_line(code: str) -> str:
    lines = code.splitlines()
    for line in lines:
        if line.strip():
            return line
    return ""


def truncate_control(sample: dict, completion: str) -> str:
    if sample["lang"] == "python":
        return truncate_to_first_line(completion)
    depth = 0
    for i, ch in enumerate(completion):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == -1:
                return completion[:i]
    return completion


def truncate_api_call(completion: str) -> str:
    depth = 0
    for i, ch in enumerate(completion):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth <= 0:
                return completion[:i + 1]
    return completion


def match_prefix_and_suffix(l1, l2):
    p = 0
    while p < len(l1) and p < len(l2):
        if l1[p] == l2[p]:
            p += 1
        else:
            break
    q = 0
    while -q < len(l1) and -q < len(l2):
        if l1[q - 1] == l2[q - 1]:
            q -= 1
        else:
            break
    return p, q


# ---------------------------------------------------------------------------
# Tree-sitter helpers (ast_utils.py, updated init)
# ---------------------------------------------------------------------------

_PARSERS: dict = {}


def get_parser(lang: str) -> Parser:
    if lang not in _PARSERS:
        _PARSERS[lang] = Parser(get_language(_PACK_LANG[lang]))
    return _PARSERS[lang]


class ASTVisitor:
    """DFS visitor over a tree-sitter tree (ported from SAFIM ast_utils.py)."""

    def __init__(self, with_ndtypes=False, print_debug_outputs=False):
        self.with_ndtypes = with_ndtypes
        self.print_debug_outputs = print_debug_outputs
        self.stack = []
        self.ndtypes = []

    def enter(self, node) -> bool:
        return True

    def leave(self, node):
        pass

    def enter_leaf(self, node):
        pass

    def print_stack(self, node):
        depth = len(self.stack)
        print(" " * depth * 2 + node.type)

    def on_enter(self, node) -> bool:
        if self.print_debug_outputs:
            self.print_stack(node)
        if self.with_ndtypes:
            self.ndtypes.append((node.start_byte, True, node.type))
        enter_fn = getattr(self, "enter_%s" % node.type, self.enter)
        r = enter_fn(node)
        if node.child_count == 0:
            self.enter_leaf(node)
        self.stack.append(node.type)
        return r

    def on_leave(self, node):
        assert self.stack.pop() == node.type
        leave_fn = getattr(self, "leave_%s" % node.type, self.leave)
        r = leave_fn(node)
        if self.with_ndtypes:
            self.ndtypes.append((node.end_byte, False, node.type))
        return r

    def walk(self, root_node):
        if root_node is None:
            return

        cursor = root_node.walk()
        has_next = True

        while has_next:
            current_node = cursor.node

            # Step 1: Try to go to next child if we continue the subtree
            if self.on_enter(current_node):
                has_next = cursor.goto_first_child()
            else:
                has_next = False

            # Step 2: Try to go to next sibling
            if not has_next:
                self.on_leave(current_node)
                has_next = cursor.goto_next_sibling()

            # Step 3: Go up until sibling exists
            while not has_next and cursor.goto_parent():
                self.on_leave(cursor.node)  # never return to this parent again
                has_next = cursor.goto_next_sibling()

    def __call__(self, root_node):
        return self.walk(root_node)


class ErrorCheckVisitor(ASTVisitor):
    def __init__(self, with_ndtypes=False):
        super().__init__(with_ndtypes)
        self.error_cnt = 0

    def enter_ERROR(self, node):
        if node.text.decode("utf-8") != ";":
            self.error_cnt += 1


def check_syntax(code: str) -> bool:
    parser = get_parser("python")
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    error_check = ErrorCheckVisitor()
    error_check(tree)
    return error_check.error_cnt == 0


def truncate_line_until_block(sample: dict, code: str) -> str:
    parser = get_parser(sample["lang"])
    lines = code.splitlines(keepends=True)
    while lines:
        eval_prefix, eval_suffix = sample["eval_prompt"].split("{{completion}}")
        eval_prefix = eval_prefix.encode("utf-8")
        eval_suffix = eval_suffix.encode("utf-8")
        completion = "".join(lines).encode("utf-8")
        if sample["lang"] == "python":
            code_bytes_0 = eval_prefix + b"pass" + eval_suffix
        else:
            code_bytes_0 = eval_prefix + eval_suffix
        code_bytes_1 = eval_prefix + completion + eval_suffix

        visitor = ErrorCheckVisitor(with_ndtypes=True)
        tree = parser.parse(code_bytes_1)
        visitor(tree)
        if visitor.error_cnt > 0:
            lines.pop()
            continue
        visitor_trace_1 = [(x, y) for _, x, y in visitor.ndtypes]

        visitor = ErrorCheckVisitor(with_ndtypes=True)
        tree = parser.parse(code_bytes_0)
        visitor(tree)
        assert visitor.error_cnt == 0
        visitor_trace_0 = [(x, y) for _, x, y in visitor.ndtypes]
        if len(visitor_trace_0) > len(visitor_trace_1):
            lines.pop()
            continue

        prefix_matched, suffix_matched = match_prefix_and_suffix(visitor_trace_0, visitor_trace_1)
        matched_diff = len(visitor_trace_0) - (prefix_matched - suffix_matched)
        if sample["lang"] == "python":
            matched_diff -= 4
        if matched_diff == 0:
            break
        else:
            lines.pop()
    return "".join(lines)


def apply_postprocessors(completion: str, sample: dict, completion_type: str,
                         post_processors: Iterable[str]) -> str:
    if post_processors is None:
        post_processors = []
    for post_processor in post_processors:
        if post_processor == "truncate_control":
            completion = truncate_control(sample, completion)
        elif post_processor == "truncate_api_call":
            completion = truncate_api_call(completion)
        elif post_processor == "truncate_line_until_block":
            completion = truncate_line_until_block(sample, completion)
        else:
            raise ValueError(post_processor)
    return completion


# ---------------------------------------------------------------------------
# Scoring (evaluate.py)
# ---------------------------------------------------------------------------

def get_function_call_params(node):
    positional_args = [ast.dump(arg) for arg in node.args]
    keyword_args = {kw.arg: ast.dump(kw.value) for kw in node.keywords}
    return positional_args, keyword_args


def function_calls_match(call1, call2):
    params1 = get_function_call_params(call1)
    params2 = get_function_call_params(call2)
    return params1 == params2


def syntax_match(code1: str, code2: str, lang: str) -> bool:
    code1 = re.sub(r"\s+", "", code1).strip()
    code2 = re.sub(r"\s+", "", code2).strip()
    if lang == "python":
        try:
            tree1 = ast.parse(code1, mode="eval")
            tree2 = ast.parse(code2, mode="eval")
            if isinstance(tree1.body, ast.Call) and isinstance(tree2.body, ast.Call):
                return function_calls_match(tree1.body, tree2.body)
        except Exception:
            pass  # fall back to simple string comparison
    return code1 == code2


# ---------------------------------------------------------------------------
# ExecEval daemon client (exec_utils.py)
# ---------------------------------------------------------------------------

EXEC_OUTCOMES = [
    "EMPTY", "COMPILATION_ERROR", "RUNTIME_ERROR", "MEMORY_LIMIT_EXCEEDED",
    "TIME_LIMIT_EXCEEDED", "WRONG_ANSWER", "MIXED",
]

LANG_TO_RUNTIME = {
    "cpp": "GNU C++17",
    "csharp": "Mono C#",
    "java": "Java 17",
    "python": "PyPy 3",
}


class ExecEvalClient:
    def __init__(self, server_url: str = "http://localhost:5000"):
        self._session = requests.Session()
        self.execute_code_url = f"{server_url}/api/execute_code"
        self.get_runtimes_url = f"{server_url}/api/all_runtimes"

    def close(self):
        self._session.close()

    def available(self, timeout: float = 2.0) -> bool:
        try:
            r = self._session.get(self.get_runtimes_url, timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def execute_code(self, language: str, source_code: str, unittests: list,
                     limits: dict | None = None, block_network: bool = True,
                     stop_on_first_fail: bool = True, use_sanitizer: bool = False,
                     compiler_program_name: str | None = None,
                     compiler_flags: str | None = None,
                     interpreter_cmd: str | None = None,
                     interpreter_flags: str | None = None,
                     sample_id: int | None = None,
                     task_id: str | int | None = None):
        if not language or not source_code or not unittests:
            return [{"exec_outcome": "COMPILATION_ERROR", "result": "",
                     "passed": False, "input": "", "output": []}], sample_id, task_id
        request_body = dict(
            language=language,
            source_code=source_code,
            unittests=unittests,
            limits=limits if isinstance(limits, dict) else None,
            compile_cmd=compiler_program_name,
            compile_flags=compiler_flags,
            execute_cmd=interpreter_cmd,
            execute_flags=interpreter_flags,
            block_network=block_network,
            stop_on_first_fail=stop_on_first_fail,
            use_sanitizer=use_sanitizer,
        )
        try:
            resp = self._session.post(
                self.execute_code_url, json=request_body,
                headers={"Content-Type": "application/json"},
            ).json()
        except (requests.RequestException, ValueError):
            return ([{"exec_outcome": "COMPILATION_ERROR", "result": "",
                      "passed": False}], sample_id, task_id)

        if "data" not in resp:
            return resp, sample_id, task_id
        return resp["data"], sample_id, task_id

    def run_test(self, problem: dict, completion: dict):
        """Run a problem's unit tests against a completion. Returns (result, passed)."""
        assert problem["task_id"] == completion["task_id"]
        code = problem["eval_prompt"].replace("{{completion}}", completion["completion"])
        result = self.execute_code(
            LANG_TO_RUNTIME[problem["lang"]], code, problem["unit_tests"],
            task_id=problem["task_id"],
        )[0]
        if not (isinstance(result, list) and isinstance(result[0], dict)):
            print(result)
            return "COMPILATION_ERROR", False
        for o in result:
            if o.get("result") is not None and len(str(o["result"])) > 1000:
                o["result"] = str(o["result"])[:1000]
        return result, all(o.get("exec_outcome") == "PASSED" for o in result)
