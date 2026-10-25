r"""leakage/scoring.py -- answer extraction and grading for the competition-math test set.

    pred        content of the LAST complete \boxed{...} in the response ("" if there is none;
                a box cut off by truncation falls back to the previous complete one)
    correct     pred == ground_truth, otherwise Math-Verify equivalence with both sides parsed
                as LaTeX (wrapped in $...$)

This is the grader behind the released rollouts (huggingface.co/datasets/zml5418/thinking-leakage-rollouts).

`last_boxed_only_string` / `remove_boxed` are the plain whole-string helpers that
leakage/directions.py uses to compare the in-<think> answer with the final one; they are not part
of grading.
"""
from typing import Optional

from math_verify import parse, verify

_cache = {}


def last_boxed_only_string(string: str) -> Optional[str]:
    """The last "\\boxed{...}" substring, "" if its braces never close, None if there is none."""
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None
    depth = 0
    for i in range(idx, len(string)):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                return string[idx:i + 1]
    return ""


def remove_boxed(s: str) -> str:
    left = r"\boxed{"
    return s[len(left):-1] if s.startswith(left) and s.endswith("}") else ""


def extract_answer(response: str) -> str:
    """Content of the last complete \\boxed{...} anywhere in the response."""
    left = r"\boxed{"
    i = response.rfind(left)
    while i != -1:
        j, depth = i + len(left), 1
        while j < len(response) and depth:
            depth += {"{": 1, "}": -1}.get(response[j], 0)
            j += 1
        if depth == 0:
            return response[i + len(left):j - 1].strip()
        i = response.rfind(left, 0, i)
    return ""


def is_equivalent(pred: str, ground_truth: str, timeout_seconds: int = 30) -> bool:
    if pred == "":
        return False
    if pred == ground_truth:
        return True
    key = (pred, ground_truth)
    if key not in _cache:
        try:
            _cache[key] = bool(verify(parse(f"${ground_truth}$"), parse(f"${pred}$"),
                                      timeout_seconds=timeout_seconds))
        except Exception:
            _cache[key] = False
    return _cache[key]


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    """{'score', 'acc', 'pred'}; score == acc in {0.0, 1.0}. data_source/extra_info are accepted for
    call-site compatibility and unused: all four test sets are graded the same way."""
    pred = extract_answer(solution_str)
    acc = 1.0 if is_equivalent(pred, str(ground_truth)) else 0.0
    return {"score": acc, "acc": acc, "pred": pred}
