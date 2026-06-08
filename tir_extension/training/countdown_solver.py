"""Brute-force oracle solver for the Countdown arithmetic task.

Given a target integer and a list of numbers, exhaustively search over all
binary expression trees (all orderings, all operator combinations, all
parenthesisations) and return the first valid arithmetic expression found.

Complexity for n numbers:
  - n=3: 3! * 4^2 * 2 structures = 192 nodes  (instant)
  - n=4: 4! * 4^3 * 5 structures ≈ 7680 nodes (< 5 ms)

The recursion produces all placements naturally — no need to enumerate tree
shapes separately since the pair-reduction loop covers every subtree.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


_OPS = ["+", "-", "*", "/"]


def solve_countdown(target: int | float, numbers: list[int]) -> str | None:
    """Return an arithmetic expression that equals *target* using every number
    in *numbers* exactly once, or ``None`` if no solution exists.

    The expression uses only ``+``, ``-``, ``*``, ``/`` and nested parentheses.
    Intermediate divisions that yield non-integers are allowed as long as the
    final result matches the target within 1e-5.
    """
    return _search([float(n) for n in numbers], [str(n) for n in numbers], float(target))


def _strip_outer(s: str) -> str:
    """Remove one balanced outer paren pair if the whole string is wrapped."""
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return s  # outer parens close before the end — don't strip
    return s[1:-1]


def _search(vals: list[float], strs: list[str], target: float) -> str | None:
    if len(vals) == 1:
        return _strip_outer(strs[0]) if abs(vals[0] - target) < 1e-5 else None

    n = len(vals)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = vals[i], vals[j]
            a_str, b_str = strs[i], strs[j]
            rest_vals = [vals[k] for k in range(n) if k != i and k != j]
            rest_strs = [strs[k] for k in range(n) if k != i and k != j]

            for op in _OPS:
                if op == "/" and abs(b) < 1e-9:
                    continue
                if op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                elif op == "*":
                    result = a * b
                else:
                    result = a / b

                combined_str = f"({a_str} {op} {b_str})"
                sol = _search([result] + rest_vals, [combined_str] + rest_strs, target)
                if sol is not None:
                    return sol

    return None


def build_oracle_response(expr: str, target: int | float) -> str:
    """Build a TIR-formatted response using the oracle expression.

    Format:
        <use_tool>calculator: {expr}</use_tool>
        <tool_result>{expr} = {target}</tool_result>
        <answer>{expr}</answer>

    The calculator tool call provides the "check" that lets the model learn
    to verify its own arithmetic before committing an answer.
    """
    target_str = str(int(target)) if float(target) == int(target) else str(target)
    return (
        f"<use_tool>calculator: {expr}</use_tool>"
        f"<tool_result>{expr} = {target_str}</tool_result>"
        f"\n<answer>{expr}</answer>"
    )


if __name__ == "__main__":
    # Quick smoke test.
    cases = [
        (98, [44, 19, 35]),
        (64, [63, 95, 96]),
        (28, [95, 11, 56]),
        (10, [1, 2, 3, 4]),
        (24, [3, 3, 8, 8]),  # classic hard case: 8 / (3 - 8/3) = 24
    ]
    for target, nums in cases:
        sol = solve_countdown(target, nums)
        resp = build_oracle_response(sol, target) if sol else None
        print(f"target={target} nums={nums}  ->  {sol}")
        if resp:
            print(f"  response:\n    {resp}\n")
