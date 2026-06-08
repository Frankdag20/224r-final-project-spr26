"""Pool of callable tools exposed to the RLOO policy.

Six tools are registered, split evenly between Countdown-relevant utilities
and intentionally irrelevant distractors. The irrelevant tools act as a
control: DSPy should learn to *not* recommend them, and the RLOO reward
penalises the policy for invoking them.

Invocation format (used in both policy outputs and SFT trajectories):

    <use_tool>tool_name: tool_input</use_tool>
    <tool_result>tool_output</tool_result>

The orchestrator generates a full response with the model, then this module
fills in any empty/missing ``<tool_result>...</tool_result>`` spans by
executing the corresponding tool deterministically.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable


# ---------------------------------------------------------------------------
# Relevant tools (helpful for the Countdown task).
# ---------------------------------------------------------------------------

# Restrict expressions to the same character whitelist used in
# evaluation/countdown.py so models can't smuggle code through the tool.
_ARITHMETIC_PATTERN = re.compile(r"^[\d+\-*/().\s]+$")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return its numeric result."""
    expression = expression.strip()
    if not expression:
        return "error: empty expression"
    if not _ARITHMETIC_PATTERN.match(expression):
        return "error: invalid characters in expression"
    try:
        result = eval(expression, {"__builtins__": None}, {})
    except Exception as exc:  # noqa: BLE001 - tool surface returns strings
        return f"error: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def number_tracker(tool_input: str) -> str:
    """Report which available numbers have not yet been used.

    Expected ``tool_input`` formats:
        "available: 1 2 3 4 | used: 1 3"
        "available=[1,2,3] used=[2]"
    """
    available = _extract_int_list(tool_input, key="available")
    used = _extract_int_list(tool_input, key="used")
    if available is None:
        return "error: could not parse 'available' numbers"
    if used is None:
        used = []

    remaining = list(available)
    for value in used:
        if value in remaining:
            remaining.remove(value)
    if not remaining:
        return "all numbers used"
    return "remaining: " + " ".join(str(v) for v in remaining)


def running_total(tool_input: str) -> str:
    """Sum a sequence of partial results to track progress toward the target.

    Input is parsed as any whitespace/comma-separated numbers. Returns the
    cumulative sum followed by the running totals after each step.
    """
    numbers = _extract_floats(tool_input)
    if not numbers:
        return "error: no numbers found"
    totals = []
    acc = 0.0
    for value in numbers:
        acc += value
        totals.append(_format_number(acc))
    return f"total: {_format_number(acc)} | steps: {' -> '.join(totals)}"


# ---------------------------------------------------------------------------
# Irrelevant tools (distractors).
# ---------------------------------------------------------------------------


def factorial(tool_input: str) -> str:
    """Compute n! for a single non-negative integer."""
    parsed = _extract_int(tool_input)
    if parsed is None or parsed < 0:
        return "error: expected a single non-negative integer"
    if parsed > 20:
        return "error: n must be <= 20 to avoid overflow"
    result = 1
    for k in range(2, parsed + 1):
        result *= k
    return str(result)


def is_prime(tool_input: str) -> str:
    """Return 'true' / 'false' depending on whether n is prime."""
    n = _extract_int(tool_input)
    if n is None:
        return "error: expected a single integer"
    if n < 2:
        return "false"
    if n % 2 == 0:
        return "true" if n == 2 else "false"
    k = 3
    while k * k <= n:
        if n % k == 0:
            return "false"
        k += 2
    return "true"


def fibonacci(tool_input: str) -> str:
    """Return the n-th Fibonacci number (0-indexed)."""
    n = _extract_int(tool_input)
    if n is None or n < 0:
        return "error: expected a single non-negative integer"
    if n > 200:
        return "error: n must be <= 200"
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return str(a)


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


TOOL_REGISTRY: dict[str, dict] = {
    "calculator": {
        "fn": calculator,
        "description": (
            "Evaluate an arithmetic expression like '3 + 4 * 2'. "
            "Only +, -, *, /, parentheses, and integers are allowed."
        ),
        "relevant": True,
    },
    "number_tracker": {
        "fn": number_tracker,
        "description": (
            "Track which of the available Countdown numbers have already been "
            "used. Pass 'available: 1 2 3 4 | used: 1 3'."
        ),
        "relevant": True,
    },
    "running_total": {
        "fn": running_total,
        "description": (
            "Cumulatively add a sequence of numbers and show the running sum "
            "after each step. Pass whitespace/comma-separated numbers. "
            "Example: '35 44 19' returns 'total: 98 | steps: 35 -> 79 -> 98'."
        ),
        "relevant": True,
    },
    "factorial": {
        "fn": factorial,
        "description": "Compute n! for a non-negative integer (n <= 20).",
        "relevant": False,
    },
    "is_prime": {
        "fn": is_prime,
        "description": "Return 'true' or 'false' for whether an integer is prime.",
        "relevant": False,
    },
    "fibonacci": {
        "fn": fibonacci,
        "description": "Return the n-th Fibonacci number (0-indexed, n <= 200).",
        "relevant": False,
    },
}


def tool_names() -> list[str]:
    return list(TOOL_REGISTRY.keys())


def relevant_tool_names() -> set[str]:
    return {name for name, meta in TOOL_REGISTRY.items() if meta["relevant"]}


def irrelevant_tool_names() -> set[str]:
    return {name for name, meta in TOOL_REGISTRY.items() if not meta["relevant"]}


def get_tool(name: str) -> Callable[[str], str] | None:
    entry = TOOL_REGISTRY.get(name)
    return entry["fn"] if entry else None


def format_tools_for_prompt(active_tools: Iterable[str]) -> str:
    """Render a human-readable tool catalogue for SFT prompts / DSPy inputs."""
    lines = []
    for name in active_tools:
        meta = TOOL_REGISTRY.get(name)
        if meta is None:
            continue
        lines.append(f"- {name}: {meta['description']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing + execution helpers.
# ---------------------------------------------------------------------------

# Matches: <use_tool>name: input</use_tool> with optional whitespace.
_USE_TOOL_PATTERN = re.compile(
    r"<use_tool>\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*</use_tool>",
    re.DOTALL,
)

# Matches a <use_tool>...</use_tool> immediately (optionally followed by
# whitespace) by a <tool_result>...</tool_result>. Used to detect tool calls
# that already have a result so we don't double-execute.
_USE_THEN_RESULT_PATTERN = re.compile(
    r"(<use_tool>\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*</use_tool>)"
    r"(\s*<tool_result>(.*?)</tool_result>)?",
    re.DOTALL,
)


def parse_tool_calls(response: str) -> list[tuple[str, str]]:
    """Extract every ``(tool_name, tool_input)`` pair from ``response``."""
    return [
        (match.group(1), match.group(2))
        for match in _USE_TOOL_PATTERN.finditer(response)
    ]


def execute_tool_calls(
    response: str,
    active_tools: Iterable[str] | None = None,
    max_calls: int = 16,
) -> str:
    """Fill in ``<tool_result>...</tool_result>`` spans for each tool call.

    - Any tool call that does not yet have a following ``<tool_result>``
      block has one appended with the deterministic output.
    - Existing ``<tool_result>`` spans are left untouched so the policy can
      condition on them during training.
    - Disabled tools (not in ``active_tools``) emit an error result so the
      policy is discouraged from calling them but the format stays valid.
    """
    active = set(active_tools) if active_tools is not None else set(TOOL_REGISTRY)
    output_parts: list[str] = []
    cursor = 0
    calls_done = 0

    for match in _USE_THEN_RESULT_PATTERN.finditer(response):
        use_tool_block = match.group(1)
        tool_name = match.group(2)
        tool_input = match.group(3) or ""
        result_block = match.group(4)

        output_parts.append(response[cursor : match.start()])
        output_parts.append(use_tool_block)

        if result_block is not None:
            output_parts.append(result_block)
        else:
            if calls_done >= max_calls:
                tool_output = "error: tool call limit reached"
            elif tool_name not in TOOL_REGISTRY:
                tool_output = f"error: unknown tool '{tool_name}'"
            elif tool_name not in active:
                tool_output = f"error: tool '{tool_name}' is disabled"
            else:
                fn = get_tool(tool_name)
                try:
                    tool_output = fn(tool_input)
                except Exception as exc:  # noqa: BLE001
                    tool_output = f"error: {exc}"
            output_parts.append(f"<tool_result>{tool_output}</tool_result>")
            calls_done += 1

        cursor = match.end()

    output_parts.append(response[cursor:])
    return "".join(output_parts)


def find_tool_result_spans(response: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character offsets of every ``<tool_result>`` span.

    Used by the TIR tokenizer to build a per-token mask so tool-output tokens
    are excluded from the policy-gradient loss.
    """
    pattern = re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL)
    return [(m.start(), m.end()) for m in pattern.finditer(response)]


# ---------------------------------------------------------------------------
# Tiny parsing utilities.
# ---------------------------------------------------------------------------


def _extract_int(text: str) -> int | None:
    matches = re.findall(r"-?\d+", text)
    if not matches:
        return None
    try:
        return int(matches[0])
    except ValueError:
        return None


def _extract_int_list(text: str, key: str | None = None) -> list[int] | None:
    if key is not None:
        # Look for "key: ..." or "key=[...]" specifically.
        m = re.search(
            rf"{re.escape(key)}\s*[:=]\s*\[?([\-\d,\s]+)\]?",
            text,
        )
        if m is None:
            return None
        segment = m.group(1)
    else:
        segment = text
    values = re.findall(r"-?\d+", segment)
    if not values:
        return None
    return [int(v) for v in values]


def _extract_floats(text: str) -> list[float]:
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"
