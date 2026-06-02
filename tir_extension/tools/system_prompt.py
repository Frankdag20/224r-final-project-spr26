"""Build the tool-integrated system prompt (I^T) for student-model prompts.

This is the tool-integrated instruction described in Tool-Star (eq. 1).
It must be prepended consistently across ALL stages: SFT, RLOO, and evaluation,
so the model always knows what tools are available and how to call them.
"""

from __future__ import annotations

from typing import Iterable

from tir_extension.tools.tool_pool import format_tools_for_prompt, relevant_tool_names


def build_tool_system_prompt(active_tools: Iterable[str] | None = None) -> str:
    """Return the system instruction prepended to every Countdown prompt.

    Args:
        active_tools: Tool names to advertise. Defaults to all relevant tools.
    """
    if active_tools is None:
        active_tools = sorted(relevant_tool_names())
    else:
        active_tools = sorted(active_tools)

    tool_catalogue = format_tools_for_prompt(active_tools)
    tool_names_str = ", ".join(active_tools)

    return (
        "You are solving the Countdown arithmetic game. You must combine the "
        "given numbers using +, -, *, / (each number used exactly once) to "
        "reach the target.\n\n"
        f"You have access to the following tools ({tool_names_str}) to assist "
        "your reasoning. You may call them at any point during your solution.\n\n"
        "Tool call format — write EXACTLY this and then stop; the system will "
        "execute the tool and insert a <tool_result> block which you should "
        "read before continuing:\n"
        "    <use_tool>tool_name: your_input</use_tool>\n"
        "    <tool_result>result_goes_here</tool_result>\n\n"
        "Worked example (target 24, numbers [3, 8]):\n"
        "    I'll verify my candidate equation with the calculator.\n"
        "    <use_tool>calculator: 3 * 8</use_tool>\n"
        "    <tool_result>24</tool_result>\n"
        "    The result is 24, which matches the target.\n"
        "    <answer>3 * 8</answer>\n\n"
        f"Available tools:\n"
        f"{tool_catalogue}\n\n"
        "Always end your response with the final equation wrapped as:\n"
        "    <answer>EXPRESSION</answer>\n"
        "where EXPRESSION uses only +, -, *, /, parentheses, and the given "
        "numbers (each used exactly once)."
    )
