"""Build the tool-integrated system prompt injected into student-model prompts.

This is used during RLOO sampling and evaluation so the policy model knows
which tools are available and how to call them.  The prompt mirrors the
format used by ``generate_trajectories.py`` for the teacher model so the
SFT warm-start and RL phases see a consistent interface.
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
        "You have access to the following tools to help solve Countdown "
        "problems. You may call them during your reasoning.\n\n"
        "Tool call format:\n"
        "    <use_tool>tool_name: your_input</use_tool>\n"
        "The system will execute the tool and insert a result:\n"
        "    <tool_result>result</tool_result>\n"
        "You can then continue reasoning based on the result.\n\n"
        f"Available tools ({tool_names_str}):\n"
        f"{tool_catalogue}\n\n"
        "Always end your response with the final equation wrapped as:\n"
        "    <answer>EXPRESSION</answer>"
    )
