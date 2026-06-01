"""Hierarchical reward design adapted from Tool-Star for Countdown TIR.

Reward formula (from eq. 3 in the Tool-Star paper):

    R = max(Acc + rM, Acc)   if Format is Good & Acc > 0
    R = 0                    if Format is Good & Acc = 0
    R = -1                   otherwise (bad format / no answer tags)

    rM = multi_tool_bonus   if >= min_tools_for_bonus *distinct* relevant tools used
       = 0                  otherwise

For Countdown:
  - "Format is Good"  <=>  <answer>...</answer> tags present
  - "Acc > 0"         <=>  correct arithmetic equation (base_score == 1.0)
  - Multiple tools    <=>  >= 2 distinct relevant tool names in the response
                           (calculator, number_tracker, running_total)

The reward range is [-1, 0, 1.0, 1.0+multi_tool_bonus] which gives RLOO/GRPO
a much stronger gradient signal than the clipped [0, 1] range used before:
  - Bad-format failures get a clear negative signal (-1)
  - Correct multi-tool responses get a small extra bonus (+0.1 default)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.countdown import compute_score
from tir_extension.tools.tool_pool import (
    parse_tool_calls,
    relevant_tool_names,
)


def compute_hierarchical_reward(
    response: str,
    ground_truth: dict,
    active_tools: Iterable[str] | None = None,
    multi_tool_bonus: float = 0.1,
    min_tools_for_bonus: int = 2,
) -> tuple[float, dict]:
    """Return (reward, metadata) using the Tool-Star hierarchical formula.

    Args:
        response: Full model response (with executed tool results spliced in).
        ground_truth: ``{target, numbers}`` dict.
        active_tools: Tool names currently active. ``None`` uses all relevant.
        multi_tool_bonus: rM value when multi-tool condition is met.
        min_tools_for_bonus: Minimum distinct relevant tools required for rM.
    """
    base_score = compute_score(response, ground_truth)

    if active_tools is None:
        active_tools = relevant_tool_names()
    active_set = set(active_tools)
    relevant_set = relevant_tool_names()

    # Determine format and accuracy from base score.
    if base_score == 0.0:
        good_format = False
        accuracy = 0.0
    elif base_score < 1.0:
        good_format = True
        accuracy = 0.0
    else:
        good_format = True
        accuracy = 1.0

    # Count distinct relevant + active tools in the response.
    tool_calls = parse_tool_calls(response)
    distinct_relevant = set(
        name
        for name, _ in tool_calls
        if name in active_set and name in relevant_set
    )
    rM = multi_tool_bonus if len(distinct_relevant) >= min_tools_for_bonus else 0.0

    # Apply hierarchical formula.
    if good_format and accuracy > 0:
        final_reward = accuracy + rM      # 1.0 or 1.0 + bonus
    elif good_format:
        final_reward = 0.0
    else:
        final_reward = -1.0

    metadata = {
        "base_score": base_score,
        "final_reward": final_reward,
        "good_format": good_format,
        "accuracy": accuracy,
        "rM": rM,
        "distinct_relevant_tools": sorted(distinct_relevant),
        "n_distinct_relevant_tools": len(distinct_relevant),
        "n_total_tool_calls": len(tool_calls),
    }
    return final_reward, metadata


def aggregate_hierarchical_metrics(metadata_list: list[dict]) -> dict[str, float]:
    """Aggregate per-rollout metadata into W&B-friendly scalars."""
    if not metadata_list:
        return {}
    n = len(metadata_list)
    return {
        "hier/mean_reward": sum(m["final_reward"] for m in metadata_list) / n,
        "hier/mean_base_score": sum(m["base_score"] for m in metadata_list) / n,
        "hier/mean_rM": sum(m["rM"] for m in metadata_list) / n,
        "hier/frac_good_format": sum(m["good_format"] for m in metadata_list) / n,
        "hier/frac_correct": sum(m["accuracy"] for m in metadata_list) / n,
        "hier/frac_bad_format": sum(not m["good_format"] for m in metadata_list) / n,
        "hier/frac_multi_tool": sum(m["rM"] > 0 for m in metadata_list) / n,
        "hier/mean_tool_calls": sum(m["n_total_tool_calls"] for m in metadata_list) / n,
    }
