"""Tool-aware reward wrapper around evaluation/countdown.compute_score.

The base Countdown reward returns 0.0 / 0.1 / 1.0. This wrapper adds:

- A per-call bonus for invoking a **relevant** tool that's currently in
  the active tool set (recommended by DSPy).
- A per-call penalty for invoking an **irrelevant** tool, regardless of
  whether it's active.

The final reward is clipped to [0, 1] so an excessive number of correct
tool calls can't dominate the base correctness signal, but the policy
still gets an early shaping signal even when ``base_score == 0``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

# Make the top-level project importable when this file is invoked from
# any nested working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.countdown import compute_score
from tir_extension.tools.tool_pool import (
    TOOL_REGISTRY,
    irrelevant_tool_names,
    parse_tool_calls,
    relevant_tool_names,
)


def compute_score_with_tools(
    response: str,
    ground_truth: dict,
    active_tools: Iterable[str] | None = None,
    tool_reward: float = 0.1,
    tool_penalty: float = 0.1,
    max_bonus_calls: int = 3,
    clip_low: float = 0.0,
    clip_high: float = 1.0,
    format_score: float = 0.1,
    score: float = 1.0,
) -> tuple[float, dict]:
    """Return ``(final_score, metadata)`` for a single rollout.

    Args:
        response: Full model response (potentially with executed tool results).
        ground_truth: ``{target, numbers}`` dict, same format as compute_score.
        active_tools: Tool names currently enabled (typically the DSPy
            recommendation). ``None`` falls back to the relevant tool set.
        tool_reward: Bonus per *relevant + active* tool invocation.
        tool_penalty: Penalty per *irrelevant* tool invocation.
        max_bonus_calls: Cap on how many relevant calls earn a bonus (so the
            policy can't farm reward by spamming calls).
        clip_low, clip_high: Final clipping range.
        format_score, score: Base Countdown reward levels (passed through).
    """
    base_score = compute_score(
        response,
        ground_truth,
        format_score=format_score,
        score=score,
    )

    if active_tools is None:
        active_tools = relevant_tool_names()
    active_set = set(active_tools)

    relevant_set = relevant_tool_names()
    irrelevant_set = irrelevant_tool_names()

    tool_calls = parse_tool_calls(response)

    relevant_invoked: list[str] = []
    irrelevant_invoked: list[str] = []
    unknown_invoked: list[str] = []
    for name, _input in tool_calls:
        if name not in TOOL_REGISTRY:
            unknown_invoked.append(name)
            continue
        if name in relevant_set and name in active_set:
            relevant_invoked.append(name)
        elif name in irrelevant_set:
            irrelevant_invoked.append(name)
        # Relevant-but-inactive tools are neither rewarded nor penalised;
        # we treat them as "neutral" so the policy can still explore.

    bonus_calls = min(len(relevant_invoked), max_bonus_calls)
    bonus = tool_reward * bonus_calls
    penalty = tool_penalty * len(irrelevant_invoked)
    raw = base_score + bonus - penalty
    final_score = max(clip_low, min(clip_high, raw))

    metadata = {
        "base_score": base_score,
        "tool_bonus": bonus,
        "tool_penalty": penalty,
        "raw_score": raw,
        "final_score": final_score,
        "tools_invoked": [name for name, _ in tool_calls],
        "relevant_invoked": relevant_invoked,
        "irrelevant_invoked": irrelevant_invoked,
        "unknown_invoked": unknown_invoked,
        "num_tool_calls": len(tool_calls),
    }
    return final_score, metadata


def aggregate_tool_metrics(
    metadata_list: list[dict],
) -> dict[str, float]:
    """Aggregate per-rollout metadata into W&B-friendly scalars."""
    if not metadata_list:
        return {}
    n = len(metadata_list)
    return {
        "tool/mean_base_score": sum(m["base_score"] for m in metadata_list) / n,
        "tool/mean_bonus": sum(m["tool_bonus"] for m in metadata_list) / n,
        "tool/mean_penalty": sum(m["tool_penalty"] for m in metadata_list) / n,
        "tool/mean_calls": sum(m["num_tool_calls"] for m in metadata_list) / n,
        "tool/mean_relevant_calls": sum(len(m["relevant_invoked"]) for m in metadata_list) / n,
        "tool/mean_irrelevant_calls": sum(len(m["irrelevant_invoked"]) for m in metadata_list) / n,
        "tool/fraction_using_any_tool": sum(
            1 for m in metadata_list if m["num_tool_calls"] > 0
        ) / n,
    }
