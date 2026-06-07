"""DSPy-driven failure-mode analysis and tool recommendation.

Given a sample of failed Countdown rollouts and the catalogue of available
tools, this module asks an LLM (via DSPy) to:

1) Diagnose the dominant failure modes.
2) Recommend which tools from the catalogue would address them.

The output is validated against ``TOOL_REGISTRY`` so the trainer can never
end up with a tool name that doesn't actually exist.
"""

from __future__ import annotations

import os
from typing import Iterable

try:
    import dspy
except ImportError as exc:  # pragma: no cover - import handled at runtime
    dspy = None
    _DSPY_IMPORT_ERROR = exc
else:
    _DSPY_IMPORT_ERROR = None

from tir_extension.data.failure_database import (
    FailureDatabase,
    FailureRecord,
    format_failures_for_analyzer,
)
from tir_extension.tools.tool_pool import TOOL_REGISTRY, format_tools_for_prompt, relevant_tool_names


DEFAULT_LM_NAME = os.environ.get("DSPY_LM", "openai/gpt-4o-mini")
DEFAULT_MAX_TOKENS = int(os.environ.get("DSPY_MAX_TOKENS", "1024"))


def _ensure_dspy() -> None:
    if dspy is None:
        raise ImportError(
            "dspy is required for the failure analyzer but is not installed. "
            f"Original import error: {_DSPY_IMPORT_ERROR}"
        )


def configure_dspy(
    model: str | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> None:
    """Configure the DSPy global LM. Safe to call repeatedly."""
    _ensure_dspy()
    lm = dspy.LM(
        model or DEFAULT_LM_NAME,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        **kwargs,
    )
    dspy.configure(lm=lm)


if dspy is not None:

    class FailureModeAnalysis(dspy.Signature):
        """Diagnose Countdown failure modes and recommend tools.

        Given a sample of failed Countdown rollouts and a catalogue of
        available tools, identify the dominant failure modes (e.g. arithmetic
        mistakes, reusing numbers, missing the target, ignoring the answer
        format) and recommend which tools from the catalogue would directly
        address them. Only recommend tools whose names appear in the
        provided catalogue. Prefer fewer high-precision tools over many
        loosely related ones.
        """

        failure_cases: str = dspy.InputField(
            desc="Block of failure cases including target, available numbers, and model response."
        )
        available_tools: str = dspy.InputField(
            desc="Tool catalogue. Each line: '- name: description'."
        )
        failure_analysis: str = dspy.OutputField(
            desc="Short paragraph summarising the dominant failure modes."
        )
        recommended_tools: list[str] = dspy.OutputField(
            desc="List of tool names from the catalogue to enable for the policy."
        )

else:  # pragma: no cover - placeholder when dspy isn't installed
    FailureModeAnalysis = None  # type: ignore[assignment]


class FailureAnalyzerModule:
    """Wraps a DSPy ChainOfThought predictor that returns validated tools."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        require_at_least_one: bool = True,
    ):
        _ensure_dspy()
        configure_dspy(model=model, max_tokens=max_tokens)
        self.predictor = dspy.ChainOfThought(FailureModeAnalysis)
        self.require_at_least_one = require_at_least_one

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def analyze(
        self,
        failure_database: FailureDatabase,
        candidate_tools: Iterable[str] | None = None,
        n_samples: int = 20,
        seed: int | None = None,
    ) -> dict:
        """Sample failures, run DSPy, and return validated recommendations.

        Returns ``{"analysis": str, "recommended_tools": list[str]}``. If
        DSPy fails (e.g. transient API error), the dict ``recommended_tools``
        falls back to the full set of candidate tool names so training can
        continue without crashing.
        """
        if candidate_tools is None:
            candidate_tools = list(TOOL_REGISTRY.keys())
        candidate_set = set(candidate_tools)

        samples = failure_database.sample(n_samples, seed=seed)
        if not samples:
            return {
                "analysis": "no failures available",
                "recommended_tools": sorted(relevant_tool_names() & candidate_set),
            }

        failure_text = format_failures_for_analyzer(samples)
        tools_text = format_tools_for_prompt(candidate_set)

        try:
            prediction = self.predictor(
                failure_cases=failure_text,
                available_tools=tools_text,
            )
        except Exception as exc:  # noqa: BLE001 - external API
            return {
                "analysis": f"dspy_error: {exc}",
                "recommended_tools": sorted(relevant_tool_names() & candidate_set),
            }

        analysis = getattr(prediction, "failure_analysis", "") or ""
        raw_tools = getattr(prediction, "recommended_tools", []) or []
        recommended = self._validate_tool_names(raw_tools, candidate_set)
        if self.require_at_least_one and not recommended:
            # Fall back to all relevant tools so the trainer never ends up
            # with an empty active set.
            recommended = sorted(candidate_set)
        return {
            "analysis": analysis,
            "recommended_tools": recommended,
        }

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_tool_names(
        raw_tools,
        candidate_set: set[str],
    ) -> list[str]:
        seen: list[str] = []
        if isinstance(raw_tools, str):
            # Sometimes the LM returns a comma-separated string instead of a list.
            raw_tools = [part.strip() for part in raw_tools.split(",")]
        for name in raw_tools:
            if not isinstance(name, str):
                continue
            name = name.strip().strip("'\"")
            if name in candidate_set and name not in seen:
                seen.append(name)
        return seen


def _sample_records(records: list[FailureRecord], n: int) -> list[FailureRecord]:
    """Convenience helper exposed for unit tests."""
    if n >= len(records):
        return list(records)
    import random as _random

    return _random.sample(records, n)
