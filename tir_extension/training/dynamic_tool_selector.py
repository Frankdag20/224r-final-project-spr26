"""Lightweight state holder for which tools are currently active.

The TIR RLOO trainer owns one ``DynamicToolSelector``. It reads
``selector.active_tools`` on every step (for reward shaping + tool
execution) and calls ``selector.reanalyze(...)`` every N steps to refresh
the recommendation from DSPy.
"""

from __future__ import annotations

import logging
from typing import Iterable

from tir_extension.data.failure_database import FailureDatabase
from tir_extension.tools.tool_pool import TOOL_REGISTRY, relevant_tool_names


logger = logging.getLogger(__name__)


class DynamicToolSelector:
    """Manages the active tool set across an RLOO training run."""

    def __init__(
        self,
        initial_tools: Iterable[str] | None = None,
        candidate_tools: Iterable[str] | None = None,
    ):
        self.candidate_tools = (
            set(candidate_tools) if candidate_tools is not None else set(TOOL_REGISTRY.keys())
        )
        if initial_tools is None:
            self.active_tools: set[str] = set(self.candidate_tools)
        else:
            self.active_tools = set(initial_tools) & self.candidate_tools
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Scheduling helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def should_reanalyze(step: int, reanalyze_every_n_steps: int) -> bool:
        """Return ``True`` when the trainer should ask DSPy for fresh tools."""
        if reanalyze_every_n_steps <= 0:
            return False
        if step <= 0:
            return False
        return step % reanalyze_every_n_steps == 0

    # ------------------------------------------------------------------
    # Mutation.
    # ------------------------------------------------------------------

    def set_active_tools(self, tools: Iterable[str]) -> set[str]:
        new_active = set(tools) & self.candidate_tools
        self.active_tools = new_active
        return self.active_tools

    def reanalyze(
        self,
        failure_db: FailureDatabase,
        analyzer,
        step: int | None = None,
        n_samples: int = 20,
        min_failures: int = 1,
    ) -> dict:
        """Refresh ``active_tools`` from DSPy. Returns the analyzer payload.

        If the failure DB is too small, the active set is left unchanged
        and an explanatory payload is recorded in ``self.history``.
        """
        if len(failure_db) < min_failures:
            entry = {
                "step": step,
                "skipped": True,
                "reason": f"failure_db has {len(failure_db)} < {min_failures}",
                "active_tools": sorted(self.active_tools),
            }
            self.history.append(entry)
            return entry

        result = analyzer.analyze(
            failure_database=failure_db,
            candidate_tools=self.candidate_tools,
            n_samples=n_samples,
        )

        # Always keep at least one relevant tool so the policy has *something*
        # to invoke; defending against pathological DSPy outputs.
        recommended = list(result.get("recommended_tools") or [])
        recommended_set = set(recommended) & self.candidate_tools
        if not recommended_set:
            recommended_set = relevant_tool_names() & self.candidate_tools

        previous = sorted(self.active_tools)
        self.active_tools = recommended_set
        entry = {
            "step": step,
            "skipped": False,
            "previous_active_tools": previous,
            "active_tools": sorted(self.active_tools),
            "analysis": result.get("analysis", ""),
        }
        self.history.append(entry)
        logger.info(
            "DSPy reanalysis @ step=%s: %s -> %s",
            step,
            previous,
            entry["active_tools"],
        )
        return entry
