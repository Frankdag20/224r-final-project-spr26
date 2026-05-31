"""Persistent database of failed Countdown rollouts.

Populated by the TIR RLOO trainer (and optionally seeded from a baseline
RLOO run). The DSPy analyzer samples from this database to recommend which
tools should be enabled for the policy.
"""

from __future__ import annotations

import json
import os
import random
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


# Map between numeric reward values from evaluation.countdown.compute_score
# and human-readable failure type labels.
NO_ANSWER_FAILURE = "no_answer"
FORMAT_BUT_WRONG_FAILURE = "wrong_numbers_or_value"
DEFAULT_DB_PATH = "/vol/checkpoints/failure_db.json"


def classify_failure(score: float) -> str | None:
    """Return a coarse failure-type label, or ``None`` if not a failure."""
    if score is None:
        return NO_ANSWER_FAILURE
    if score >= 1.0:
        return None
    if score <= 0.0:
        return NO_ANSWER_FAILURE
    return FORMAT_BUT_WRONG_FAILURE


@dataclass
class FailureRecord:
    """One failed rollout. Kept small so JSON serialisation stays cheap."""

    prompt: str
    response: str
    ground_truth: dict[str, Any]
    score: float
    failure_type: str
    step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailureRecord":
        return cls(
            prompt=payload["prompt"],
            response=payload["response"],
            ground_truth=payload["ground_truth"],
            score=float(payload["score"]),
            failure_type=payload.get("failure_type", FORMAT_BUT_WRONG_FAILURE),
            step=payload.get("step"),
        )


class FailureDatabase:
    """In-memory list of FailureRecord plus JSON persistence helpers."""

    def __init__(self, path: str | None = None, max_records: int = 5000):
        self.path = path
        self.max_records = max_records
        self.records: list[FailureRecord] = []
        # Used to avoid concurrent writers when multiple Ray actors share
        # the same on-disk DB file (e.g. seeding from baseline + new run).
        self._lock = threading.Lock()
        if path is not None and os.path.exists(path):
            self.load(path)

    # ------------------------------------------------------------------
    # Mutating API.
    # ------------------------------------------------------------------

    def add(
        self,
        prompt: str,
        response: str,
        ground_truth: dict[str, Any],
        score: float,
        step: int | None = None,
    ) -> bool:
        """Add a failure if the score qualifies. Returns ``True`` if stored."""
        failure_type = classify_failure(score)
        if failure_type is None:
            return False
        record = FailureRecord(
            prompt=prompt,
            response=response,
            ground_truth=dict(ground_truth) if isinstance(ground_truth, dict) else ground_truth,
            score=float(score),
            failure_type=failure_type,
            step=step,
        )
        with self._lock:
            self.records.append(record)
            if len(self.records) > self.max_records:
                # Drop oldest entries first so the DB stays bounded but biased
                # toward recent failure modes.
                overflow = len(self.records) - self.max_records
                self.records = self.records[overflow:]
        return True

    def extend(self, records: Iterable[FailureRecord]) -> None:
        with self._lock:
            self.records.extend(records)
            if len(self.records) > self.max_records:
                overflow = len(self.records) - self.max_records
                self.records = self.records[overflow:]

    def clear(self) -> None:
        with self._lock:
            self.records.clear()

    # ------------------------------------------------------------------
    # Read-only API.
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def sample(self, n: int, seed: int | None = None) -> list[FailureRecord]:
        with self._lock:
            if not self.records:
                return []
            k = min(n, len(self.records))
            rng = random.Random(seed) if seed is not None else random
            return rng.sample(self.records, k)

    def failure_type_counts(self) -> dict[str, int]:
        with self._lock:
            counter: Counter[str] = Counter(r.failure_type for r in self.records)
        return dict(counter)

    # ------------------------------------------------------------------
    # Persistence.
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        path = path or self.path
        if path is None:
            raise ValueError("FailureDatabase.save requires a path")
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with self._lock:
            payload = [r.to_dict() for r in self.records]
        with open(path, "w") as fh:
            json.dump(payload, fh)
        self.path = path
        return path

    def load(self, path: str | None = None) -> int:
        path = path or self.path
        if path is None or not os.path.exists(path):
            return 0
        with open(path, "r") as fh:
            payload = json.load(fh)
        records = [FailureRecord.from_dict(item) for item in payload]
        with self._lock:
            self.records = records
        self.path = path
        return len(records)


def format_failures_for_analyzer(
    records: Iterable[FailureRecord],
    max_chars_per_record: int = 600,
) -> str:
    """Render failures as a single string for the DSPy analyzer prompt.

    Truncated per-record so a couple dozen samples fit in a reasonable
    context window.
    """
    blocks = []
    for i, record in enumerate(records):
        gt = record.ground_truth
        target = gt.get("target")
        numbers = gt.get("numbers")
        response = record.response
        if len(response) > max_chars_per_record:
            response = response[: max_chars_per_record - 3] + "..."
        blocks.append(
            f"[failure {i+1}] type={record.failure_type} score={record.score}\n"
            f"  target={target}, numbers={numbers}\n"
            f"  response: {response}"
        )
    return "\n\n".join(blocks)
