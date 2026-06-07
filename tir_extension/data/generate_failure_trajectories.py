"""Build the mixed-quality dataset required for RCTP-FT (Stage 1 of RC-GRPO).

Reads a JSON file of successful tool-using trajectories and produces a paired
set of failure trajectories by replacing each <answer> with an expression that
uses valid syntax but evaluates to the wrong value.  The output is a merged
JSON with both success ("high") and failure ("low") records.

The reward_label field drives the token injected during RCTP-FT:
  "high" -> [Reward Goal: <|high reward|>]
  "low"  -> [Reward Goal: <|low reward|>]

Usage (local test):
    python tir_extension/data/generate_failure_trajectories.py \\
        --input_path tir_large_dataset_partial.json \\
        --output_path tir_rctp_mixed.json

Usage (Modal):
    modal run modal_train.py gen_failure_traj -- \\
        --input_path /vol/checkpoints/tir_large_dataset-6626-2.json.ckpt.json \\
        --output_path /vol/checkpoints/tir_rctp_mixed.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.countdown import compute_score

_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def make_failure_completion(completion: str, numbers: list[int]) -> str:
    """Replace <answer> with an expression that uses only a subset of the
    available numbers so validate_equation fails -> score=0.1."""
    # Use first two numbers summed; for single-number problems fall back to n+1.
    if len(numbers) >= 2:
        wrong_expr = f"{numbers[0]} + {numbers[1]}"
    else:
        wrong_expr = str(numbers[0] + 1)
    return _ANSWER_RE.sub(f"<answer>{wrong_expr}</answer>", completion, count=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True,
                        help="Path to success records (score=1.0) JSON.")
    parser.add_argument("--output_path", required=True,
                        help="Output path for mixed success+failure JSON.")
    parser.add_argument("--max_records", type=int, default=0,
                        help="Cap on success records to use (0 = all).")
    parser.add_argument("--keep_tool_calls", action="store_true", default=True,
                        help="Keep tool-call prefix in failure trajectories.")
    args = parser.parse_args()

    with open(args.input_path) as fh:
        records = json.load(fh)

    success_records = [r for r in records if float(r.get("score", 0)) >= 1.0]
    if args.max_records > 0:
        success_records = success_records[:args.max_records]
    print(f"Loaded {len(success_records)} success records from {args.input_path}")

    mixed: list[dict] = []
    n_success = n_failure = n_accidental = 0

    for rec in success_records:
        # Success record: high-reward label.
        mixed.append({**rec, "reward_label": "high"})
        n_success += 1

        numbers = rec["numbers"]
        fail_completion = make_failure_completion(rec["completion"], numbers)
        gt = {"target": rec["target"], "numbers": numbers}
        fail_score = compute_score(fail_completion, gt)

        if fail_score >= 1.0:
            # Accidental correct answer — skip this failure.
            n_accidental += 1
            continue

        mixed.append({
            **rec,
            "completion": fail_completion,
            "score": float(fail_score),
            "reward_label": "low",
        })
        n_failure += 1

    print(
        f"Mixed dataset: {n_success} success + {n_failure} failure records "
        f"({n_accidental} skipped as accidental successes)"
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as fh:
        json.dump(mixed, fh, indent=2)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
