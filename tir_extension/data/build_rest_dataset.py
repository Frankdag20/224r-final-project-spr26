"""Build a ReST (Rejection Sampling Fine-Tuning) dataset from eval rollouts.

Takes an eval JSON produced by countdown_eval.py (which stores N responses
per prompt with scores) and filters to keep only correct responses
(score == 1.0), optionally requiring tool use.  The result is a SFT-ready
JSON that tir_extension/sft_tir/sft_tir.py can train on directly.

This is one iteration of the ReST loop:
  1. Sample N responses from current policy (done by countdown_eval.py)
  2. Filter by reward (done here)
  3. Fine-tune on winners (done by tir_sft)
  4. Repeat

Usage (Modal):
  modal run modal_train.py build_rest_dataset -- \
      --eval_path /vol/evaluation/eval_results/tir_sft_eval.json \
      --output_path /vol/checkpoints/rest_dataset.json \
      --require_tools 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def has_tool_calls(text: str) -> bool:
    return bool(re.search(r"<use_tool>", text))


def build_rest_dataset(
    eval_path: str,
    min_score: float = 1.0,
    require_tools: bool = False,
    train_frac: float = 0.9,
) -> tuple[list[dict], dict]:
    with open(eval_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    records: list[dict] = []
    stats = {"total_prompts": 0, "total_responses": 0,
             "kept": 0, "with_tools": 0}

    for row in rows:
        stats["total_prompts"] += 1
        prompt = row["prompt"]
        responses = row["response"]  # list of lists or list of strings
        scores = row["scores"]       # list of lists or list of floats

        # Flatten one level if responses is [[r1, r2, ...]] per prompt
        if responses and isinstance(responses[0], list):
            responses = [r for group in responses for r in group]
            scores = [s for group in scores for s in group]

        for resp, score in zip(responses, scores):
            stats["total_responses"] += 1
            if score < min_score:
                continue
            if require_tools and not has_tool_calls(resp):
                continue
            if has_tool_calls(resp):
                stats["with_tools"] += 1
            records.append({
                "prompt": prompt,
                "completion": resp,
                "score": score,
            })
            stats["kept"] += 1

    return records, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str,
                        default="/vol/evaluation/eval_results/tir_sft_eval.json",
                        help="Eval JSON from countdown_eval.py.")
    parser.add_argument("--output_path", type=str,
                        default="/vol/checkpoints/rest_dataset.json")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="Minimum score to keep a response (1.0 = correct only).")
    parser.add_argument("--require_tools", type=int, default=0,
                        help="1 = only keep responses that contain tool calls.")
    parser.add_argument("--train_frac", type=float, default=0.9)
    args = parser.parse_args()

    print(f"Building ReST dataset from {args.eval_path} ...")
    records, stats = build_rest_dataset(
        eval_path=args.eval_path,
        min_score=args.min_score,
        require_tools=bool(args.require_tools),
    )

    print(f"Prompts: {stats['total_prompts']}")
    print(f"Total responses: {stats['total_responses']}")
    print(f"Kept (score>={args.min_score}): {stats['kept']} "
          f"({stats['kept']/max(stats['total_responses'],1):.1%})")
    print(f"With tool calls: {stats['with_tools']} "
          f"({stats['with_tools']/max(stats['kept'],1):.1%} of kept)")

    import random
    random.shuffle(records)
    split = int(len(records) * args.train_frac)

    # tir_sft expects a flat list (not train/test dict)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved {len(records)} records to {args.output_path}")

    if records:
        ex = records[0]
        print("\n--- Example record ---")
        print("PROMPT (last 100):", ex["prompt"][-100:])
        print("COMPLETION (first 200):", ex["completion"][:200])
        print("SCORE:", ex["score"])


if __name__ == "__main__":
    main()
