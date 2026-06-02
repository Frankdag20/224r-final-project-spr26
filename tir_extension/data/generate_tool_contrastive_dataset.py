"""Generate tool-contrastive preference pairs for IPO/DPO.

Chosen  : tool-using, CORRECT trajectory (from tir_trajectories JSON)
Rejected: WRONG response for the same problem (from failure_db JSON or
          a second trajectories file filtered to score < 1.0)

The key fix vs. the naive strip-tool-calls approach: we need a genuine
quality gap between chosen and rejected.  Stripping tool calls from a
correct solution often still leaves a correct answer, giving IPO zero
signal.  Using actually-wrong responses as rejected creates a real gap:
  "tool-using + correct → preferred"
  "no tools + wrong    → less preferred"

Sources for rejected responses (tried in order):
  1. --failure_db_path   : failure_database.json from RLOO training
  2. --rejected_path     : a second trajectories JSON (score filtered)
  3. Fallback            : strip tool calls (original behaviour, weak)

Output format: {"train": [...], "test": [...]}
  Each record: {"query": str, "response_ws": str, "response_ls": str}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_USE_TOOL_RE = re.compile(r"<use_tool>.*?</use_tool>", re.DOTALL)
_TOOL_RESULT_RE = re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL)


def strip_tool_calls(text: str) -> str:
    text = _USE_TOOL_RE.sub("", text)
    text = _TOOL_RESULT_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_user_message(prompt: str) -> str | None:
    m = re.search(r"<\|im_start\|>user\n(.*?)\n?<\|im_end\|>", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"User:\s*(.*?)(?:\nAssistant:|\Z)", prompt, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return None


def normalize_query(query: str) -> str:
    """Collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", query.strip().lower())


def load_failure_db(path: str) -> dict[str, list[str]]:
    """Load failure_db.json → {normalized_query: [bad_response, ...]}."""
    with open(path) as f:
        records = json.load(f)

    # failure_db stores a list of FailureRecord dicts
    if isinstance(records, list):
        entries = records
    elif isinstance(records, dict) and "records" in records:
        entries = records["records"]
    else:
        entries = []

    by_query: dict[str, list[str]] = defaultdict(list)
    for rec in entries:
        prompt = rec.get("prompt", "")
        response = rec.get("response", "")
        score = rec.get("score", 0.0)
        if score >= 1.0 or not response:
            continue
        q = extract_user_message(prompt) or prompt
        by_query[normalize_query(q)].append(response)

    print(f"Failure DB: {len(entries)} records → {len(by_query)} unique queries with failures")
    return dict(by_query)


def build_pairs(
    chosen_records: list[dict],
    failure_db: dict[str, list[str]] | None,
    rejected_records: list[dict] | None,
    min_tool_calls: int = 1,
    fallback_strip: bool = True,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)

    # Build a lookup from normalised query → list of wrong responses
    # from the secondary rejected_records file (score < 1.0)
    rejected_by_query: dict[str, list[str]] = defaultdict(list)
    if rejected_records:
        for rec in rejected_records:
            if rec.get("score", 1.0) >= 1.0:
                continue
            q = extract_user_message(rec.get("prompt", "")) or rec.get("prompt", "")
            rejected_by_query[normalize_query(q)].append(rec.get("completion", ""))

    pairs: list[dict] = []
    skipped = 0

    for rec in chosen_records:
        score = rec.get("score", 0.0)
        if score < 1.0:
            skipped += 1
            continue

        completion = rec.get("completion", "")
        prompt = rec.get("prompt", "")

        n_tools = len(re.findall(r"<use_tool>", completion))
        if n_tools < min_tool_calls:
            skipped += 1
            continue

        query = extract_user_message(prompt)
        if query is None:
            skipped += 1
            continue

        nq = normalize_query(query)

        # --- find a rejected response ---
        rejected: str | None = None

        # 1. failure DB
        if failure_db and nq in failure_db:
            candidates = failure_db[nq]
            rejected = rng.choice(candidates)

        # 2. rejected_records file
        if rejected is None and nq in rejected_by_query:
            rejected = rng.choice(rejected_by_query[nq])

        # 3. fallback: strip tool calls (weak but better than nothing)
        if rejected is None and fallback_strip:
            stripped = strip_tool_calls(completion)
            if len(stripped) >= 20 and stripped != completion.strip():
                rejected = stripped

        if rejected is None:
            skipped += 1
            continue

        pairs.append({
            "query": query,
            "response_ws": completion,
            "response_ls": rejected,
        })

    print(f"Built {len(pairs)} pairs, skipped {skipped} records.")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="/vol/checkpoints/tir_trajectories_2000.json",
                        help="Chosen trajectories (tool-using, correct).")
    parser.add_argument("--output_path", type=str,
                        default="/vol/checkpoints/tool_contrastive_ipo_v2.json")
    parser.add_argument("--failure_db_path", type=str,
                        default="/vol/checkpoints/failure_db.json",
                        help="failure_database.json from RLOO run (preferred rejected source).")
    parser.add_argument("--rejected_path", type=str, default=None,
                        help="Optional second trajectories file filtered to score<1 for rejected.")
    parser.add_argument("--min_tool_calls", type=int, default=1)
    parser.add_argument("--fallback_strip", type=int, default=1,
                        help="Fall back to tool-stripping if no failure DB match (1=yes).")
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading chosen trajectories from {args.input_path} ...")
    with open(args.input_path) as f:
        chosen_records = json.load(f)
    print(f"  {len(chosen_records)} chosen records.")

    failure_db: dict | None = None
    if args.failure_db_path and os.path.exists(args.failure_db_path):
        print(f"Loading failure DB from {args.failure_db_path} ...")
        failure_db = load_failure_db(args.failure_db_path)
    else:
        print("No failure DB found — will use fallback strip only.")

    rejected_records: list | None = None
    if args.rejected_path and os.path.exists(args.rejected_path):
        print(f"Loading rejected trajectories from {args.rejected_path} ...")
        with open(args.rejected_path) as f:
            rejected_records = json.load(f)

    pairs = build_pairs(
        chosen_records=chosen_records,
        failure_db=failure_db,
        rejected_records=rejected_records,
        min_tool_calls=args.min_tool_calls,
        fallback_strip=bool(args.fallback_strip),
        seed=args.seed,
    )

    random.shuffle(pairs)
    split = int(len(pairs) * args.train_frac)
    output = {"train": pairs[:split], "test": pairs[split:]}

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    n_from_db = sum(1 for p in pairs if "<use_tool>" not in p["response_ls"])
    print(f"Saved {len(output['train'])} train / {len(output['test'])} test pairs")
    print(f"  Rejected from failure DB or wrong trajectories: {n_from_db}/{len(pairs)}")
    print(f"  Rejected from strip fallback: {len(pairs)-n_from_db}/{len(pairs)}")
    print(f"  -> {args.output_path}")

    if output["train"]:
        ex = output["train"][0]
        print("\n--- Example pair ---")
        print("QUERY:", ex["query"][:100])
        print("CHOSEN (first 150):", ex["response_ws"][:150])
        print("REJECTED (first 150):", ex["response_ls"][:150])


if __name__ == "__main__":
    main()
