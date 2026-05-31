"""Generate tool-using Countdown trajectories with a teacher LLM.

For each prompt in the training split we ask an LLM to solve the problem
while emitting `<use_tool>` calls. We then deterministically execute the
tool calls, score the final answer with the standard Countdown reward,
and keep trajectories that succeed (or optionally those that score
``format_score``).

Output is a JSON list of ``{prompt, completion, target, numbers, score}``
dicts, persisted on the Modal volume by default. The next stage
(``sft_tir.py``) trains an SFT model on those trajectories with tool-result
loss masking.

CLI examples (run via Modal):

    modal run modal_train.py tir_gen -- \\
        --n_problems 500 --teacher_model deepseek-chat

    modal run modal_train.py tir_gen -- \\
        --n_problems 200 --keep_format_score True --require_tool_use False
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

# Make the top-level project importable when this file is invoked as
# ``python tir_extension/sft_tir/generate_trajectories.py`` on Modal.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import load_dataset

from evaluation.countdown import compute_score
from tir_extension.tools.tool_pool import (
    TOOL_REGISTRY,
    execute_tool_calls,
    format_tools_for_prompt,
    parse_tool_calls,
    relevant_tool_names,
)


DEFAULT_OUTPUT_PATH = "/vol/checkpoints/tir_trajectories.json"


def build_system_prompt(
    active_tools: Iterable[str],
    require_tool_use: bool = True,
) -> str:
    """Build the system prompt sent to the teacher LLM.

    When ``require_tool_use=True`` (the default), the prompt explicitly
    forbids solutions that don't include at least one ``<use_tool>`` call
    and includes a concrete worked example so the model knows the exact
    format expected.
    """
    tool_catalogue = format_tools_for_prompt(active_tools)
    tool_names_str = ", ".join(sorted(active_tools))

    if require_tool_use:
        obligation = (
            f"You MUST use at least one tool ({tool_names_str}) in every "
            "solution. A response with no <use_tool> call will be rejected "
            "and you will be asked to try again, even if the final answer "
            "is mathematically correct. Do not skip the tool step."
        )
    else:
        obligation = (
            f"You may call any of the following tools ({tool_names_str}) "
            "to assist your reasoning."
        )

    return (
        "You are solving the Countdown arithmetic game. You must combine the "
        "given numbers using +, -, *, / (each number used exactly once) to "
        "reach the target.\n\n"
        f"{obligation}\n\n"
        "Tool call format — write EXACTLY this and then stop; the system "
        "inserts a <tool_result> block which you read before continuing:\n"
        "    <use_tool>tool_name: your_input</use_tool>\n"
        "    <tool_result>result_goes_here</tool_result>\n\n"
        "Worked example (target 24, numbers [3, 8]):\n"
        "    I'll verify my candidate equation with the calculator.\n"
        "    <use_tool>calculator: 3 * 8</use_tool>\n"
        "    <tool_result>24</tool_result>\n"
        "    The result is 24, which matches the target.\n"
        "    <answer>3 * 8</answer>\n\n"
        "Available tools:\n"
        f"{tool_catalogue}\n\n"
        "Always end your response with the final equation wrapped as:\n"
        "    <answer>EXPRESSION</answer>\n"
        "where EXPRESSION uses only +, -, *, /, parentheses, and the given "
        "numbers (each used exactly once)."
    )


def call_teacher(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Issue a single chat completion request."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def generate_one(
    client,
    model: str,
    system_prompt: str,
    prompt: str,
    ground_truth: dict,
    active_tools: set[str],
    max_attempts: int,
    max_tokens: int,
    temperature: float,
    keep_format_score: bool,
    require_tool_use: bool = True,
) -> dict | None:
    """Sample up to ``max_attempts`` completions and return the first usable one.

    When ``require_tool_use=True``, any response that contains no
    ``<use_tool>`` tags is immediately retried without scoring — it doesn't
    count as a valid attempt. This forces the teacher to demonstrate tool use
    before we accept its solution as a training example.
    """
    best_record = None
    for attempt in range(max_attempts):
        try:
            raw = call_teacher(
                client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 - external API
            print(f"  teacher call failed (attempt {attempt+1}): {exc}", flush=True)
            time.sleep(2 ** attempt)
            continue

        # Gate: if the model produced no tool calls and we require them, retry.
        if require_tool_use and not parse_tool_calls(raw):
            print(
                f"  attempt {attempt+1}: no <use_tool> found — retrying",
                flush=True,
            )
            continue

        # Deterministically materialise any <tool_result> spans the teacher
        # left blank, then score the final answer.
        completion = execute_tool_calls(raw, active_tools=active_tools)
        score = compute_score(completion, ground_truth)

        record = {
            "prompt": prompt,
            "completion": completion,
            "target": ground_truth["target"],
            "numbers": ground_truth["numbers"],
            "score": score,
            "num_tool_calls": len(parse_tool_calls(raw)),
        }

        if score >= 1.0:
            return record
        if keep_format_score and score > 0 and best_record is None:
            best_record = record
    return best_record


def iterate_prompts(dataset_name: str, split: str, n_problems: int, seed: int):
    ds = load_dataset(dataset_name, split=split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if n_problems > 0:
        indices = indices[:n_problems]
    for idx in indices:
        row = ds[idx]
        prompt = row["prompt"]
        ground_truth = row.get("ground_truth")
        if ground_truth is None:
            ground_truth = {"target": row["target"], "numbers": row["nums"]}
        yield prompt, ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="asingh15/countdown_tasks_3to4")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n_problems", type=int, default=200,
                        help="Number of distinct prompts to sample. <=0 means all.")
    parser.add_argument("--max_attempts", type=int, default=5,
                        help="Max teacher samples per prompt before giving up.")
    parser.add_argument("--teacher_model", type=str,
                        default=os.environ.get("DSPY_LM", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--keep_format_score", type=lambda x: x.lower() == "true",
                        default=False,
                        help="If True, keep partial-credit completions as fallback.")
    parser.add_argument("--require_tool_use", type=lambda x: x.lower() == "true",
                        default=True,
                        help="If True (default), reject responses with no <use_tool> call.")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--active_tools", type=str, default="",
                        help="Comma-separated subset of tool names. Defaults to all relevant tools.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.active_tools.strip():
        active_tools = {
            name.strip() for name in args.active_tools.split(",") if name.strip()
        }
        unknown = active_tools - set(TOOL_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown tool names: {sorted(unknown)}")
    else:
        # Default: expose all relevant tools to the teacher.
        active_tools = relevant_tool_names()

    system_prompt = build_system_prompt(active_tools, require_tool_use=args.require_tool_use)

    print(f"Generating trajectories: model={args.teacher_model}, "
          f"n_problems={args.n_problems}, require_tool_use={args.require_tool_use}, "
          f"active_tools={sorted(active_tools)}", flush=True)

    # Imported here so the file can still be imported on environments without
    # the OpenAI client (e.g. the trainer process).
    from openai import OpenAI

    client = OpenAI()

    records: list[dict] = []
    successes = 0
    partials = 0
    tool_use_count = 0  # problems attempted so far that had at least one tool call
    attempted = 0

    for i, (prompt, ground_truth) in enumerate(
        iterate_prompts(args.dataset_name, args.split, args.n_problems, args.seed)
    ):
        attempted += 1
        record = generate_one(
            client=client,
            model=args.teacher_model,
            system_prompt=system_prompt,
            prompt=prompt,
            ground_truth=ground_truth,
            active_tools=active_tools,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            keep_format_score=args.keep_format_score,
            require_tool_use=args.require_tool_use,
        )
        if record is None:
            if (i + 1) % 10 == 0:
                tool_use_rate = f"{tool_use_count}/{attempted}"
                print(
                    f"[{i+1}] kept={len(records)} (full={successes}, partial={partials})"
                    f" | tool_use={tool_use_rate}",
                    flush=True,
                )
            continue

        if record.get("num_tool_calls", 0) > 0:
            tool_use_count += 1

        records.append(record)
        if record["score"] >= 1.0:
            successes += 1
        else:
            partials += 1

        if (i + 1) % 10 == 0:
            tool_use_rate = f"{tool_use_count}/{attempted}"
            print(
                f"[{i+1}] kept={len(records)} (full={successes}, partial={partials})"
                f" | tool_use={tool_use_rate}",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as fh:
        json.dump(records, fh)
    print(
        f"Wrote {len(records)} trajectories ({successes} full, {partials} partial, "
        f"{tool_use_count} with tool calls) to {args.output_path}"
    )


if __name__ == "__main__":
    main()
