"""Label Countdown dataset problems with difficulty using a large teacher model.

Runs Qwen3-72B (thinking disabled) on each problem twice:
  1. No tools — single-turn generation
  2. With tools — multi-turn tool execution

Extracts per-problem: correctness, self-assessed difficulty (1-10),
and reasoning trace length (chars before <answer>).

Usage (via Modal — see label_difficulty_modal.py):
    python tir_extension/curriculum/label_difficulty.py \
        --shard_id 0 --num_shards 5 \
        --output_dir /vol/curriculum/difficulty_labels
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import os
import re
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluation.countdown import compute_score
from tir_extension.tools.tool_pool import execute_tool_calls, relevant_tool_names
from tir_extension.tools.system_prompt import build_tool_system_prompt

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

DIFFICULTY_INSTRUCTION = (
    "\n\nAfter providing your answer, rate the difficulty of this problem "
    "on a scale of 1 to 10 (1 = trivial, 10 = extremely hard). "
    "You may briefly explain your reasoning, but you MUST end with your "
    "final rating as a single integer inside tags like this: "
    "<difficulty>N</difficulty>"
)

# The no-tools system prompt matches what the vanilla RLOO student sees:
# just "You are a helpful assistant." (the dataset's built-in system prompt).
# The user message in the dataset already contains full solve instructions.
# We only append the difficulty rating instruction.
NO_TOOLS_SYSTEM = "You are a helpful assistant." + DIFFICULTY_INSTRUCTION


def build_with_tools_system():
    """Build the with-tools system prompt including difficulty instruction."""
    base = build_tool_system_prompt(active_tools=sorted(relevant_tool_names()))
    return base + DIFFICULTY_INSTRUCTION


def extract_user_message(raw_prompt: str) -> str:
    """Extract the user message content from a pre-formatted Qwen chat template prompt."""
    # The dataset prompt looks like:
    #   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{USER_MSG}<|im_end|>\n...
    match = re.search(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
        raw_prompt,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return raw_prompt


def build_prompt(tokenizer, system_text: str, user_text: str) -> str:
    """Build a chat prompt with thinking disabled (Qwen3 enable_thinking=False)."""
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_difficulty(response: str) -> int | None:
    """Extract difficulty score from <difficulty>N</difficulty> tags."""
    match = re.search(r"<difficulty>\s*(\d+)\s*</difficulty>", response)
    if match:
        val = int(match.group(1))
        return max(1, min(10, val))
    return None


def reasoning_length_before_answer(response: str) -> int:
    """Character count of text before the first <answer> tag."""
    match = re.search(r"<answer>", response)
    if match:
        return match.start()
    return len(response)


# ---------------------------------------------------------------------------
# Multi-turn tool execution (adapted from countdown_eval_tir.py)
# ---------------------------------------------------------------------------

def generate_multi_turn(
    llm: LLM,
    prompts: list[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_tool_turns: int,
    active_tools: set[str],
) -> list[str]:
    """Generate with iterative tool execution. Returns one response per prompt."""
    num_rollouts = len(prompts)
    contexts = list(prompts)
    generated = [""] * num_rollouts
    finished = [False] * num_rollouts
    tokens_used = [0] * num_rollouts

    for turn in range(max_tool_turns + 1):
        active_idx = [i for i in range(num_rollouts) if not finished[i]]
        if not active_idx:
            break

        active_contexts = [contexts[i] for i in active_idx]
        remaining = [max(1, max_tokens - tokens_used[i]) for i in active_idx]
        max_tok = max(remaining)

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tok,
            n=1,
            stop=["</use_tool>"],
            include_stop_str_in_output=True,
        )
        outputs = llm.generate(active_contexts, params)

        for idx, output in zip(active_idx, outputs):
            new_text = output.outputs[0].text
            new_token_count = (
                len(output.outputs[0].token_ids)
                if hasattr(output.outputs[0], "token_ids")
                else 0
            )

            generated[idx] += new_text
            tokens_used[idx] += new_token_count

            if tokens_used[idx] >= max_tokens:
                finished[idx] = True
                continue

            if "</use_tool>" in new_text:
                executed = execute_tool_calls(
                    generated[idx], active_tools=active_tools, max_calls=1,
                )
                if executed != generated[idx]:
                    generated[idx] = executed
                else:
                    finished[idx] = True
                    continue
                contexts[idx] = prompts[idx] + generated[idx]
            else:
                finished[idx] = True

    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--dataset_name", type=str, default="asingh15/countdown_tasks_3to4")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Number of prompts per vLLM batch")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_tool_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Use 0 for greedy (deterministic difficulty labels)")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    args = parser.parse_args()

    # Load dataset and shard
    print(f"Loading dataset {args.dataset_name} split={args.dataset_split}")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    total = len(ds)
    shard_size = (total + args.num_shards - 1) // args.num_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, total)
    ds_shard = ds.select(range(start, end))
    print(f"Shard {args.shard_id}/{args.num_shards}: examples {start}-{end} ({len(ds_shard)} total)")

    # Load model
    print(f"Loading {args.model_name} with TP={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_model_len,
        enable_chunked_prefill=True,
    )

    active_tools = relevant_tool_names()
    with_tools_sys = build_with_tools_system()

    # Load tokenizer for Qwen3 chat template (enable_thinking=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build all prompts
    raw_prompts = list(ds_shard["prompt"])
    targets = list(ds_shard["target"])
    nums_list = list(ds_shard["nums"])

    user_messages = [extract_user_message(p) for p in raw_prompts]
    no_tools_prompts = [build_prompt(tokenizer, NO_TOOLS_SYSTEM, u) for u in user_messages]
    with_tools_prompts = [build_prompt(tokenizer, with_tools_sys, u) for u in user_messages]

    n = len(ds_shard)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"shard_{args.shard_id}.jsonl")

    # --- Resume support: count existing lines to skip already-done problems ---
    already_done = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    already_done += 1
    if already_done > 0:
        print(f"Resuming shard {args.shard_id}: {already_done}/{n} already done, "
              f"skipping to problem {already_done}")

    # Process in batches, appending to output file
    t0 = time.time()

    for batch_start in range(already_done, n, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n)
        batch_idx = list(range(batch_start, batch_end))

        # --- No-tools pass (single-turn) ---
        nt_prompts = [no_tools_prompts[i] for i in batch_idx]
        nt_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=1,
        )
        nt_outputs = llm.generate(nt_prompts, nt_params)
        nt_responses = [o.outputs[0].text for o in nt_outputs]

        # --- With-tools pass (multi-turn) ---
        wt_prompts = [with_tools_prompts[i] for i in batch_idx]
        wt_responses = generate_multi_turn(
            llm=llm,
            prompts=wt_prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_tool_turns=args.max_tool_turns,
            active_tools=active_tools,
        )

        # --- Extract features and append to file ---
        with open(output_path, "a") as f:
            for j, idx in enumerate(batch_idx):
                ground_truth = {"target": targets[idx], "numbers": nums_list[idx]}

                nt_resp = nt_responses[j]
                wt_resp = wt_responses[j]

                result = {
                    "index": start + idx,
                    "target": targets[idx],
                    "nums": nums_list[idx],
                    "no_tools_correct": compute_score(nt_resp, ground_truth) >= 1.0,
                    "no_tools_difficulty": extract_difficulty(nt_resp),
                    "no_tools_reasoning_length": reasoning_length_before_answer(nt_resp),
                    "no_tools_response": nt_resp,
                    "with_tools_correct": compute_score(wt_resp, ground_truth) >= 1.0,
                    "with_tools_difficulty": extract_difficulty(wt_resp),
                    "with_tools_reasoning_length": reasoning_length_before_answer(wt_resp),
                    "with_tools_response": wt_resp,
                }
                f.write(json.dumps(result) + "\n")

        elapsed = time.time() - t0
        done = batch_end - already_done
        total_done = batch_end
        rate = done / elapsed if elapsed > 0 else 0
        remaining = n - total_done
        eta = remaining / rate / 60 if rate > 0 else 0
        print(f"Shard {args.shard_id}: {total_done}/{n} ({total_done/n:.0%}) | "
              f"{rate:.1f} problems/sec | ETA {eta:.0f}min")

    elapsed = time.time() - t0
    new_done = n - already_done
    print(f"Shard {args.shard_id} complete: {new_done} new problems in {elapsed/60:.1f}min "
          f"({new_done/elapsed:.1f} problems/sec)" if elapsed > 0 else
          f"Shard {args.shard_id} complete: nothing new to process")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
