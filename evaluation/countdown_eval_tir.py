"""Batch evaluation for TIR-trained Countdown checkpoints.

Like ``countdown_eval.py`` but supports multi-turn tool execution during
generation.  For each prompt, the model generates iteratively: when it
emits ``</use_tool>``, the tool is executed and the result is appended
before generation resumes.

Usage (local):
    python evaluation/countdown_eval_tir.py \
        --model_path /path/to/tir_sft_checkpoint/model \
        --output_name tir_eval_run

Usage (Modal):
    modal run modal_train.py eval_tir -- \
        --model_path /vol/checkpoints/tir_sft_checkpoints/.../model \
        --output_name tir_eval_run
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import os

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluation.countdown import compute_score
from tir_extension.tools.tool_pool import (
    execute_tool_calls,
    relevant_tool_names,
)
from tir_extension.tools.system_prompt import build_tool_system_prompt


def load_checkpoint(model_path, max_model_len=2048, gpu_memory_utilization=0.9,
                    max_num_batched_tokens=4096, enable_chunked_prefill=True,
                    max_num_seqs=16):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_seqs=max_num_seqs,
    )
    return tokenizer, llm


def generate_multi_turn(
    llm: LLM,
    prompts: list[str],
    num_responses: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    max_tokens: int,
    max_tool_turns: int,
    active_tools: set[str],
) -> list[list[str]]:
    """Generate with iterative tool execution.

    Returns list of shape [n_prompts][num_responses] of completed response strings.
    """
    # Expand prompts for independent rollouts
    expanded = []
    for prompt in prompts:
        for _ in range(num_responses):
            expanded.append(prompt)

    num_rollouts = len(expanded)
    contexts = list(expanded)
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
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tok,
            n=1,
            stop=["</use_tool>", "</answer>"],
            include_stop_str_in_output=True,
        )
        outputs = llm.generate(active_contexts, params)

        for idx, output in zip(active_idx, outputs):
            new_text = output.outputs[0].text
            new_token_count = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], "token_ids") else 0

            generated[idx] += new_text
            tokens_used[idx] += new_token_count

            if "</answer>" in new_text or tokens_used[idx] >= max_tokens:
                finished[idx] = True
                continue

            if "</use_tool>" in new_text:
                executed = execute_tool_calls(
                    generated[idx],
                    active_tools=active_tools,
                    max_calls=1,
                )
                if executed != generated[idx]:
                    generated[idx] = executed
                else:
                    finished[idx] = True
                    continue
                contexts[idx] = expanded[idx] + generated[idx]
            else:
                finished[idx] = True

    # Reshape
    result = []
    for i in range(len(prompts)):
        group = []
        for j in range(num_responses):
            group.append(generated[i * num_responses + j])
        result.append(group)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--enable_chunked_prefill", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, default="asingh15/countdown_tasks_3to4")
    parser.add_argument("--output_dir", type=str, default="evaluation/eval_results")
    parser.add_argument("--output_name", type=str, default="tir_eval_run")
    parser.add_argument("--num_responses", type=int, default=16)
    parser.add_argument("--max_tool_turns", type=int, default=5)
    parser.add_argument("--no_tools", action="store_true",
                        help="Disable tool execution (baseline comparison)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer, llm = load_checkpoint(
        args.model_path,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.max_num_batched_tokens,
        args.enable_chunked_prefill,
        args.max_num_seqs,
    )

    loaded_dataset = load_dataset(args.eval_dataset, split="test")
    prompt_df = loaded_dataset.to_pandas()
    prompts = list(prompt_df["prompt"])

    active_tools = set() if args.no_tools else relevant_tool_names()

    if args.no_tools:
        # Single-pass generation (same as baseline countdown_eval.py)
        params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_tokens=args.max_tokens,
            n=args.num_responses,
        )
        outputs = llm.generate(prompts, params)
        all_responses = []
        for output in outputs:
            all_responses.append([o.text for o in output.outputs])
    else:
        # Multi-turn tool-integrated generation
        all_responses = generate_multi_turn(
            llm=llm,
            prompts=prompts,
            num_responses=args.num_responses,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_tokens=args.max_tokens,
            max_tool_turns=args.max_tool_turns,
            active_tools=active_tools,
        )

    # Score
    response_col = []
    scores_col = []
    for i, responses in enumerate(all_responses):
        row = prompt_df.iloc[i]
        ground_truth = {"target": row["target"], "numbers": row["nums"]}
        curr_scores = []
        for resp in responses:
            curr_scores.append(compute_score(resp, ground_truth))
        response_col.append(responses)
        scores_col.append(curr_scores)

    prompt_df["response"] = response_col
    prompt_df["scores"] = scores_col

    os.makedirs(args.output_dir, exist_ok=True)
    output_ds = Dataset.from_pandas(prompt_df)
    output_ds.to_json(f"{args.output_dir}/{args.output_name}.json")
    print(f"Saved evaluation results to {args.output_dir}/{args.output_name}.json")
