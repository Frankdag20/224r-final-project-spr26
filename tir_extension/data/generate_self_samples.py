"""Generate self-samples from a policy for ReST and IPO, with REAL multi-turn
tool execution on the TRAINING split.

This fixes the two failure modes of our earlier ReST/IPO data:
  1. Earlier ReST used only the 50 TEST-split eval prompts -> overfitting.
     Here we sample many TRAIN-split prompts for a diverse dataset.
  2. Earlier trajectories had HALLUCINATED tool outputs (single-turn).
     Here we execute tools for real (multi-turn), so the kept trajectories
     match how the model is evaluated.

Outputs two datasets from the same sampling pass:
  - ReST  : flat list [{prompt, completion, score}] of CORRECT (reward==1)
            trajectories, ready for tir_sft.
  - IPO   : {train:[...], test:[...]} of (query, response_ws, response_ls)
            pairs where chosen is a correct rollout and rejected is a wrong
            rollout FOR THE SAME PROMPT (a real, informative preference).

Usage (Modal):
  modal run modal_train.py gen_self_samples -- \
      --model_path /vol/checkpoints/tir_sft_checkpoints/sft_gut_check/tir_sft_run2/model \
      --n_prompts 600 --num_samples 8 \
      --rest_out /vol/checkpoints/rest_dataset_v2.json \
      --ipo_out  /vol/checkpoints/ipo_dataset_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import random
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset

from evaluation.countdown import compute_score
from tir_extension.tools.tool_pool import TOOL_REGISTRY, get_tool, parse_tool_calls


def extract_user_message(prompt: str) -> str:
    m = re.search(r"<\|im_start\|>user\n(.*?)\n?<\|im_end\|>", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt


def generate_multi_turn(llm, tokenizer, prompts, n, sp_kwargs,
                        max_model_len=2048, max_tool_calls=8):
    """Multi-turn generation with real tool execution. Returns [[str]*n]*P."""
    n_prompts = len(prompts)
    contexts = [[prompts[p] for _ in range(n)] for p in range(n_prompts)]
    responses = [[""] * n for _ in range(n_prompts)]
    done = [[False] * n for _ in range(n_prompts)]
    max_ctx = max_model_len - 256

    for turn in range(max_tool_calls + 1):
        pending = [(p, r) for p in range(n_prompts) for r in range(n) if not done[p][r]]
        if not pending:
            break
        stop = ["</answer>", "</use_tool>"] if turn < max_tool_calls else ["</answer>"]
        sp = SamplingParams(n=1, stop=stop, include_stop_str_in_output=True, **sp_kwargs)

        safe = [(p, r) for p, r in pending
                if len(tokenizer.encode(contexts[p][r], add_special_tokens=False)) < max_ctx]
        for p, r in pending:
            if (p, r) not in safe:
                done[p][r] = True
        if not safe:
            break

        outs = llm.generate([contexts[p][r] for p, r in safe], sp)
        for (p, r), out in zip(safe, outs):
            gen = out.outputs[0].text
            responses[p][r] += gen
            if "</use_tool>" in gen and turn < max_tool_calls:
                calls = parse_tool_calls(gen)
                if calls:
                    name, inp = calls[-1]
                    if name in TOOL_REGISTRY:
                        try:
                            res = get_tool(name)(inp)
                        except Exception as exc:
                            res = f"error: {exc}"
                    else:
                        res = f"error: unknown tool '{name}'"
                    block = f"<tool_result>{res}</tool_result>"
                    responses[p][r] += block
                    contexts[p][r] += gen + block
                else:
                    done[p][r] = True
            else:
                done[p][r] = True
    return responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="asingh15/countdown_tasks_3to4")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--n_prompts", type=int, default=600)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--rest_out", type=str, default="/vol/checkpoints/rest_dataset_v2.json")
    ap.add_argument("--ipo_out", type=str, default="/vol/checkpoints/ipo_dataset_v2.json")
    ap.add_argument("--max_pairs_per_prompt", type=int, default=2)
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_chunked_prefill=True, max_num_seqs=64)

    ds = load_dataset(args.dataset, split=args.split)
    n = min(args.n_prompts, len(ds))
    idxs = random.sample(range(len(ds)), n)
    rows = [ds[i] for i in idxs]
    prompts = [r["prompt"] for r in rows]

    sp_kwargs = dict(temperature=args.temperature, top_p=args.top_p,
                     max_tokens=args.max_tokens)
    print(f"Sampling {args.num_samples} responses for {n} TRAIN prompts (multi-turn)...")
    all_resp = generate_multi_turn(llm, tokenizer, prompts, args.num_samples,
                                   sp_kwargs, max_model_len=args.max_model_len)

    rest_records = []
    ipo_pairs = []
    n_correct = n_wrong = 0

    for row, resp_group in zip(rows, all_resp):
        gt = {"target": row["target"], "numbers": row["nums"]}
        query = extract_user_message(row["prompt"])
        correct, wrong = [], []
        for resp in resp_group:
            score = compute_score(resp, gt)
            if score == 1.0:
                correct.append(resp)
            else:
                wrong.append(resp)

        n_correct += len(correct)
        n_wrong += len(wrong)

        # ReST: keep all correct trajectories (real tool outputs).
        for resp in correct:
            rest_records.append({"prompt": row["prompt"], "completion": resp, "score": 1.0})

        # IPO: pair correct (chosen) vs wrong (rejected) for the SAME prompt.
        random.shuffle(correct)
        random.shuffle(wrong)
        for c, w in zip(correct[:args.max_pairs_per_prompt],
                        wrong[:args.max_pairs_per_prompt]):
            ipo_pairs.append({"query": query, "response_ws": c, "response_ls": w})

    print(f"Correct rollouts: {n_correct} | Wrong: {n_wrong}")
    print(f"ReST records: {len(rest_records)} | IPO pairs: {len(ipo_pairs)}")

    os.makedirs(os.path.dirname(args.rest_out) or ".", exist_ok=True)
    with open(args.rest_out, "w") as f:
        json.dump(rest_records, f, indent=2)
    print(f"Saved ReST -> {args.rest_out}")

    random.shuffle(ipo_pairs)
    split = int(len(ipo_pairs) * args.train_frac)
    ipo_out = {"train": ipo_pairs[:split], "test": ipo_pairs[split:]}
    with open(args.ipo_out, "w") as f:
        json.dump(ipo_out, f, indent=2)
    print(f"Saved IPO -> {args.ipo_out} ({len(ipo_out['train'])} train / {len(ipo_out['test'])} test)")


if __name__ == "__main__":
    main()
