"""Batch evaluation script for Countdown checkpoints using vLLM.

Supports two evaluation modes controlled by --multi_turn:
  0 (default): single-pass generation — model generates the full response
               including any tool calls in one shot. Tool results are NOT
               executed; the model predicts them itself.
  1           : multi-turn generation — model stops on </use_tool>, the
               tool is actually executed, the real result is fed back, and
               generation continues. This is the fair evaluation mode for
               models trained with multi-turn tool use.
"""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
import os
import re
import sys
from datasets import load_dataset, Dataset
import pandas as pd
from evaluation.countdown import compute_score

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def load_checkpoint(model_path, max_model_len=2048, gpu_memory_utilization=0.9,
                    max_num_batched_tokens=4096, enable_chunked_prefill=True, max_num_seqs=16):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization,
              max_num_batched_tokens=max_num_batched_tokens,
              enable_chunked_prefill=enable_chunked_prefill,
              max_num_seqs=max_num_seqs)
    return tokenizer, llm


def generate_multi_turn(llm, tokenizer, prompts, num_responses, temperature, top_p, top_k,
                        min_p, max_tokens, max_model_len=2048, max_tool_calls=8):
    """Generate with real tool execution: stop on </use_tool>, run the tool,
    feed the result back, continue until </answer>."""
    from tir_extension.tools.tool_pool import TOOL_REGISTRY, get_tool, parse_tool_calls

    n = num_responses
    n_prompts = len(prompts)

    contexts = [[prompts[p] for _ in range(n)] for p in range(n_prompts)]
    responses = [[""] * n for _ in range(n_prompts)]
    done = [[False] * n for _ in range(n_prompts)]

    for turn in range(max_tool_calls + 1):
        pending = [(p, r) for p in range(n_prompts) for r in range(n) if not done[p][r]]
        if not pending:
            break

        stop_tokens = ["</answer>", "</use_tool>"] if turn < max_tool_calls else ["</answer>"]
        sp = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            max_tokens=max_tokens, n=1,
            stop=stop_tokens, include_stop_str_in_output=True,
        )

        # Filter contexts that are too long using the actual tokenizer
        # (char//4 underestimates for math-heavy text and lets over-length
        # prompts slip through, crashing vLLM).
        max_ctx = (max_model_len or 2048) - 256
        safe_pending, safe_ctxs = [], []
        for p, r in pending:
            ctx = contexts[p][r]
            if len(tokenizer.encode(ctx, add_special_tokens=False)) < max_ctx:
                safe_pending.append((p, r))
                safe_ctxs.append(ctx)
            else:
                done[p][r] = True

        if not safe_ctxs:
            break

        outputs = llm.generate(safe_ctxs, sp)

        for (p, r), out in zip(safe_pending, outputs):
            generated = out.outputs[0].text
            responses[p][r] += generated

            if "</use_tool>" in generated and turn < max_tool_calls:
                calls = parse_tool_calls(generated)
                if calls:
                    name, inp = calls[-1]
                    if name in TOOL_REGISTRY:
                        try:
                            result = get_tool(name)(inp)
                        except Exception as exc:
                            result = f"error: {exc}"
                    else:
                        result = f"error: unknown tool '{name}'"
                    tool_block = f"<tool_result>{result}</tool_result>"
                    responses[p][r] += tool_block
                    contexts[p][r] += generated + tool_block
                else:
                    done[p][r] = True
            else:
                done[p][r] = True

    return responses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    parser.add_argument('--enable_chunked_prefill', type=bool, default=True)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--max_num_seqs', type=int, default=16)
    parser.add_argument('--model_path', type=str, default='PATH_TO_CHECKPOINT')
    parser.add_argument('--eval_dataset', type=str, default='asingh15/countdown_tasks_3to4')
    parser.add_argument('--output_dir', type=str, default='evaluation/eval_results')
    parser.add_argument('--output_name', type=str, default='OUTPUT_NAME')
    parser.add_argument('--num_responses', type=int, default=16)
    parser.add_argument('--multi_turn', type=int, default=0,
                        help='1=execute tools during generation (fair eval for multi-turn models).')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer, llm = load_checkpoint(
        args.model_path, args.max_model_len, args.gpu_memory_utilization,
        args.max_num_batched_tokens, args.enable_chunked_prefill, args.max_num_seqs,
    )

    loaded_dataset = load_dataset(args.eval_dataset, split='test')
    prompt_df = loaded_dataset.to_pandas()
    prompts = prompt_df['prompt'].tolist()

    if args.multi_turn:
        print(f"Multi-turn eval: executing tools during generation.")
        all_responses = generate_multi_turn(
            llm, tokenizer, prompts,
            num_responses=args.num_responses,
            temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, min_p=args.min_p,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
        )
        response = all_responses
        scores = []
        for i, resp_group in enumerate(all_responses):
            row = prompt_df.iloc[i]
            gt = {'target': row['target'], 'numbers': row['nums']}
            scores.append([compute_score(r, gt) for r in resp_group])
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            min_p=args.min_p, max_tokens=args.max_tokens, n=args.num_responses,
        )
        outputs = llm.generate(prompts, sampling_params)
        response, scores = [], []
        for i, output in enumerate(outputs):
            row = prompt_df.iloc[i]
            gt = {'target': row['target'], 'numbers': row['nums']}
            curr_r, curr_s = [], []
            for o in output.outputs:
                curr_r.append(o.text)
                curr_s.append(compute_score(o.text, gt))
            response.append(curr_r)
            scores.append(curr_s)

    prompt_df['response'] = response
    prompt_df['scores'] = scores
    os.makedirs(args.output_dir, exist_ok=True)
    output_ds = Dataset.from_pandas(prompt_df)
    output_ds.to_json(f'{args.output_dir}/{args.output_name}.json')
