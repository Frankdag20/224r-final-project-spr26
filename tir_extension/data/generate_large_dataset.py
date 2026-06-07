"""Generate a large, calculator-dominant tool-using dataset with difficulty scores.

Two generation backends are supported:

  LOCAL (default) — vLLM runs the teacher model on the Modal H100. No external
    API key required; costs only Modal GPU compute credits.
    --local_model   Qwen/Qwen2.5-7B-Instruct   (or any HF model)

  API — calls an OpenAI-compatible external API (DeepSeek, OpenAI, etc.).
    Requires DEEPSEEK_API_KEY or OPENAI_API_KEY in the environment.
    --teacher_model deepseek-chat  --no_local

Each problem is solved with the calculator tool, then optionally scored for
difficulty (1–5) by the same model.  The difficulty score enables curriculum
training: sort by difficulty and feed easy → hard.

Output schema (one record per problem, JSON list):
  {
    "prompt":               str,
    "completion":           str,   # <use_tool>/<tool_result>/<answer> interleaved
    "target":               int,
    "numbers":              list[int],
    "score":                float,  # verifier: 0.0 / 0.1 / 1.0
    "num_tool_calls":       int,
    "tool_names_used":      list[str],
    "difficulty_llm":       int | null,   # 1–5
    "difficulty_rationale": str | null,
    "difficulty_structural": float,       # heuristic 0–1
    "split":                str           # train / val / test
  }

CLI examples (Modal):

    # 2 000 problems on the H100 using a local Qwen teacher (default)
    modal run modal_train.py gen_large_dataset -- \\
        --n_problems 2000 --local_model Qwen/Qwen2.5-7B-Instruct

    # same but skip LLM difficulty scoring (faster)
    modal run modal_train.py gen_large_dataset -- \\
        --n_problems 2000 --score_difficulty False

    # external DeepSeek API (needs DEEPSEEK_API_KEY env var)
    modal run modal_train.py gen_large_dataset -- \\
        --n_problems 2000 --no_local --teacher_model deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable

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

DEFAULT_OUTPUT_PATH = "/vol/checkpoints/tir_large_dataset.json"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_system_prompt(active_tools: Iterable[str]) -> str:
    tool_catalogue = format_tools_for_prompt(active_tools)
    tool_names_str = ", ".join(sorted(active_tools))
    return (
        "You are solving the Countdown arithmetic game. "
        "Combine ALL of the given numbers using +, -, *, / "
        "(each number used exactly once) to reach the target.\n\n"
        f"You MUST call the calculator tool at least once to verify your "
        f"arithmetic before committing an answer. "
        f"Available tools: {tool_names_str}.\n\n"
        "Tool call format — write this and stop; the system fills in the result:\n"
        "    <use_tool>calculator: EXPRESSION</use_tool>\n"
        "    <tool_result>RESULT</tool_result>\n\n"
        "Worked example (target 24, numbers [3, 8]):\n"
        "    I need 3 * 8 = 24. Let me verify.\n"
        "    <use_tool>calculator: 3 * 8</use_tool>\n"
        "    <tool_result>24</tool_result>\n"
        "    The result is 24, which matches the target.\n"
        "    <answer>3 * 8</answer>\n\n"
        "Available tools:\n"
        f"{tool_catalogue}\n\n"
        "Always end your response with:\n"
        "    <answer>EXPRESSION</answer>\n"
        "where EXPRESSION uses only +, -, *, /, parentheses, and the given "
        "numbers (each exactly once)."
    )


_DIFFICULTY_SYSTEM = (
    "You are an expert in combinatorial arithmetic puzzles. "
    "Rate the difficulty of a Countdown puzzle on a scale of 1-5:\n"
    "  1 = trivial   (<=3 numbers, obvious single operation)\n"
    "  2 = easy      (3-4 numbers, 1-2 steps, small search)\n"
    "  3 = moderate  (4 numbers, multi-step, mild backtracking needed)\n"
    "  4 = hard      (4 numbers, many dead ends, non-obvious ordering)\n"
    "  5 = very hard (4 numbers, deeply nested fractions or large gaps)\n\n"
    'Respond with EXACTLY this JSON and nothing else: {"score": <integer 1-5>, "rationale": "<one sentence>"}'
)

_SCORE_RE = re.compile(r'"score"\s*:\s*([1-5])')


# ---------------------------------------------------------------------------
# Local vLLM backend
# ---------------------------------------------------------------------------

class LocalVLLMTeacher:
    """Wraps a vLLM engine for batched teacher generation on the local GPU."""

    def __init__(self, model_name: str, max_model_len: int = 4096, temperature: float = 0.7, max_tokens: int = 1024):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        hf_home = os.environ.get("HF_HOME", "/vol/cache/huggingface")
        os.environ.setdefault("HF_HOME", hf_home)

        print(f"Loading local teacher model: {model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.85,
            max_model_len=max_model_len,
            max_num_seqs=32,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._SP = SamplingParams
        print("Local teacher ready.", flush=True)

    def _format(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_one(self, system: str, user: str, temperature: float | None = None, max_tokens: int | None = None) -> str:
        prompt = self._format(system, user)
        sp = self._SP(
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        outputs = self.llm.generate([prompt], sp)
        return outputs[0].outputs[0].text

    def generate_batch(self, requests: list[tuple[str, str]], temperature: float | None = None, max_tokens: int | None = None) -> list[str]:
        """Generate for a list of (system, user) pairs in one vLLM call."""
        prompts = [self._format(sys, usr) for sys, usr in requests]
        sp = self._SP(
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        outputs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# External API backend (OpenAI-compatible)
# ---------------------------------------------------------------------------

class APITeacher:
    def __init__(self, model: str, base_url: str | None, api_key: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_one(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def generate_batch(self, requests: list[tuple[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> list[str]:
        return [self.generate_one(sys, usr, temperature, max_tokens) for sys, usr in requests]


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

def score_difficulty_llm(teacher, target: int, numbers: list[int], max_retries: int = 3) -> tuple[int | None, str | None]:
    user_msg = f"Target: {target}\nNumbers: {numbers}"
    for attempt in range(max_retries):
        try:
            text = teacher.generate_one(_DIFFICULTY_SYSTEM, user_msg, temperature=0.2, max_tokens=120)
            try:
                parsed = json.loads(text)
                sc = int(parsed.get("score", 0))
                if 1 <= sc <= 5:
                    return sc, parsed.get("rationale")
            except Exception:
                pass
            m = _SCORE_RE.search(text)
            if m:
                return int(m.group(1)), None
        except Exception as exc:
            print(f"  difficulty scoring failed (attempt {attempt+1}): {exc}", flush=True)
            time.sleep(2 ** attempt)
    return None, None


def structural_difficulty(target: int, numbers: list[int]) -> float:
    n = len(numbers)
    max_reachable = sum(abs(x) for x in numbers)
    target_gap = abs(target) / max(max_reachable, 1)
    order_complexity = math.log1p(math.factorial(n) * (4 ** (n - 1)))
    order_norm = order_complexity / math.log1p(math.factorial(4) * 64)
    score = 0.4 * min(n / 4.0, 1.0) + 0.3 * min(target_gap, 1.0) + 0.3 * min(order_norm, 1.0)
    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_one_record(
    teacher,
    system_prompt: str,
    prompt: str,
    ground_truth: dict,
    active_tools: set[str],
    max_attempts: int,
    max_tokens: int,
    temperature: float,
) -> dict | None:
    for attempt in range(max_attempts):
        try:
            raw = teacher.generate_one(system_prompt, prompt, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            print(f"  teacher call failed (attempt {attempt+1}): {exc}", flush=True)
            time.sleep(min(2 ** attempt, 30))
            continue

        calls = parse_tool_calls(raw)
        if not any(name == "calculator" for name, _ in calls):
            continue

        completion = execute_tool_calls(raw, active_tools=active_tools)
        score = compute_score(completion, ground_truth)

        if score >= 1.0:
            return {
                "prompt": prompt,
                "completion": completion,
                "target": ground_truth["target"],
                "numbers": ground_truth["numbers"],
                "score": score,
                "num_tool_calls": len(calls),
                "tool_names_used": sorted({name for name, _ in calls}),
            }
    return None


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
        gt = row.get("ground_truth") or {"target": row["target"], "numbers": row["nums"]}
        yield prompt, gt


def assign_split(i: int, total: int, val_frac: float, test_frac: float) -> str:
    frac = i / max(total - 1, 1)
    if frac < (1.0 - val_frac - test_frac):
        return "train"
    if frac < (1.0 - test_frac):
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate large calculator-dominant dataset with difficulty scores."
    )
    parser.add_argument("--dataset_name", default="asingh15/countdown_tasks_3to4")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n_problems", type=int, default=2000,
                        help="Problems to attempt. <=0 = all.")
    parser.add_argument("--max_attempts", type=int, default=8,
                        help="Teacher retries per problem before skipping.")

    # --- Backend selection ---
    parser.add_argument("--local_model", type=str, default=DEFAULT_LOCAL_MODEL,
                        help="HuggingFace model to run locally via vLLM (default). "
                             "Pass empty string '' to disable and use --teacher_model API.")
    parser.add_argument("--no_local", action="store_true",
                        help="Force use of the external API instead of local vLLM.")

    # --- External API backend ---
    parser.add_argument("--teacher_model", default=os.environ.get("TEACHER_MODEL", "deepseek-chat"),
                        help="Model name for external API calls (used when --no_local).")
    parser.add_argument("--teacher_base_url", default=os.environ.get("TEACHER_BASE_URL", "https://api.deepseek.com"),
                        help="Base URL for the external OpenAI-compatible API.")

    # --- Generation params ---
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM max_model_len for local mode.")
    parser.add_argument("--active_tools", default="calculator",
                        help="Comma-separated tool names. Default: 'calculator'.")

    # --- Difficulty scoring ---
    parser.add_argument("--score_difficulty", type=lambda x: x.lower() == "true", default=True,
                        help="Ask the teacher to rate each problem 1-5. Default True.")

    # --- Output ---
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve active tools
    if args.active_tools.strip():
        active_tools = {n.strip() for n in args.active_tools.split(",") if n.strip()}
        unknown = active_tools - set(TOOL_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown tools: {sorted(unknown)}")
    else:
        active_tools = relevant_tool_names()

    # Build teacher
    use_local = bool(args.local_model) and not args.no_local
    if use_local:
        teacher = LocalVLLMTeacher(
            model_name=args.local_model,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        backend_desc = f"local vLLM ({args.local_model})"
    else:
        api_key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        if not api_key:
            raise RuntimeError(
                "No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY, "
                "or use the default local vLLM mode (omit --no_local)."
            )
        teacher = APITeacher(
            model=args.teacher_model,
            base_url=args.teacher_base_url,
            api_key=api_key,
        )
        backend_desc = f"API ({args.teacher_model} @ {args.teacher_base_url})"

    print(
        f"Dataset generation: backend={backend_desc}, "
        f"n_problems={args.n_problems}, active_tools={sorted(active_tools)}, "
        f"score_difficulty={args.score_difficulty}",
        flush=True,
    )

    system_prompt = build_system_prompt(active_tools)

    prompt_iter = list(iterate_prompts(args.dataset_name, args.split, args.n_problems, args.seed))
    total = len(prompt_iter)
    records: list[dict] = []
    attempted = 0
    skipped = 0

    # Resume from checkpoint if it exists
    checkpoint_path = args.output_path + ".ckpt.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as fh:
            records = json.load(fh)
        print(f"Resumed from checkpoint: {len(records)} records already saved.", flush=True)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    for i, (prompt, gt) in enumerate(prompt_iter):
        # Skip problems already completed in a previous run
        if i < len(records):
            continue

        attempted += 1
        record = generate_one_record(
            teacher=teacher,
            system_prompt=system_prompt,
            prompt=prompt,
            ground_truth=gt,
            active_tools=active_tools,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        if record is None:
            skipped += 1
            if attempted % 50 == 0:
                print(f"[{attempted}/{total}] kept={len(records)}, skipped={skipped}", flush=True)
            continue

        record["difficulty_structural"] = structural_difficulty(record["target"], record["numbers"])

        if args.score_difficulty:
            d_score, d_rationale = score_difficulty_llm(teacher, record["target"], record["numbers"])
        else:
            d_score, d_rationale = None, None

        record["difficulty_llm"] = d_score
        record["difficulty_rationale"] = d_rationale
        record["split"] = assign_split(len(records), total, args.val_frac, args.test_frac)
        records.append(record)

        # Checkpoint every 100 records so a crash/timeout loses at most 100
        if len(records) % 100 == 0:
            with open(checkpoint_path, "w") as fh:
                json.dump(records, fh)
            print(f"  [checkpoint] saved {len(records)} records", flush=True)

        if attempted % 50 == 0 or attempted == total:
            print(
                f"[{attempted}/{total}] kept={len(records)}, skipped={skipped} | "
                f"last_difficulty={d_score}",
                flush=True,
            )

    # Final save
    with open(args.output_path, "w") as fh:
        json.dump(records, fh, indent=2)
    # Remove checkpoint file now that we have the final output
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    scored = [r for r in records if r["difficulty_llm"] is not None]
    if scored:
        avg_d = sum(r["difficulty_llm"] for r in scored) / len(scored)
        dist = {k: sum(1 for r in scored if r["difficulty_llm"] == k) for k in range(1, 6)}
        print(f"Difficulty distribution: {dist} (mean={avg_d:.2f})")

    splits = {s: sum(1 for r in records if r["split"] == s) for s in ("train", "val", "test")}
    print(
        f"Wrote {len(records)} records to {args.output_path} | splits: {splits} | "
        f"tools used: {sorted({t for r in records for t in r['tool_names_used']})}"
    )


if __name__ == "__main__":
    main()
