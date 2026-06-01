"""RLOO training loop with failure-mode-aware tool integration.

Subclasses ``RLOOTrainer`` to:

1) Execute any ``<use_tool>...</use_tool>`` calls in each sampled response
   and splice the real outputs in as ``<tool_result>...</tool_result>``.
2) Score rollouts with ``compute_score_with_tools`` so the policy gets a
   shaping signal for invoking the *right* tools.
3) Push every failed rollout into a persistent ``FailureDatabase`` and
   periodically ask a DSPy analyzer to refresh the set of active tools.
4) Build a ``tool_result_mask`` so the deterministic tool-output tokens
   are excluded from the policy-gradient / entropy / KL terms.

Designed to run on Modal exactly like the baseline RLOO trainer:

    modal run modal_train.py tir_rloo -- --num_training_steps 250
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import ray
import wandb

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")

from rloo_trainer.rloo import RLOOTrainer
from tir_extension.data.failure_database import FailureDatabase, DEFAULT_DB_PATH
from tir_extension.tools.tool_pool import (
    TOOL_REGISTRY,
    execute_tool_calls,
    find_tool_result_spans,
    relevant_tool_names,
)
from tir_extension.training.dynamic_tool_selector import DynamicToolSelector
from tir_extension.training.failure_aware_reward import (
    aggregate_tool_metrics,
    compute_score_with_tools,
)


class TIRRLOOTrainer(RLOOTrainer):
    """RLOO trainer with failure-aware tool integration."""

    def __init__(
        self,
        # TIR-specific.
        reanalyze_every_n_steps: int = 3,
        tool_reward: float = 0.1,
        tool_penalty: float = 0.1,
        min_failures_before_analysis: int = 50,
        failure_db_path: str | None = DEFAULT_DB_PATH,
        max_failures_per_db: int = 5000,
        analyzer_sample_size: int = 20,
        dspy_model: str | None = None,
        dspy_max_tokens: int | None = None,
        initial_active_tools: str | None = None,
        max_tool_call_truncation: int = 16,
        save_failure_db_every_n_steps: int = 5,
        # Everything else is forwarded straight to RLOOTrainer.
        **rloo_kwargs,
    ):
        super().__init__(**rloo_kwargs)

        self.reanalyze_every_n_steps = reanalyze_every_n_steps
        self.tool_reward = tool_reward
        self.tool_penalty = tool_penalty
        self.min_failures_before_analysis = min_failures_before_analysis
        self.failure_db_path = failure_db_path
        self.analyzer_sample_size = analyzer_sample_size
        self.dspy_model = dspy_model
        self.dspy_max_tokens = dspy_max_tokens
        self.max_tool_call_truncation = max_tool_call_truncation
        self.save_failure_db_every_n_steps = save_failure_db_every_n_steps

        # Bounded persistent log of failed rollouts (seeded if a baseline run
        # already wrote one to the same path).
        self.failure_db = FailureDatabase(
            path=self.failure_db_path,
            max_records=max_failures_per_db,
        )

        initial_tools: set[str] | None
        if initial_active_tools:
            initial_tools = {
                name.strip()
                for name in initial_active_tools.split(",")
                if name.strip()
            }
            unknown = initial_tools - set(TOOL_REGISTRY)
            if unknown:
                raise ValueError(f"Unknown initial_active_tools: {sorted(unknown)}")
        else:
            # Start with every relevant tool active. DSPy will reduce/expand
            # this on the first reanalysis once enough failures accumulate.
            initial_tools = relevant_tool_names()

        self.tool_selector = DynamicToolSelector(
            initial_tools=initial_tools,
            candidate_tools=set(TOOL_REGISTRY),
        )

        # Build the analyzer lazily so that runs without DSPy installed can
        # still construct the trainer (they will fail only when reanalyse is
        # actually invoked, which is gated on having enough failures anyway).
        self._analyzer = None

        # Mirror these on wandb config so they appear alongside the existing
        # RLOOTrainer hyperparameters.
        self.wandb.config.update(
            {
                "tir_reanalyze_every_n_steps": reanalyze_every_n_steps,
                "tir_tool_reward": tool_reward,
                "tir_tool_penalty": tool_penalty,
                "tir_min_failures_before_analysis": min_failures_before_analysis,
                "tir_failure_db_path": failure_db_path,
                "tir_initial_active_tools": sorted(self.tool_selector.active_tools),
                "tir_candidate_tools": sorted(self.tool_selector.candidate_tools),
            }
        )

    # ------------------------------------------------------------------
    # Analyzer lazy-init.
    # ------------------------------------------------------------------

    def _get_analyzer(self):
        if self._analyzer is None:
            from tir_extension.tools.dspy_failure_analyzer import (
                FailureAnalyzerModule,
            )

            self._analyzer = FailureAnalyzerModule(
                model=self.dspy_model,
                max_tokens=self.dspy_max_tokens,
            )
        return self._analyzer

    # ------------------------------------------------------------------
    # Tokenization with tool-result mask.
    # ------------------------------------------------------------------

    def tokenize_batch_tir(
        self, batch: dict, group_size_override: int | None = None
    ) -> dict[str, np.ndarray]:
        """Same flattening as ``tokenize_batch`` plus a tool-result token mask.

        Args:
            batch: dict with prompt/response/rewards/sample_log_probs lists.
            group_size_override: number of responses per prompt when different
                from ``self.group_size`` (e.g. during the self-critic phase).

        Returns the standard tokenized fields and an additional
        ``tool_result_mask`` array shaped ``[batch*group, seq_len]`` with 1
        on tokens whose character range falls inside a
        ``<tool_result>...</tool_result>`` span (always within the response
        portion).
        """
        n = group_size_override if group_size_override is not None else self.group_size

        all_prompts = batch["prompt"]
        all_responses = batch["response"]
        all_rewards = batch["rewards"]
        all_sample_log_probs = batch["sample_log_probs"]

        prompts_repeated = [item for item in all_prompts for _ in range(n)]
        responses_flat = [item for sublist in all_responses for item in sublist]
        rewards_flat = [item for sublist in all_rewards for item in sublist]
        sample_lp_flat = [item for sublist in all_sample_log_probs for item in sublist]
        assert len(prompts_repeated) == len(responses_flat) == len(rewards_flat)

        # Prompts: identical handling to the parent class.
        self.tokenizer.padding_side = "left"
        tokenized_prompts = self.tokenizer(
            prompts_repeated,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="np",
        )

        # Responses: also request offset_mapping so we can build the
        # tool-result mask without retokenising each span.
        self.tokenizer.padding_side = "right"
        tokenized_responses = self.tokenizer(
            responses_flat,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_response_length,
            return_tensors="np",
            return_offsets_mapping=True,
        )

        prompt_input_ids = tokenized_prompts["input_ids"]
        prompt_attention_mask = tokenized_prompts["attention_mask"]
        response_input_ids = tokenized_responses["input_ids"]
        response_attention_mask = tokenized_responses["attention_mask"]
        offsets = tokenized_responses["offset_mapping"]

        response_tool_mask = np.zeros_like(response_input_ids, dtype=np.int64)
        for i, response_text in enumerate(responses_flat):
            spans = find_tool_result_spans(response_text)
            if not spans:
                continue
            for tok_idx in range(response_input_ids.shape[1]):
                start, end = int(offsets[i, tok_idx, 0]), int(offsets[i, tok_idx, 1])
                if start == end:
                    continue
                for span_start, span_end in spans:
                    if start < span_end and end > span_start:
                        response_tool_mask[i, tok_idx] = 1
                        break

        # Concatenate prompt + response halves to match the parent layout.
        input_ids = np.concatenate([prompt_input_ids, response_input_ids], axis=1)
        attention_mask = np.concatenate(
            [prompt_attention_mask, response_attention_mask], axis=1
        )
        is_response_token = np.concatenate(
            [np.zeros_like(prompt_input_ids), np.ones_like(response_input_ids)],
            axis=1,
        )
        tool_result_mask = np.concatenate(
            [np.zeros_like(prompt_input_ids), response_tool_mask], axis=1
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "is_response_token": is_response_token,
            "tool_result_mask": tool_result_mask,
            "rewards": np.array(rewards_flat, dtype=np.float32),
            "sample_log_probs": np.array(sample_lp_flat, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------

    def train(self):  # type: ignore[override]
        import random
        import shutil

        last_checkpoint_dir = None
        global_step = 0
        for epoch in range(self.num_epochs):
            if global_step > 0 and global_step == self.num_training_steps:
                break
            for train_iter, batch in enumerate(self.train_dataloader):
                if global_step > 0 and global_step == self.num_training_steps:
                    break

                # ----- 1) Sample -----
                print(
                    f"[TIR] Sampling, epoch={epoch} step={global_step}",
                    flush=True,
                )
                model_path = (
                    self.model_name
                    if last_checkpoint_dir is None
                    else os.path.join(last_checkpoint_dir, "model")
                )
                self._create_sampling_worker(model_path)

                all_prompts = batch["prompt"]
                all_ground_truth = batch["ground_truth"]
                assert (
                    len(all_prompts) == len(all_ground_truth) == self.batch_size
                ), (
                    f"len(all_prompts) = {len(all_prompts)}, "
                    f"len(all_ground_truth) = {len(all_ground_truth)}, "
                    f"self.batch_size = {self.batch_size}"
                )
                all_raw_responses, all_sample_log_probs = ray.get(
                    self.sampling_worker.generate.remote(all_prompts)
                )

                # ----- 2) Tool execution -----
                active_tools = set(self.tool_selector.active_tools)
                all_responses: list[list[str]] = []
                for raw_group in all_raw_responses:
                    materialised = [
                        execute_tool_calls(
                            response,
                            active_tools=active_tools,
                            max_calls=self.max_tool_call_truncation,
                        )
                        for response in raw_group
                    ]
                    all_responses.append(materialised)

                # ----- 3) Tool-aware rewards -----
                all_rewards: list[list[float]] = []
                all_meta: list[dict] = []
                for resp_group, gt in zip(all_responses, all_ground_truth):
                    group_rewards = []
                    for resp in resp_group:
                        score, meta = compute_score_with_tools(
                            response=resp,
                            ground_truth=gt,
                            active_tools=active_tools,
                            tool_reward=self.tool_reward,
                            tool_penalty=self.tool_penalty,
                        )
                        group_rewards.append(score)
                        all_meta.append(meta)
                    all_rewards.append(group_rewards)

                reward_mean = float(np.mean(all_rewards))
                base_score_mean = float(
                    np.mean([m["base_score"] for m in all_meta])
                )
                print(
                    f"[TIR] step={global_step} reward_mean={reward_mean:.3f} "
                    f"base_score_mean={base_score_mean:.3f} "
                    f"active_tools={sorted(active_tools)}",
                    flush=True,
                )

                # ----- 4) Failure DB -----
                for prompt, resp_group, gt, group_rewards in zip(
                    all_prompts, all_responses, all_ground_truth, all_rewards
                ):
                    for resp, score in zip(resp_group, group_rewards):
                        self.failure_db.add(
                            prompt=prompt,
                            response=resp,
                            ground_truth=gt,
                            score=score,
                            step=global_step,
                        )

                generation_table = self._build_generation_table(
                    all_prompts, all_responses, all_rewards
                )

                # ----- 5) Tokenize -----
                tokenized = self.tokenize_batch_tir(
                    {
                        "prompt": all_prompts,
                        "response": all_responses,
                        "rewards": all_rewards,
                        "sample_log_probs": all_sample_log_probs,
                    }
                )

                # ----- 6) Update -----
                optimizer_path = (
                    None
                    if last_checkpoint_dir is None
                    else os.path.join(last_checkpoint_dir, "optimizer.pt")
                )
                scheduler_path = (
                    None
                    if last_checkpoint_dir is None
                    else os.path.join(last_checkpoint_dir, "scheduler.pt")
                )
                self._create_update_worker(model_path, optimizer_path, scheduler_path)

                all_metrics = ray.get(
                    self.update_worker.update_gradient_accumulation.remote(
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized["attention_mask"],
                        is_response_token=tokenized["is_response_token"],
                        rewards=tokenized["rewards"],
                        sample_log_probs=tokenized["sample_log_probs"],
                        tool_result_mask=tokenized["tool_result_mask"],
                    )
                )

                # ----- 7) Checkpoint -----
                if self.save_every_n_steps > 0 and global_step % self.save_every_n_steps == 0:
                    save_dir = os.path.join(
                        self.save_dir,
                        self.wandb_project,
                        self.wandb_name,
                        f"epoch_{epoch}_step_{global_step}",
                    )
                else:
                    save_dir = os.path.join(
                        self.save_dir,
                        self.wandb_project,
                        self.wandb_name,
                        "latest_checkpoint",
                    )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir, exist_ok=True)

                save_model_path = os.path.join(save_dir, "model")
                save_optimizer_path = os.path.join(save_dir, "optimizer.pt")
                save_scheduler_path = os.path.join(save_dir, "scheduler.pt")
                ray.get(
                    self.update_worker.update_checkpoint_paths.remote(
                        model_path=save_model_path,
                        optimizer_path=save_optimizer_path,
                        scheduler_path=save_scheduler_path,
                        load_checkpoint=False,
                    )
                )
                ray.get(self.update_worker.save_checkpoint.remote())
                last_checkpoint_dir = save_dir

                # ----- 8) Periodic DSPy re-analysis -----
                reanalysis_payload = None
                if DynamicToolSelector.should_reanalyze(
                    global_step + 1, self.reanalyze_every_n_steps
                ):
                    try:
                        analyzer = self._get_analyzer()
                        reanalysis_payload = self.tool_selector.reanalyze(
                            failure_db=self.failure_db,
                            analyzer=analyzer,
                            step=global_step,
                            n_samples=self.analyzer_sample_size,
                            min_failures=self.min_failures_before_analysis,
                        )
                        print(
                            f"[TIR] DSPy reanalysis: {reanalysis_payload}",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[TIR] reanalysis failed: {exc}", flush=True)

                # ----- 9) Periodic failure-DB persistence -----
                if (
                    self.failure_db_path
                    and self.save_failure_db_every_n_steps > 0
                    and (global_step + 1) % self.save_failure_db_every_n_steps == 0
                ):
                    try:
                        self.failure_db.save(self.failure_db_path)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[TIR] failed to save failure DB: {exc}", flush=True)

                # ----- 10) Logging -----
                tool_metrics = aggregate_tool_metrics(all_meta)
                failure_counts = self.failure_db.failure_type_counts()
                log_dict = {
                    "train/epoch": epoch,
                    "train/train_iter": train_iter,
                    "train/global_step": global_step,
                    "sampling/reward_mean": reward_mean,
                    "sampling/base_score_mean": base_score_mean,
                    "tir/failure_db_size": len(self.failure_db),
                    "tir/n_active_tools": len(self.tool_selector.active_tools),
                    "tir/active_tools": ",".join(
                        sorted(self.tool_selector.active_tools)
                    ),
                    **{f"train/{k}": v for k, v in all_metrics.items()},
                    **tool_metrics,
                    **{
                        f"tir/failure_type_{name}": count
                        for name, count in failure_counts.items()
                    },
                }
                if reanalysis_payload is not None:
                    log_dict["tir/reanalysis_step"] = reanalysis_payload.get(
                        "step", global_step
                    )
                if generation_table is not None:
                    log_dict["samples/generations"] = generation_table

                self.wandb.log(log_dict, step=global_step)

                global_step += 1

        # Persist final failure DB and tear down Ray actors.
        if self.failure_db_path:
            try:
                self.failure_db.save(self.failure_db_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[TIR] failed to save failure DB at exit: {exc}", flush=True)

        if self.sampling_worker is not None:
            ray.kill(self.sampling_worker)
            self.sampling_worker = None
        if self.update_worker is not None:
            ray.kill(self.update_worker)
            self.update_worker = None
        ray.shutdown()
        self.wandb.finish()


# ---------------------------------------------------------------------------
# CLI mirroring rloo.py + the TIR-specific flags.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="asingh15/qwen-sft-countdown-defaultproj")
    parser.add_argument("--ref_model_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str,
                        default="asingh15/countdown_tasks_3to4")
    parser.add_argument("--wandb_project", type=str, default="tir_rloo_project")
    parser.add_argument("--wandb_name", type=str, default="tir_rloo_test")
    parser.add_argument("--lr_schedule", type=str, default="constant")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--kl_divergence_coefficient", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_training_steps", type=int, default=250)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--enable_chunked_prefill", action="store_true")
    parser.add_argument("--disable_chunked_prefill", action="store_true")
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--max_table_rows", type=int, default=20)
    parser.add_argument("--save_every_n_steps", type=int, default=-1)
    parser.add_argument("--save_dir", type=str,
                        default="/vol/checkpoints/tir_rloo_checkpoints")

    # TIR-specific.
    parser.add_argument("--reanalyze_every_n_steps", type=int, default=3)
    parser.add_argument("--tool_reward", type=float, default=0.1)
    parser.add_argument("--tool_penalty", type=float, default=0.1)
    parser.add_argument("--min_failures_before_analysis", type=int, default=50)
    parser.add_argument("--failure_db_path", type=str, default=DEFAULT_DB_PATH)
    parser.add_argument("--max_failures_per_db", type=int, default=5000)
    parser.add_argument("--analyzer_sample_size", type=int, default=20)
    parser.add_argument("--dspy_model", type=str, default=None)
    parser.add_argument("--dspy_max_tokens", type=int, default=None)
    parser.add_argument("--initial_active_tools", type=str, default=None,
                        help="Comma-separated subset of tool names. Defaults to all relevant.")
    parser.add_argument("--max_tool_call_truncation", type=int, default=16)
    parser.add_argument("--save_failure_db_every_n_steps", type=int, default=5)
    return parser


def main():
    args = _build_parser().parse_args()

    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        raise ValueError(
            "Cannot set both --enable_chunked_prefill and --disable_chunked_prefill."
        )
    if args.disable_chunked_prefill:
        enable_chunked_prefill = False
    else:
        enable_chunked_prefill = True

    # Split args into base-RLOO kwargs and TIR-specific kwargs.
    tir_kwargs = {
        "reanalyze_every_n_steps": args.reanalyze_every_n_steps,
        "tool_reward": args.tool_reward,
        "tool_penalty": args.tool_penalty,
        "min_failures_before_analysis": args.min_failures_before_analysis,
        "failure_db_path": args.failure_db_path,
        "max_failures_per_db": args.max_failures_per_db,
        "analyzer_sample_size": args.analyzer_sample_size,
        "dspy_model": args.dspy_model,
        "dspy_max_tokens": args.dspy_max_tokens,
        "initial_active_tools": args.initial_active_tools,
        "max_tool_call_truncation": args.max_tool_call_truncation,
        "save_failure_db_every_n_steps": args.save_failure_db_every_n_steps,
    }
    rloo_kwargs = dict(
        model_name=args.model_name,
        ref_model_name=args.ref_model_name,
        tokenizer_name=args.tokenizer_name,
        dataset_name=args.dataset_name,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        lr_schedule=args.lr_schedule,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        group_size=args.group_size,
        entropy_coefficient=args.entropy_coefficient,
        kl_divergence_coefficient=args.kl_divergence_coefficient,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.gradient_clipping,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_seqs=args.max_num_seqs,
        num_training_steps=args.num_training_steps,
        max_table_rows=args.max_table_rows,
        save_every_n_steps=args.save_every_n_steps,
        save_dir=args.save_dir,
    )

    ray.init()
    trainer = TIRRLOOTrainer(**tir_kwargs, **rloo_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
