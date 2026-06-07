"""Stage 2 of RC-GRPO: Reward-Conditioned Group Relative Policy Optimization.

Subclasses TIRRLOOTrainer.  The single change to the training loop is:

  For each GRPO group of G rollouts, sample a reward token per rollout
  from Psample(r) (p fraction <|high reward|>, 1-p fraction <|low reward|>),
  inject it into the prompt, and generate ONE response per conditioned prompt.

  Because some rollouts in the group are steered toward high quality and
  others toward low quality, the within-group reward variance is guaranteed
  to be non-zero even after a strong SFT warm-start — restoring informative
  GRPO advantages (see Proposition 4.3 in the RC-GRPO paper).

The rest of the RLOO mechanics (importance-weighted PG, KL, entropy, tool
masking) are inherited unchanged from TIRRLOOTrainer / RLOOTrainer.

CLI (test with 5 steps):
    modal run modal_train.py rc_grpo -- \\
        --model_name /vol/checkpoints/rctp_ft_checkpoint/model \\
        --num_training_steps 5 \\
        --batch_size 4 --group_size 4 \\
        --rc_grpo_p 0.5 \\
        --wandb_name rc_grpo_test
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import ray

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tir_extension.training.rloo_tir import TIRRLOOTrainer
from tir_extension.training.failure_aware_reward import (
    aggregate_tool_metrics,
    compute_score_with_tools,
)
from tir_extension.sft_tir.rctp_ft import HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN, inject_reward_token


class RCGRPOTrainer(TIRRLOOTrainer):
    """TIR-RLOO trainer with reward-conditioned rollout generation (RC-GRPO)."""

    def __init__(
        self,
        rc_grpo_p: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # p = proportion of <|high reward|> tokens per group.
        self.rc_grpo_p = rc_grpo_p
        self.wandb.config.update({"rc_grpo_p": rc_grpo_p}, allow_val_change=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_reward_tokens(self, n: int) -> list[str]:
        """Sample n reward tokens from Psample(r) = p*(high) + (1-p)*(low)."""
        return [
            HIGH_REWARD_TOKEN if random.random() < self.rc_grpo_p else LOW_REWARD_TOKEN
            for _ in range(n)
        ]

    def _conditioned_prompts(
        self, prompts: list[str]
    ) -> tuple[list[str], list[list[str]]]:
        """Return (flat_conditioned, tokens_per_group).

        flat_conditioned : batch_size * group_size prompts, each with its reward
                           token injected into the user turn.
        tokens_per_group : batch_size x group_size list of the tokens used.
        """
        tokens_per_group: list[list[str]] = []
        flat_conditioned: list[str] = []
        for prompt in prompts:
            group_tokens = self._sample_reward_tokens(self.group_size)
            tokens_per_group.append(group_tokens)
            for tok in group_tokens:
                flat_conditioned.append(inject_reward_token(prompt, tok))
        return flat_conditioned, tokens_per_group

    # ------------------------------------------------------------------
    # Tokenization override: accept pre-expanded conditioned prompts.
    # ------------------------------------------------------------------

    def tokenize_batch_rc_grpo(
        self,
        conditioned_flat: list[str],
        responses: list[list[str]],
        rewards: list[list[float]],
        sample_log_probs: list[list[float]],
    ) -> dict[str, np.ndarray]:
        """Like tokenize_batch_tir but prompts are already expanded 1-per-rollout."""
        return self.tokenize_batch_tir(
            {
                "prompt": conditioned_flat,
                "response": responses,
                "rewards": rewards,
                "sample_log_probs": sample_log_probs,
            },
            group_size_override=1,
        )

    # ------------------------------------------------------------------
    # Main training loop (overrides TIRRLOOTrainer.train).
    # ------------------------------------------------------------------

    def train(self):  # type: ignore[override]
        import shutil

        last_checkpoint_dir = None
        global_step = 0

        for epoch in range(self.num_epochs):
            if global_step >= self.num_training_steps:
                break

            for train_iter, batch in enumerate(self.train_dataloader):
                if global_step >= self.num_training_steps:
                    break

                # ----- 1) Build conditioned prompts -----
                all_prompts = batch["prompt"]          # batch_size original prompts
                all_ground_truth = batch["ground_truth"]
                active_tools = set(self.tool_selector.active_tools)

                print(
                    f"[RC-GRPO] Sampling epoch={epoch} step={global_step} "
                    f"batch={len(all_prompts)} group={self.group_size} p={self.rc_grpo_p}",
                    flush=True,
                )

                # Each rollout gets its own reward-token-conditioned prompt.
                flat_conditioned, tokens_per_group = self._conditioned_prompts(all_prompts)

                model_path = (
                    self.model_name
                    if last_checkpoint_dir is None
                    else os.path.join(last_checkpoint_dir, "model")
                )
                self._create_sampling_worker(model_path)

                # ----- 2) Generate 1 response per conditioned prompt -----
                if self.multi_turn:
                    flat_responses, flat_logprobs = ray.get(
                        self.sampling_worker.generate_multi_turn.remote(
                            flat_conditioned,
                            active_tools=list(active_tools),
                            max_tool_calls=self.max_tool_call_truncation,
                            n=1,
                        )
                    )
                else:
                    from tir_extension.tools.tool_pool import execute_tool_calls
                    flat_raw, flat_logprobs = ray.get(
                        self.sampling_worker.generate.remote(flat_conditioned, n=1)
                    )
                    flat_responses = [
                        [execute_tool_calls(r[0], active_tools=active_tools,
                                            max_calls=self.max_tool_call_truncation)]
                        for r in flat_raw
                    ]

                # flat_responses: (batch_size * group_size) x 1
                # Reshape to (batch_size) x (group_size).
                bs, gs = len(all_prompts), self.group_size
                all_responses: list[list[str]] = []
                all_sample_log_probs: list[list[float]] = []
                for i in range(bs):
                    all_responses.append(
                        [flat_responses[i * gs + j][0] for j in range(gs)]
                    )
                    all_sample_log_probs.append(
                        [flat_logprobs[i * gs + j][0] for j in range(gs)]
                    )

                # ----- 3) Reward -----
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

                reward_mean = float(np.mean([r for group in all_rewards for r in group]))
                base_score_mean = float(np.mean([m["base_score"] for m in all_meta]))

                # Log within-group advantage spread (key RC-GRPO health metric).
                advantage_spreads = []
                for group_r in all_rewards:
                    if len(group_r) > 1:
                        advantage_spreads.append(max(group_r) - min(group_r))
                mean_adv_spread = float(np.mean(advantage_spreads)) if advantage_spreads else 0.0

                # Fraction of high/low tokens this step.
                flat_tokens = [t for group in tokens_per_group for t in group]
                frac_high = flat_tokens.count(HIGH_REWARD_TOKEN) / len(flat_tokens)

                print(
                    f"[RC-GRPO] step={global_step} reward={reward_mean:.3f} "
                    f"base={base_score_mean:.3f} adv_spread={mean_adv_spread:.3f} "
                    f"frac_high={frac_high:.2f}",
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

                # ----- 5) Tokenize (conditioned prompts, 1 response each) -----
                tokenized = self.tokenize_batch_rc_grpo(
                    conditioned_flat=flat_conditioned,
                    responses=all_responses,
                    rewards=all_rewards,
                    sample_log_probs=all_sample_log_probs,
                )

                # ----- 6) Update -----
                # Kill the sampling worker first so vLLM releases GPU memory
                # before the update worker loads the model. Without this, both
                # workers compete for GPU memory and load_checkpoint stalls long
                # enough to trigger the Ray gRPC keepalive watchdog timeout.
                if self.sampling_worker is not None:
                    ray.kill(self.sampling_worker)
                    self.sampling_worker = None

                optimizer_path = (
                    None if last_checkpoint_dir is None
                    else os.path.join(last_checkpoint_dir, "optimizer.pt")
                )
                scheduler_path = (
                    None if last_checkpoint_dir is None
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
                        self.save_dir, self.wandb_project, self.wandb_name,
                        f"epoch_{epoch}_step_{global_step}",
                    )
                else:
                    save_dir = os.path.join(
                        self.save_dir, self.wandb_project, self.wandb_name,
                        "latest_checkpoint",
                    )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                ray.get(
                    self.update_worker.update_checkpoint_paths.remote(
                        model_path=os.path.join(save_dir, "model"),
                        optimizer_path=os.path.join(save_dir, "optimizer.pt"),
                        scheduler_path=os.path.join(save_dir, "scheduler.pt"),
                        load_checkpoint=False,
                    )
                )
                ray.get(self.update_worker.save_checkpoint.remote())
                last_checkpoint_dir = save_dir

                # ----- 8) Periodic DSPy re-analysis -----
                reanalysis_payload = None
                from tir_extension.training.dynamic_tool_selector import DynamicToolSelector
                if DynamicToolSelector.should_reanalyze(
                    global_step + 1, self.reanalyze_every_n_steps
                ):
                    try:
                        reanalysis_payload = self.tool_selector.reanalyze(
                            failure_db=self.failure_db,
                            analyzer=self._get_analyzer(),
                            step=global_step,
                            n_samples=self.analyzer_sample_size,
                            min_failures=self.min_failures_before_analysis,
                        )
                    except Exception as exc:
                        print(f"[RC-GRPO] reanalysis failed: {exc}", flush=True)

                # ----- 9) Failure DB persistence -----
                if (
                    self.failure_db_path
                    and self.save_failure_db_every_n_steps > 0
                    and (global_step + 1) % self.save_failure_db_every_n_steps == 0
                ):
                    try:
                        self.failure_db.save(self.failure_db_path)
                    except Exception as exc:
                        print(f"[RC-GRPO] failure DB save failed: {exc}", flush=True)

                # ----- 10) Logging -----
                tool_metrics = aggregate_tool_metrics(all_meta)
                failure_counts = self.failure_db.failure_type_counts()
                log_dict = {
                    "train/epoch": epoch,
                    "train/train_iter": train_iter,
                    "train/global_step": global_step,
                    "sampling/reward_mean": reward_mean,
                    "sampling/base_score_mean": base_score_mean,
                    "rc_grpo/advantage_spread_mean": mean_adv_spread,
                    "rc_grpo/frac_high_reward_token": frac_high,
                    "tir/failure_db_size": len(self.failure_db),
                    "tir/n_active_tools": len(self.tool_selector.active_tools),
                    **{f"train/{k}": v for k, v in all_metrics.items()},
                    **tool_metrics,
                    **{f"tir/failure_type_{k}": v for k, v in failure_counts.items()},
                }
                if reanalysis_payload is not None:
                    log_dict["tir/reanalysis_step"] = reanalysis_payload.get(
                        "step", global_step
                    )
                if generation_table is not None:
                    log_dict["samples/generations"] = generation_table
                self.wandb.log(log_dict, step=global_step)

                global_step += 1

        # Cleanup
        if self.failure_db_path:
            try:
                self.failure_db.save(self.failure_db_path)
            except Exception:
                pass
        if self.sampling_worker is not None:
            ray.kill(self.sampling_worker)
            self.sampling_worker = None
        if self.update_worker is not None:
            ray.kill(self.update_worker)
            self.update_worker = None
        ray.shutdown()
        self.wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    from tir_extension.training.rloo_tir import _build_parser as _tir_parser
    parser = _tir_parser()
    # Override defaults to sensible RC-GRPO starting points.
    parser.description = "RC-GRPO: reward-conditioned GRPO (Stage 2)."
    parser.set_defaults(
        wandb_project="rc_grpo_project",
        wandb_name="rc_grpo_run",
        save_dir="/vol/checkpoints/rc_grpo_checkpoints",
    )
    parser.add_argument("--rc_grpo_p", type=float, default=0.5,
                        help="Fraction of rollouts conditioned on <|high reward|>.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        raise ValueError("Cannot set both --enable_chunked_prefill and --disable_chunked_prefill.")
    enable_chunked_prefill = not args.disable_chunked_prefill

    tir_kwargs = dict(
        reanalyze_every_n_steps=args.reanalyze_every_n_steps,
        tool_reward=args.tool_reward,
        tool_penalty=args.tool_penalty,
        min_failures_before_analysis=args.min_failures_before_analysis,
        failure_db_path=args.failure_db_path,
        max_failures_per_db=args.max_failures_per_db,
        analyzer_sample_size=args.analyzer_sample_size,
        dspy_model=args.dspy_model,
        dspy_max_tokens=args.dspy_max_tokens,
        initial_active_tools=args.initial_active_tools,
        max_tool_call_truncation=args.max_tool_call_truncation,
        save_failure_db_every_n_steps=args.save_failure_db_every_n_steps,
        multi_turn=bool(args.multi_turn),
    )
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

    ray.init(
        _system_config={
            "grpc_keepalive_time_ms": 30000,
            "grpc_keepalive_timeout_ms": 120000,
        }
    )
    trainer = RCGRPOTrainer(rc_grpo_p=args.rc_grpo_p, **tir_kwargs, **rloo_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
