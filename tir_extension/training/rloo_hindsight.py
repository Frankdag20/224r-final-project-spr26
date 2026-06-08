"""Hindsight RLOO: oracle-augmented tool-integrated RL.

Extends ``TIRRLOOTrainer`` with a Hindsight Experience Replay (HER) step
applied at the group level.

Motivation
----------
RLOO (and GRPO) require *within-group reward variance* to produce a non-zero
policy gradient. When the base model is weak (~2-6% accuracy) and rewards are
sparse (binary 0/1), most groups are all-zero => zero gradient => no learning.

Fix
---
After computing rewards, for every group where *all* rollouts received reward 0:

1. Invoke the brute-force Countdown oracle (``countdown_solver.solve_countdown``)
   to find a correct arithmetic expression.
2. Build a synthetic oracle response in the TIR format
   ``<use_tool>calculator: expr</use_tool><tool_result>...</tool_result>
   <answer>expr</answer>``
3. Replace *one* of the failed rollouts in the group with this oracle rollout
   (reward = 1.0).

This guarantees within-group variance for previously unsolvable groups, giving
the policy a gradient that pushes *toward* the oracle trajectory and *away from*
the replaced failure. Group size stays fixed at G (no batch shape changes).

IS-weight note
--------------
The oracle rollout was not sampled by the current policy, so its
``sample_log_probs`` is unknown. When the batch contains oracle rollouts we
disable IS-weighting for the *entire* step (``sample_log_probs=None`` =>
standard policy gradient with weights=1). The resulting variance increase is
acceptable: oracle augmentation only fires on the hardest groups, and the
added gradient signal on those groups outweighs the variance cost.
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import ray

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")

from tir_extension.training.rloo_tir import TIRRLOOTrainer, _build_parser as _tir_build_parser
from tir_extension.training.countdown_solver import solve_countdown, build_oracle_response
from tir_extension.training.failure_aware_reward import (
    aggregate_tool_metrics,
    compute_score_with_tools,
)
from tir_extension.training.dynamic_tool_selector import DynamicToolSelector
from tir_extension.tools.tool_pool import execute_tool_calls


class HindsightTIRRLOOTrainer(TIRRLOOTrainer):
    """TIR-RLOO with Hindsight Experience Replay for all-zero reward groups."""

    def __init__(self, oracle_replace_idx: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.oracle_replace_idx = oracle_replace_idx
        self.wandb.config.update(
            {"hindsight/oracle_replace_idx": oracle_replace_idx},
            allow_val_change=True,
        )

    # ------------------------------------------------------------------
    # Oracle augmentation helper
    # ------------------------------------------------------------------

    def _augment_with_oracle(
        self,
        all_responses: list[list[str]],
        all_rewards: list[list[float]],
        all_ground_truth: list[dict],
        active_tools: set[str],
    ) -> tuple[list[list[str]], list[list[float]], bool, int]:
        """Replace one rollout in each all-zero group with an oracle solution.

        Returns (aug_responses, aug_rewards, any_oracle, n_augmented).
        """
        aug_responses = [list(r) for r in all_responses]
        aug_rewards = [list(r) for r in all_rewards]
        n_augmented = 0

        for i, (group_rewards, gt) in enumerate(zip(aug_rewards, all_ground_truth)):
            if max(group_rewards) > 0.0:
                continue

            expr = solve_countdown(gt["target"], gt["numbers"])
            if expr is None:
                continue

            oracle_resp = build_oracle_response(expr, gt["target"])
            oracle_score, _ = compute_score_with_tools(
                response=oracle_resp,
                ground_truth=gt,
                active_tools=active_tools,
                tool_reward=self.tool_reward,
                tool_penalty=self.tool_penalty,
            )

            slot = self.oracle_replace_idx % len(group_rewards)
            aug_responses[i][slot] = oracle_resp
            aug_rewards[i][slot] = oracle_score
            n_augmented += 1

        return aug_responses, aug_rewards, n_augmented > 0, n_augmented

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):  # type: ignore[override]
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
                    f"[HindsightTIR] Sampling epoch={epoch} step={global_step}",
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
                    f"len(all_prompts)={len(all_prompts)}, "
                    f"len(all_ground_truth)={len(all_ground_truth)}, "
                    f"self.batch_size={self.batch_size}"
                )
                active_tools = set(self.tool_selector.active_tools)

                if self.multi_turn:
                    all_responses, all_sample_log_probs = ray.get(
                        self.sampling_worker.generate_multi_turn.remote(
                            all_prompts,
                            active_tools=list(active_tools),
                            max_tool_calls=self.max_tool_call_truncation,
                        )
                    )
                else:
                    all_raw_responses, all_sample_log_probs = ray.get(
                        self.sampling_worker.generate.remote(all_prompts)
                    )
                    all_responses = [
                        [
                            execute_tool_calls(
                                r,
                                active_tools=active_tools,
                                max_calls=self.max_tool_call_truncation,
                            )
                            for r in group
                        ]
                        for group in all_raw_responses
                    ]

                # ----- 2) Tool-aware rewards -----
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

                reward_mean_pre = float(np.mean(all_rewards))
                base_score_mean = float(np.mean([m["base_score"] for m in all_meta]))

                # ----- 3) Hindsight oracle augmentation -----
                all_responses, all_rewards, any_oracle, n_augmented = (
                    self._augment_with_oracle(
                        all_responses,
                        all_rewards,
                        all_ground_truth,
                        active_tools,
                    )
                )

                reward_mean = float(np.mean(all_rewards))
                oracle_fraction = n_augmented / max(len(all_rewards), 1)
                print(
                    f"[HindsightTIR] step={global_step} "
                    f"reward_pre={reward_mean_pre:.3f} reward_post={reward_mean:.3f} "
                    f"base={base_score_mean:.3f} oracle_groups={n_augmented} "
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
                # Disable IS-weighting when oracle rollouts are present because
                # their sample_log_probs are not from the current policy.
                effective_sample_log_probs = (
                    None if any_oracle else tokenized["sample_log_probs"]
                )

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
                        sample_log_probs=effective_sample_log_probs,
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
                            f"[HindsightTIR] DSPy reanalysis: {reanalysis_payload}",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[HindsightTIR] reanalysis failed: {exc}", flush=True)

                # ----- 9) Periodic failure-DB persistence -----
                if (
                    self.failure_db_path
                    and self.save_failure_db_every_n_steps > 0
                    and (global_step + 1) % self.save_failure_db_every_n_steps == 0
                ):
                    try:
                        self.failure_db.save(self.failure_db_path)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[HindsightTIR] failed to save failure DB: {exc}", flush=True)

                # ----- 10) Logging -----
                tool_metrics = aggregate_tool_metrics(all_meta)
                failure_counts = self.failure_db.failure_type_counts()
                log_dict = {
                    "train/epoch": epoch,
                    "train/train_iter": train_iter,
                    "train/global_step": global_step,
                    "sampling/reward_mean_pre_oracle": reward_mean_pre,
                    "sampling/reward_mean": reward_mean,
                    "sampling/base_score_mean": base_score_mean,
                    "hindsight/oracle_groups": n_augmented,
                    "hindsight/oracle_fraction": oracle_fraction,
                    "hindsight/is_disabled": float(any_oracle),
                    "tir/failure_db_size": len(self.failure_db),
                    "tir/n_active_tools": len(self.tool_selector.active_tools),
                    "tir/active_tools": ",".join(sorted(self.tool_selector.active_tools)),
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

        # Teardown — _create_* already swap workers, so only the last one is alive.
        if self.sampling_worker is not None:
            ray.kill(self.sampling_worker)
            self.sampling_worker = None
        if self.update_worker is not None:
            ray.kill(self.update_worker)
            self.update_worker = None
        ray.shutdown()
        self.wandb.finish()


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------


def _build_parser():
    parser = _tir_build_parser()
    parser.description = "Hindsight RLOO: oracle-augmented TIR training."
    parser.set_defaults(
        wandb_project="hindsight_tir_project",
        wandb_name="hindsight_run",
        save_dir="/vol/checkpoints/hindsight_tir_checkpoints",
    )
    parser.add_argument(
        "--oracle_replace_idx",
        type=int,
        default=0,
        help="Which group slot to replace with the oracle rollout (default=0).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        raise ValueError(
            "Cannot set both --enable_chunked_prefill and --disable_chunked_prefill."
        )
    enable_chunked_prefill = not args.disable_chunked_prefill

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
        "multi_turn": bool(args.multi_turn),
        "oracle_replace_idx": args.oracle_replace_idx,
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
    trainer = HindsightTIRRLOOTrainer(**tir_kwargs, **rloo_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
