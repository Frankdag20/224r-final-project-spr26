"""Self-Critic GRPO/RLOO trainer for Countdown TIR.

Extends ``TIRRLOOTrainer`` with two additions:

1. **Hierarchical Reward** (Tool-Star eq. 3):
       R = -1       bad format (no <answer> tags)
       R =  0       valid format, wrong answer
       R =  1.0     correct answer
       R =  1.1     correct answer + >= 2 distinct relevant tools used
   This larger reward range [-1, 1.1] gives RLOO much stronger gradient
   signal than the previous clipped [0, 1] range.

2. **Self-Critic GRPO Phase** (Tool-Star Algorithm 1, adapted):
   Every ``self_critic_every_k_steps`` RL steps the trainer:
     a) Samples ``self_critic_n_samples`` responses per query for a random
        batch of ``self_critic_n_queries`` prompts from the training set.
     b) Scores each response with the hierarchical reward.
        Responses with reward >= 1.0 are "positive"; rest are "negative".
     c) Runs one full RLOO update with these self-evaluated rollouts, using
        ``self_critic_n_samples`` as the group size so the leave-one-out
        baseline is computed within the larger group.

   The self-critic phase teaches the policy to distinguish good from bad
   trajectories based on its *own* outputs, not just the curated training set.

Modal usage:

    modal run modal_train.py tir_self_critic -- --num_training_steps 300
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import ray
import wandb

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

from tir_extension.training.rloo_tir import TIRRLOOTrainer
from tir_extension.data.failure_database import DEFAULT_DB_PATH
from tir_extension.tools.tool_pool import (
    TOOL_REGISTRY,
    execute_tool_calls,
    relevant_tool_names,
)
from tir_extension.training.dynamic_tool_selector import DynamicToolSelector
from tir_extension.training.hierarchical_reward import (
    aggregate_hierarchical_metrics,
    compute_hierarchical_reward,
)
from tir_extension.training.failure_aware_reward import aggregate_tool_metrics


class SelfCriticTIRTrainer(TIRRLOOTrainer):
    """RLOO + hierarchical reward + periodic self-critic GRPO phase."""

    def __init__(
        self,
        self_critic_every_k_steps: int = 10,
        self_critic_n_samples: int = 8,
        self_critic_n_queries: int = 4,
        multi_tool_bonus: float = 0.1,
        min_tools_for_bonus: int = 2,
        # Curriculum: start easy (1 tool), tighten to min_tools_for_bonus
        # once this fraction of num_training_steps is reached. Set to 0.0
        # to disable curriculum (use min_tools_for_bonus from the start).
        curriculum_phase2_frac: float = 0.5,
        **tir_kwargs,
    ):
        # Disable the old tool_reward / tool_penalty shaping from TIRRLOOTrainer
        # since we now use the hierarchical formula exclusively.
        tir_kwargs.setdefault("tool_reward", 0.0)
        tir_kwargs.setdefault("tool_penalty", 0.0)
        super().__init__(**tir_kwargs)

        self.self_critic_every_k_steps = self_critic_every_k_steps
        self.self_critic_n_samples = self_critic_n_samples
        self.self_critic_n_queries = self_critic_n_queries
        self.multi_tool_bonus = multi_tool_bonus
        self.min_tools_for_bonus = min_tools_for_bonus
        self.curriculum_phase2_frac = curriculum_phase2_frac

        self.wandb.config.update(
            {
                "sc_every_k_steps": self_critic_every_k_steps,
                "sc_n_samples": self_critic_n_samples,
                "sc_n_queries": self_critic_n_queries,
                "sc_multi_tool_bonus": multi_tool_bonus,
                "sc_min_tools_for_bonus": min_tools_for_bonus,
                "sc_curriculum_phase2_frac": curriculum_phase2_frac,
            }
        )

    # ------------------------------------------------------------------
    # Curriculum: gradually tighten the multi-tool bonus threshold.
    # ------------------------------------------------------------------

    def _current_min_tools(self, global_step: int) -> int:
        """Return the active min_tools_for_bonus for this step.

        Phase 1 (steps 0 .. phase2_start-1): require only 1 tool — easy
            signal that encourages any tool use at all.
        Phase 2 (steps phase2_start .. end): tighten to min_tools_for_bonus
            (default 2) — reward collaborative multi-tool use.

        Set curriculum_phase2_frac=0.0 to skip phase 1 entirely.
        """
        if self.curriculum_phase2_frac <= 0.0:
            return self.min_tools_for_bonus
        phase2_start = int(self.num_training_steps * self.curriculum_phase2_frac)
        if global_step < phase2_start:
            return 1
        return self.min_tools_for_bonus

    # ------------------------------------------------------------------
    # Reward helper (replaces the clipped tool-aware reward).
    # ------------------------------------------------------------------

    def _score_responses(
        self,
        all_responses: list[list[str]],
        all_ground_truth: list[dict],
        active_tools: set[str],
        global_step: int = 0,
    ) -> tuple[list[list[float]], list[dict]]:
        """Score all response groups with the hierarchical reward."""
        min_tools = self._current_min_tools(global_step)
        all_rewards: list[list[float]] = []
        all_meta: list[dict] = []
        for resp_group, gt in zip(all_responses, all_ground_truth):
            group_rewards = []
            for resp in resp_group:
                reward, meta = compute_hierarchical_reward(
                    response=resp,
                    ground_truth=gt,
                    active_tools=active_tools,
                    multi_tool_bonus=self.multi_tool_bonus,
                    min_tools_for_bonus=min_tools,
                )
                group_rewards.append(reward)
                all_meta.append(meta)
            all_rewards.append(group_rewards)
        return all_rewards, all_meta

    # ------------------------------------------------------------------
    # Self-critic GRPO phase.
    # ------------------------------------------------------------------

    def _run_self_critic_phase(
        self,
        global_step: int,
        model_path: str,
        optimizer_path: str | None,
        scheduler_path: str | None,
        all_train_prompts: list[str],
        all_train_gt: list[dict],
    ) -> dict:
        """Sample many rollouts and run one RLOO update with hierarchical reward.

        Returns a dict of scalar metrics for W&B logging.
        """
        n_q = min(self.self_critic_n_queries, len(all_train_prompts))
        indices = random.sample(range(len(all_train_prompts)), n_q)
        sc_prompts = [all_train_prompts[i] for i in indices]
        sc_gt = [all_train_gt[i] for i in indices]

        active_tools = set(self.tool_selector.active_tools)

        # --- sample self_critic_n_samples responses per query ---
        self._create_sampling_worker(model_path)
        if self.multi_turn:
            sc_responses, sc_sample_lp = ray.get(
                self.sampling_worker.generate_multi_turn.remote(
                    sc_prompts,
                    active_tools=list(active_tools),
                    max_tool_calls=self.max_tool_call_truncation,
                    n=self.self_critic_n_samples,
                )
            )
        else:
            sc_raw_responses, sc_sample_lp = ray.get(
                self.sampling_worker.generate.remote(
                    sc_prompts, n=self.self_critic_n_samples
                )
            )
            sc_responses = [
                [execute_tool_calls(r, active_tools=active_tools,
                                    max_calls=self.max_tool_call_truncation)
                 for r in group]
                for group in sc_raw_responses
            ]

        # --- hierarchical reward ---
        sc_rewards, sc_meta = self._score_responses(sc_responses, sc_gt, active_tools,
                                                     global_step=global_step)

        n_pos = sum(1 for g in sc_rewards for r in g if r >= 1.0)
        n_neg = sum(1 for g in sc_rewards for r in g if r < 1.0)
        print(
            f"[SC] step={global_step} self-critic phase: "
            f"n_queries={n_q} n_samples={self.self_critic_n_samples} "
            f"positive={n_pos} negative={n_neg} "
            f"mean_reward={np.mean(sc_rewards):.3f}",
            flush=True,
        )

        # --- add failures to DB ---
        for prompt, resp_group, gt, grp_rewards in zip(sc_prompts, sc_responses, sc_gt, sc_rewards):
            for resp, score in zip(resp_group, grp_rewards):
                self.failure_db.add(
                    prompt=prompt,
                    response=resp,
                    ground_truth=gt,
                    score=score,
                    step=global_step,
                )

        # --- tokenize (reuse TIR tokenizer with tool-result mask) ---
        tokenized = self.tokenize_batch_tir(
            {
                "prompt": sc_prompts,
                "response": sc_responses,
                "rewards": sc_rewards,
                "sample_log_probs": sc_sample_lp,
            },
            group_size_override=self.self_critic_n_samples,
        )

        # --- update: one RLOO step with self_critic_n_samples group size ---
        self._create_update_worker(model_path, optimizer_path, scheduler_path)
        sc_metrics = ray.get(
            self.update_worker.update_gradient_accumulation.remote(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                is_response_token=tokenized["is_response_token"],
                rewards=tokenized["rewards"],
                sample_log_probs=tokenized["sample_log_probs"],
                tool_result_mask=tokenized["tool_result_mask"],
                group_size_override=self.self_critic_n_samples,
            )
        )

        hier_metrics = aggregate_hierarchical_metrics(sc_meta)
        return {
            **{f"sc/{k}": v for k, v in sc_metrics.items()},
            **{f"sc/{k}": v for k, v in hier_metrics.items()},
            "sc/n_positive": n_pos,
            "sc/n_negative": n_neg,
        }

    # ------------------------------------------------------------------
    # Main training loop (overrides TIRRLOOTrainer.train).
    # ------------------------------------------------------------------

    def train(self):  # type: ignore[override]
        # Cache the full training set for self-critic query sampling.
        all_train_data = list(self.train_dataloader.dataset)
        all_train_prompts = [d["prompt"] for d in all_train_data]
        all_train_gt = [d["ground_truth"] for d in all_train_data]

        last_checkpoint_dir = None
        global_step = 0

        for epoch in range(self.num_epochs):
            if global_step >= self.num_training_steps:
                break
            for train_iter, batch in enumerate(self.train_dataloader):
                if global_step >= self.num_training_steps:
                    break

                # =========================================================
                # 1) Regular TIR RLOO step (hierarchical reward).
                # =========================================================
                print(
                    f"[SC] epoch={epoch} step={global_step} — RLOO step",
                    flush=True,
                )
                model_path = (
                    self.model_name
                    if last_checkpoint_dir is None
                    else os.path.join(last_checkpoint_dir, "model")
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

                self._create_sampling_worker(model_path)
                all_prompts = batch["prompt"]
                all_ground_truth = batch["ground_truth"]

                all_raw_responses, all_sample_lp = ray.get(
                    self.sampling_worker.generate.remote(all_prompts)
                )

                active_tools = set(self.tool_selector.active_tools)
                if self.multi_turn:
                    all_responses, all_sample_lp = ray.get(
                        self.sampling_worker.generate_multi_turn.remote(
                            all_prompts,
                            active_tools=list(active_tools),
                            max_tool_calls=self.max_tool_call_truncation,
                        )
                    )
                    all_sample_log_probs = all_sample_lp
                else:
                    all_responses = [
                        [execute_tool_calls(r, active_tools=active_tools,
                                            max_calls=self.max_tool_call_truncation)
                         for r in group]
                        for group in all_raw_responses
                    ]

                all_rewards, all_meta = self._score_responses(
                    all_responses, all_ground_truth, active_tools,
                    global_step=global_step,
                )

                reward_mean = float(np.mean(all_rewards))
                print(
                    f"[SC] step={global_step} reward_mean={reward_mean:.3f} "
                    f"active_tools={sorted(active_tools)}",
                    flush=True,
                )

                # Failure DB.
                for prompt, resp_group, gt, grp_rewards in zip(
                    all_prompts, all_responses, all_ground_truth, all_rewards
                ):
                    for resp, score in zip(resp_group, grp_rewards):
                        self.failure_db.add(
                            prompt=prompt, response=resp,
                            ground_truth=gt, score=score, step=global_step,
                        )

                generation_table = self._build_generation_table(
                    all_prompts, all_responses, all_rewards
                )

                tokenized = self.tokenize_batch_tir(
                    {
                        "prompt": all_prompts,
                        "response": all_responses,
                        "rewards": all_rewards,
                        "sample_log_probs": all_sample_lp,
                    }
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

                # Checkpoint.
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

                # Periodic DSPy re-analysis.
                reanalysis_payload = None
                if DynamicToolSelector.should_reanalyze(global_step + 1, self.reanalyze_every_n_steps):
                    try:
                        analyzer = self._get_analyzer()
                        reanalysis_payload = self.tool_selector.reanalyze(
                            failure_db=self.failure_db,
                            analyzer=analyzer,
                            step=global_step,
                            n_samples=self.analyzer_sample_size,
                            min_failures=self.min_failures_before_analysis,
                        )
                        print(f"[SC] DSPy reanalysis: {reanalysis_payload}", flush=True)
                    except Exception as exc:
                        print(f"[SC] reanalysis failed: {exc}", flush=True)

                # Periodic failure-DB persistence.
                if (
                    self.failure_db_path
                    and self.save_failure_db_every_n_steps > 0
                    and (global_step + 1) % self.save_failure_db_every_n_steps == 0
                ):
                    try:
                        self.failure_db.save(self.failure_db_path)
                    except Exception as exc:
                        print(f"[SC] failed to save failure DB: {exc}", flush=True)

                hier_metrics = aggregate_hierarchical_metrics(all_meta)
                failure_counts = self.failure_db.failure_type_counts()
                log_dict = {
                    "train/epoch": epoch,
                    "train/train_iter": train_iter,
                    "train/global_step": global_step,
                    "sampling/reward_mean": reward_mean,
                    "tir/failure_db_size": len(self.failure_db),
                    "tir/n_active_tools": len(self.tool_selector.active_tools),
                    "tir/active_tools": ",".join(sorted(self.tool_selector.active_tools)),
                    "curriculum/min_tools_for_bonus": self._current_min_tools(global_step),
                    "curriculum/phase": 1 if self._current_min_tools(global_step) == 1 else 2,
                    **{f"train/{k}": v for k, v in all_metrics.items()},
                    **hier_metrics,
                    **{f"tir/failure_type_{name}": count for name, count in failure_counts.items()},
                }
                if reanalysis_payload is not None:
                    log_dict["tir/reanalysis_step"] = reanalysis_payload.get("step", global_step)
                if generation_table is not None:
                    log_dict["samples/generations"] = generation_table

                # =========================================================
                # 2) Self-critic GRPO phase (every K steps).
                # =========================================================
                sc_log_dict: dict = {}
                if (global_step + 1) % self.self_critic_every_k_steps == 0:
                    print(
                        f"[SC] step={global_step} — self-critic phase",
                        flush=True,
                    )
                    try:
                        sc_log_dict = self._run_self_critic_phase(
                            global_step=global_step,
                            model_path=os.path.join(last_checkpoint_dir, "model"),
                            optimizer_path=os.path.join(last_checkpoint_dir, "optimizer.pt"),
                            scheduler_path=os.path.join(last_checkpoint_dir, "scheduler.pt"),
                            all_train_prompts=all_train_prompts,
                            all_train_gt=all_train_gt,
                        )

                        # Save updated checkpoint after self-critic step.
                        ray.get(
                            self.update_worker.update_checkpoint_paths.remote(
                                model_path=save_model_path,
                                optimizer_path=save_optimizer_path,
                                scheduler_path=save_scheduler_path,
                                load_checkpoint=False,
                            )
                        )
                        ray.get(self.update_worker.save_checkpoint.remote())

                    except Exception as exc:
                        print(f"[SC] self-critic phase failed: {exc}", flush=True)

                log_dict.update(sc_log_dict)
                self.wandb.log(log_dict, step=global_step)
                global_step += 1

        # Final cleanup.
        if self.failure_db_path:
            try:
                self.failure_db.save(self.failure_db_path)
            except Exception as exc:
                print(f"[SC] failed to save failure DB at exit: {exc}", flush=True)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="asingh15/qwen-sft-countdown-defaultproj")
    parser.add_argument("--ref_model_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str,
                        default="asingh15/countdown_tasks_3to4")
    parser.add_argument("--wandb_project", type=str, default="tir_self_critic_project")
    parser.add_argument("--wandb_name", type=str, default="tir_sc_test")
    parser.add_argument("--lr_schedule", type=str, default="constant")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--kl_divergence_coefficient", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_training_steps", type=int, default=300)
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
                        default="/vol/checkpoints/tir_sc_checkpoints")
    # TIR base flags.
    parser.add_argument("--reanalyze_every_n_steps", type=int, default=3)
    parser.add_argument("--min_failures_before_analysis", type=int, default=50)
    parser.add_argument("--failure_db_path", type=str, default=DEFAULT_DB_PATH)
    parser.add_argument("--max_failures_per_db", type=int, default=5000)
    parser.add_argument("--analyzer_sample_size", type=int, default=20)
    parser.add_argument("--dspy_model", type=str, default=None)
    parser.add_argument("--dspy_max_tokens", type=int, default=None)
    parser.add_argument("--initial_active_tools", type=str, default=None)
    parser.add_argument("--max_tool_call_truncation", type=int, default=16)
    parser.add_argument("--save_failure_db_every_n_steps", type=int, default=5)
    # Self-critic specific.
    parser.add_argument("--self_critic_every_k_steps", type=int, default=10,
                        help="Run self-critic GRPO phase every K RL steps.")
    parser.add_argument("--self_critic_n_samples", type=int, default=8,
                        help="Responses sampled per query in self-critic phase.")
    parser.add_argument("--self_critic_n_queries", type=int, default=4,
                        help="Queries used per self-critic phase.")
    parser.add_argument("--multi_tool_bonus", type=float, default=0.1,
                        help="rM bonus for using >= min_tools_for_bonus tools.")
    parser.add_argument("--min_tools_for_bonus", type=int, default=2,
                        help="Distinct relevant tools needed to earn rM.")
    parser.add_argument("--curriculum_phase2_frac", type=float, default=0.5,
                        help="Fraction of total steps before tightening tool bonus threshold. 0=disable.")
    parser.add_argument("--multi_turn", type=int, default=1,
                        help="1=multi-turn tool use (model sees real results), 0=single-turn legacy.")
    return parser


def main():
    args = _build_parser().parse_args()

    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        raise ValueError(
            "Cannot set both --enable_chunked_prefill and --disable_chunked_prefill."
        )
    enable_chunked_prefill = not args.disable_chunked_prefill

    tir_kwargs = {
        "reanalyze_every_n_steps": args.reanalyze_every_n_steps,
        "tool_reward": 0.0,
        "tool_penalty": 0.0,
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
    sc_kwargs = {
        "self_critic_every_k_steps": args.self_critic_every_k_steps,
        "self_critic_n_samples": args.self_critic_n_samples,
        "self_critic_n_queries": args.self_critic_n_queries,
        "multi_tool_bonus": args.multi_tool_bonus,
        "min_tools_for_bonus": args.min_tools_for_bonus,
        "curriculum_phase2_frac": args.curriculum_phase2_frac,
        "multi_turn": bool(args.multi_turn),
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
    trainer = SelfCriticTIRTrainer(**sc_kwargs, **tir_kwargs, **rloo_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
