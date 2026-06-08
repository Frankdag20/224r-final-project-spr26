"""Curriculum TIR RLOO: gradually shift training from medium → hard problems.

Subclasses ``TIRRLOOTrainer`` to replace the fixed dataloader with a
curriculum-aware sampler that transitions from 100% medium problems to
100% hard problems over a configurable number of steps.

Supports two curriculum strategies via ``--curriculum_strategy``:
  - ``numcount``: 3-number = medium, 4-number = hard (uses ``numcount_difficulty`` column)
  - ``solvability``: teacher-labeled tool solvability (uses ``solvability`` column)

Also supports ``--curriculum_strategy none`` for a uniform-sampling baseline.

Dataset: ``sbfisher/countdown_curriculum`` (490k problems with both label types).

Usage:

    modal run modal_train.py tir_curriculum -- \
        --dataset_name sbfisher/countdown_curriculum \
        --curriculum_strategy numcount \
        --num_training_steps 150 \
        --curriculum_end_step 110
"""

from __future__ import annotations

import argparse
import os
import random
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

from datasets import load_dataset

from evaluation.countdown import compute_score
from rloo_trainer.rloo_dataset import RLOODataset, get_dataloaders
from tir_extension.training.hierarchical_reward import (
    compute_hierarchical_reward,
    aggregate_hierarchical_metrics,
)
from tir_extension.training.rloo_tir import TIRRLOOTrainer, _inject_tool_system_prompt


class CurriculumTIRRLOOTrainer(TIRRLOOTrainer):
    """TIR RLOO with medium→hard curriculum scheduling."""

    def __init__(
        self,
        curriculum_end_step: int = 110,
        curriculum_strategy: str = "numcount",
        **kwargs,
    ):
        # The curriculum trainer uses _sample_curriculum_batch instead of the
        # parent's dataloader, so we stub out get_dataloaders to avoid loading
        # the curriculum dataset through RLOODataset (which can fail due to HF
        # metadata schema mismatches with extra columns like difficulty_score).
        import rloo_trainer.rloo_dataset as _ds_mod
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        _orig_get_dl = _ds_mod.get_dataloaders

        def _dummy_get_dataloaders(dataset_name, splits=None, **dl_kwargs):
            """Return empty dataloaders — curriculum trainer never uses them."""
            dummy = TensorDataset(torch.zeros(1))
            return {s: DataLoader(dummy) for s in (splits or ["train"])}

        _ds_mod.get_dataloaders = _dummy_get_dataloaders
        super().__init__(**kwargs)
        _ds_mod.get_dataloaders = _orig_get_dl
        self.curriculum_end_step = curriculum_end_step
        self.curriculum_strategy = curriculum_strategy

        # Load dataset and split by difficulty.
        # Use trust_remote_code and ignore HF metadata schema to handle
        # datasets where the parquet has extra columns (e.g. difficulty_score)
        # that aren't in the auto-generated dataset card.
        print(f"[Curriculum] Loading dataset {self.dataset_name} for curriculum split")
        print(f"[Curriculum] Strategy: {curriculum_strategy}")
        try:
            ds = load_dataset(self.dataset_name, split="train")
        except Exception:
            # Schema mismatch — load parquet directly, bypassing HF metadata
            from huggingface_hub import hf_hub_download
            pq_path = hf_hub_download(
                repo_id=self.dataset_name,
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset",
            )
            ds = load_dataset("parquet", data_files=pq_path, split="train")
            print(f"[Curriculum] Loaded via direct parquet (bypassed HF metadata)")

        if curriculum_strategy == "none":
            # No curriculum — all examples go into one pool
            self.all_examples = [
                {"prompt": ds[i]["prompt"], "ground_truth": ds[i]["ground_truth"]}
                for i in range(len(ds))
            ]
            self.medium_examples = self.all_examples
            self.hard_examples = self.all_examples
            print(f"[Curriculum] No curriculum: {len(self.all_examples)} total examples (uniform sampling)")
        elif curriculum_strategy == "hard_only":
            # Tool-Star style: RL on hard (unsolved) problems only
            self.all_examples = [
                {"prompt": ds[i]["prompt"], "ground_truth": ds[i]["ground_truth"]}
                for i in range(len(ds)) if ds[i]["solvability"] == "hard"
            ]
            self.medium_examples = self.all_examples
            self.hard_examples = self.all_examples
            print(f"[Curriculum] Hard-only (Tool-Star): {len(self.all_examples)} examples (uniform sampling)")
        elif curriculum_strategy == "score":
            # Score-based curriculum: sort by difficulty score, ramp eligible
            # pool size from easiest 20% to 100% by count (not threshold).
            all_with_score = [
                (ds[i]["difficulty_score"],
                 {"prompt": ds[i]["prompt"], "ground_truth": ds[i]["ground_truth"]})
                for i in range(len(ds))
            ]
            all_with_score.sort(key=lambda x: x[0])
            self.sorted_examples = [ex for _, ex in all_with_score]

            # Print score distribution
            from collections import Counter
            score_counts = Counter(ds["difficulty_score"])
            for s in sorted(score_counts):
                print(f"[Curriculum] Score {s}: {score_counts[s]} examples")

            # Starting fraction: include easiest 20% at step 0
            self.score_start_frac = 0.2
            n_start = int(self.score_start_frac * len(self.sorted_examples))
            print(f"[Curriculum] Score ramp: {n_start} examples (easiest 20%) at step 0 "
                  f"→ {len(self.sorted_examples)} (100%) at step {self.curriculum_end_step}")

            # For config logging
            self.medium_examples = self.sorted_examples[:n_start]
            self.hard_examples = self.sorted_examples[n_start:]
        else:
            if curriculum_strategy == "numcount":
                diff_col = "numcount_difficulty"
            elif curriculum_strategy == "solvability":
                diff_col = "solvability"
            else:
                raise ValueError(f"Unknown curriculum_strategy: {curriculum_strategy}")

            self.medium_examples = [
                {"prompt": ds[i]["prompt"], "ground_truth": ds[i]["ground_truth"]}
                for i in range(len(ds)) if ds[i][diff_col] == "medium"
            ]
            self.hard_examples = [
                {"prompt": ds[i]["prompt"], "ground_truth": ds[i]["ground_truth"]}
                for i in range(len(ds)) if ds[i][diff_col] == "hard"
            ]
            print(f"[Curriculum] Medium: {len(self.medium_examples)}, Hard: {len(self.hard_examples)}")
            print(f"[Curriculum] Schedule: 100% medium at step 0 → 100% hard at step {self.curriculum_end_step}")

        self.wandb.config.update({
            "curriculum_end_step": curriculum_end_step,
            "curriculum_strategy": curriculum_strategy,
            "curriculum_n_medium": len(self.medium_examples),
            "curriculum_n_hard": len(self.hard_examples),
        })

    def _get_hard_fraction(self, step: int) -> float:
        """Linear ramp from 0.0 to 1.0 over [0, curriculum_end_step]."""
        if step >= self.curriculum_end_step:
            return 1.0
        return step / self.curriculum_end_step

    def _get_score_pool_size(self, step: int) -> int:
        """Linear ramp from start_frac to 1.0 of sorted examples."""
        total = len(self.sorted_examples)
        if step >= self.curriculum_end_step:
            return total
        frac = self.score_start_frac + (1.0 - self.score_start_frac) * step / self.curriculum_end_step
        return max(1, int(frac * total))

    def _sample_curriculum_batch(self, step: int) -> dict:
        """Sample a batch with curriculum-weighted mix of medium and hard."""
        if self.curriculum_strategy in ("none", "hard_only"):
            batch_examples = random.choices(self.all_examples, k=self.batch_size)
        elif self.curriculum_strategy == "score":
            pool_size = self._get_score_pool_size(step)
            batch_examples = random.choices(self.sorted_examples[:pool_size], k=self.batch_size)
        else:
            hard_frac = self._get_hard_fraction(step)
            n_hard = int(round(hard_frac * self.batch_size))
            n_medium = self.batch_size - n_hard
            batch_examples = (
                random.choices(self.medium_examples, k=n_medium)
                + random.choices(self.hard_examples, k=n_hard)
            )
            random.shuffle(batch_examples)

        return {
            "prompt": [ex["prompt"] for ex in batch_examples],
            "ground_truth": [ex["ground_truth"] for ex in batch_examples],
        }

    def train(self):  # type: ignore[override]
        """Run curriculum RLOO training.

        Identical to ``TIRRLOOTrainer.train()`` except data comes from
        ``_sample_curriculum_batch`` instead of the dataloader.
        """
        import random as rng
        import shutil

        last_checkpoint_dir = None
        global_step = 0
        # Collect prompts+gt for self-critic sampling
        self_critic_buffer: list[tuple[str, dict]] = []

        for epoch in range(self.num_epochs):
            if global_step > 0 and global_step == self.num_training_steps:
                break

            # Instead of iterating the dataloader, we generate batches on the fly.
            # Each iteration = one training step (same as original).
            while global_step < self.num_training_steps:
                # ----- Curriculum batch (only difference from original) -----
                batch = self._sample_curriculum_batch(global_step)

                if self.curriculum_strategy == "score":
                    pool_size = self._get_score_pool_size(global_step)
                    print(
                        f"[Curriculum] step={global_step} score_pool_size="
                        f"{pool_size}/{len(self.sorted_examples)} "
                        f"({100*pool_size/len(self.sorted_examples):.1f}%)",
                        flush=True,
                    )
                elif self.curriculum_strategy in ("none", "hard_only"):
                    print(
                        f"[Curriculum] step={global_step} ({self.curriculum_strategy})",
                        flush=True,
                    )
                else:
                    hard_frac = self._get_hard_fraction(global_step)
                    print(
                        f"[Curriculum] step={global_step} hard_frac={hard_frac:.2f} "
                        f"(medium={self.batch_size - int(round(hard_frac * self.batch_size))}, "
                        f"hard={int(round(hard_frac * self.batch_size))})",
                        flush=True,
                    )

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

                all_prompts_raw = batch["prompt"]
                all_ground_truth = batch["ground_truth"]
                assert (
                    len(all_prompts_raw) == len(all_ground_truth) == self.batch_size
                )

                # Inject I^T
                all_prompts = [
                    _inject_tool_system_prompt(p, self._tool_system_prompt)
                    for p in all_prompts_raw
                ]
                all_responses, all_sample_log_probs = ray.get(
                    self.sampling_worker.generate.remote(all_prompts)
                )

                # ----- 2) Rewards -----
                all_rewards: list[list[float]] = []
                all_meta: list[dict] = []
                for resp_group, gt in zip(all_responses, all_ground_truth):
                    group_rewards = []
                    for resp in resp_group:
                        if self.use_vanilla_reward:
                            base = compute_score(resp, gt)
                            group_rewards.append(base)
                            all_meta.append({
                                "base_score": base,
                                "final_reward": base,
                                "good_format": base > 0.0,
                                "accuracy": 1.0 if base >= 1.0 else 0.0,
                                "rM": 0.0,
                                "distinct_relevant_tools": [],
                                "n_distinct_relevant_tools": 0,
                                "n_total_tool_calls": 0,
                            })
                        else:
                            reward, meta = compute_hierarchical_reward(
                                response=resp,
                                ground_truth=gt,
                                active_tools=self.active_tools,
                                multi_tool_bonus=self.multi_tool_bonus,
                                min_tools_for_bonus=self.min_tools_for_bonus,
                            )
                            group_rewards.append(reward)
                            all_meta.append(meta)
                    all_rewards.append(group_rewards)

                reward_mean = float(np.mean(all_rewards))
                base_score_mean = float(
                    np.mean([m["base_score"] for m in all_meta])
                )
                print(
                    f"[TIR] step={global_step} reward_mean={reward_mean:.3f} "
                    f"base_score_mean={base_score_mean:.3f}",
                    flush=True,
                )

                generation_table = self._build_generation_table(
                    all_prompts, all_responses, all_rewards
                )

                # Buffer prompts for self-critic
                if self.self_critic:
                    for p, gt in zip(all_prompts_raw, all_ground_truth):
                        self_critic_buffer.append((p, gt))
                    if len(self_critic_buffer) > self.batch_size * 10:
                        self_critic_buffer = self_critic_buffer[-self.batch_size * 5:]

                # ----- 3) Tokenize -----
                tokenized = self.tokenize_batch_tir(
                    {
                        "prompt": all_prompts,
                        "response": all_responses,
                        "rewards": all_rewards,
                        "sample_log_probs": all_sample_log_probs,
                    }
                )

                # ----- 4) RLOO Update -----
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

                # ----- 5) Checkpoint -----
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

                # ----- 6) Self-Critic DPO (Tool-Star Algorithm 1) -----
                dpo_metrics = None
                if (
                    self.self_critic
                    and (global_step + 1) % self.self_critic_every_k == 0
                    and len(self_critic_buffer) >= self.batch_size
                ):
                    sc_prompts = rng.sample(
                        self_critic_buffer,
                        min(self.batch_size, len(self_critic_buffer)),
                    )
                    sc_model_path = os.path.join(last_checkpoint_dir, "model")
                    dpo_metrics = self._run_self_critic(
                        sc_model_path, sc_prompts, global_step
                    )

                # ----- 7) Logging -----
                hier_metrics = aggregate_hierarchical_metrics(all_meta)
                rollout_accuracy = float(
                    np.mean([1.0 if m["base_score"] >= 1.0 else 0.0 for m in all_meta])
                )

                curriculum_metrics = {}
                if self.curriculum_strategy == "score":
                    pool_size = self._get_score_pool_size(global_step)
                    curriculum_metrics["curriculum/score_pool_frac"] = pool_size / len(self.sorted_examples)
                elif self.curriculum_strategy not in ("none", "hard_only"):
                    curriculum_metrics["curriculum/hard_frac"] = self._get_hard_fraction(global_step)

                log_dict = {
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "sampling/reward_mean": reward_mean,
                    "sampling/base_score_mean": base_score_mean,
                    "sampling/rollout_accuracy": rollout_accuracy,
                    **{f"train/{k}": v for k, v in all_metrics.items()},
                    **hier_metrics,
                    **curriculum_metrics,
                }
                if dpo_metrics is not None:
                    log_dict.update({
                        f"self_critic/{k}": v for k, v in dpo_metrics.items()
                    })
                if generation_table is not None:
                    log_dict["samples/generations"] = generation_table

                self.wandb.log(log_dict, step=global_step)
                global_step += 1

        # Tear down
        if self.sampling_worker is not None:
            ray.kill(self.sampling_worker)
            self.sampling_worker = None
        if self.update_worker is not None:
            ray.kill(self.update_worker)
            self.update_worker = None
        ray.shutdown()
        self.wandb.finish()

        print(f"[Curriculum] Training complete at step {global_step}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Base RLOO args
    parser.add_argument("--model_name", type=str,
                        default="sbfisher/tir-sft-3tool_from_sft")
    parser.add_argument("--ref_model_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str,
                        default="sbfisher/countdown_curriculum")
    parser.add_argument("--wandb_project", type=str, default="tir_rloo_curriculum")
    parser.add_argument("--wandb_name", type=str, default="curriculum_medium_to_hard")
    parser.add_argument("--lr_schedule", type=str, default="constant")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--kl_divergence_coefficient", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_training_steps", type=int, default=150)
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
                        default="/vol/checkpoints/tir_rloo_curriculum")

    # TIR-specific
    parser.add_argument("--initial_active_tools", type=str, default=None)
    parser.add_argument("--max_tool_turns", type=int, default=5)
    parser.add_argument("--multi_tool_bonus", type=float, default=0.1)
    parser.add_argument("--min_tools_for_bonus", type=int, default=2)
    parser.add_argument("--use_vanilla_reward", action="store_true")

    # Self-critic
    parser.add_argument("--self_critic", action="store_true")
    parser.add_argument("--self_critic_every_k", type=int, default=5)
    parser.add_argument("--self_critic_n_samples", type=int, default=8)
    parser.add_argument("--self_critic_beta", type=float, default=0.3)
    parser.add_argument("--self_critic_threshold", type=float, default=1.0)

    # Curriculum-specific
    parser.add_argument("--curriculum_end_step", type=int, default=110,
                        help="Step at which training is 100%% hard problems.")
    parser.add_argument("--curriculum_strategy", type=str, default="numcount",
                        choices=["numcount", "solvability", "score", "none", "hard_only"],
                        help="How to split medium/hard: numcount (3-num/4-num), "
                             "solvability (teacher-labeled tool solvability), "
                             "score (teacher difficulty score 1-10, threshold ramp), "
                             "none (uniform sampling baseline), "
                             "or hard_only (Tool-Star: RL on unsolved problems only).")
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

    tir_kwargs = {
        "initial_active_tools": args.initial_active_tools,
        "max_tool_turns": args.max_tool_turns,
        "multi_tool_bonus": args.multi_tool_bonus,
        "min_tools_for_bonus": args.min_tools_for_bonus,
        "use_vanilla_reward": args.use_vanilla_reward,
        "self_critic": args.self_critic,
        "self_critic_every_k": args.self_critic_every_k,
        "self_critic_n_samples": args.self_critic_n_samples,
        "self_critic_beta": args.self_critic_beta,
        "self_critic_threshold": args.self_critic_threshold,
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
    trainer = CurriculumTIRRLOOTrainer(
        curriculum_end_step=args.curriculum_end_step,
        curriculum_strategy=args.curriculum_strategy,
        **tir_kwargs,
        **rloo_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
