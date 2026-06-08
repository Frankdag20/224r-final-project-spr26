"""Curriculum TIR RLOO: gradually shift training from medium → hard problems.

Subclasses ``TIRRLOOTrainer`` to replace the fixed dataloader with a
curriculum-aware sampler that transitions from 100% medium problems to
100% hard problems over a configurable number of steps.

Usage:

    modal run modal_train.py tir_curriculum -- \
        --dataset_name mih123/RL_calc_training_set \
        --model_name sbfisher/tir-sft-3tool_from_sft \
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
from tir_extension.training.rloo_tir import TIRRLOOTrainer, _inject_tool_system_prompt


class CurriculumTIRRLOOTrainer(TIRRLOOTrainer):
    """TIR RLOO with medium→hard curriculum scheduling."""

    def __init__(
        self,
        curriculum_end_step: int = 110,
        **kwargs,
    ):
        # Override dataset loading to handle missing test split.
        # We patch before super().__init__ so the base class doesn't crash.
        original_get_dataloaders = get_dataloaders.__wrapped__ if hasattr(get_dataloaders, '__wrapped__') else None

        import rloo_trainer.rloo_dataset as _ds_mod
        _orig_get_dl = _ds_mod.get_dataloaders

        def _safe_get_dataloaders(dataset_name, splits=None, **dl_kwargs):
            """Load only available splits; create empty loader for missing ones."""
            from datasets import get_dataset_split_names
            try:
                available = set(get_dataset_split_names(dataset_name))
            except Exception:
                available = {"train"}
            safe_splits = [s for s in (splits or ["train"]) if s in available]
            result = _orig_get_dl(dataset_name, splits=safe_splits, **dl_kwargs)
            # Provide a dummy test loader if missing
            for s in (splits or []):
                if s not in result:
                    result[s] = _orig_get_dl(dataset_name, splits=["train"], **dl_kwargs)["train"]
            return result

        _ds_mod.get_dataloaders = _safe_get_dataloaders
        super().__init__(**kwargs)
        _ds_mod.get_dataloaders = _orig_get_dl
        self.curriculum_end_step = curriculum_end_step

        # Load dataset and split by difficulty
        print(f"[Curriculum] Loading dataset {self.dataset_name} for curriculum split")
        ds = load_dataset(self.dataset_name, split="train")

        self.medium_examples = [
            {"prompt": ds[i]["prompt"], "ground_truth": ds[i]["ground_truth"]}
            for i in range(len(ds)) if ds[i]["difficulty"] == "medium"
        ]
        self.hard_examples = [
            {"prompt": ds[i]["prompt"], "ground_truth": ds[i]["ground_truth"]}
            for i in range(len(ds)) if ds[i]["difficulty"] == "hard"
        ]
        print(f"[Curriculum] Medium: {len(self.medium_examples)}, Hard: {len(self.hard_examples)}")
        print(f"[Curriculum] Schedule: 100% medium at step 0 → 100% hard at step {self.curriculum_end_step}")

        self.wandb.config.update({
            "curriculum_end_step": curriculum_end_step,
            "curriculum_n_medium": len(self.medium_examples),
            "curriculum_n_hard": len(self.hard_examples),
        })

    def _get_hard_fraction(self, step: int) -> float:
        """Linear ramp from 0.0 to 1.0 over [0, curriculum_end_step]."""
        if step >= self.curriculum_end_step:
            return 1.0
        return step / self.curriculum_end_step

    def _sample_curriculum_batch(self, step: int) -> dict:
        """Sample a batch with curriculum-weighted mix of medium and hard."""
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

    def train(self):
        """Run curriculum RLOO training."""
        import shutil

        last_checkpoint_dir = None
        global_step = 0

        # Collect prompts+gt for self-critic sampling
        self_critic_buffer: list[tuple[str, dict]] = []

        while global_step < self.num_training_steps:
            # ----- Sample curriculum batch -----
            hard_frac = self._get_hard_fraction(global_step)
            batch = self._sample_curriculum_batch(global_step)

            print(
                f"[Curriculum] step={global_step} hard_frac={hard_frac:.2f} "
                f"(medium={self.batch_size - int(round(hard_frac * self.batch_size))}, "
                f"hard={int(round(hard_frac * self.batch_size))})",
                flush=True,
            )

            # ----- 1) Sample -----
            print(
                f"[TIR] Sampling, step={global_step}",
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
            assert len(all_prompts_raw) == len(all_ground_truth) == self.batch_size

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
                        from tir_extension.training.hierarchical_reward import compute_hierarchical_reward
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
            accuracy_mean = float(
                np.mean([m["accuracy"] for m in all_meta])
            )
            print(
                f"[TIR] step={global_step} reward_mean={reward_mean:.3f} "
                f"base_score_mean={base_score_mean:.3f} accuracy={accuracy_mean:.3f}",
                flush=True,
            )

            generation_table = self._build_generation_table(
                all_prompts, all_responses, all_rewards
            )

            # Buffer prompts for self-critic
            if self.self_critic:
                for p, gt in zip(all_prompts_raw, all_ground_truth):
                    self_critic_buffer.append((p, gt))
                if len(self_critic_buffer) > 64:
                    self_critic_buffer = self_critic_buffer[-64:]

            # ----- 3) Tokenize + Update -----
            tokenized = self._tokenize_rollouts_tir(
                batch={"prompt": all_prompts},
                all_responses=all_responses,
                all_rewards=all_rewards,
                all_sample_log_probs=all_sample_log_probs,
            )

            save_dir = os.path.join(
                self.save_dir, self.wandb_project, self.wandb_name, "latest_checkpoint"
            )
            os.makedirs(save_dir, exist_ok=True)
            save_model_path = os.path.join(save_dir, "model")
            save_optimizer_path = os.path.join(save_dir, "optimizer.pt")
            save_scheduler_path = os.path.join(save_dir, "scheduler.pt")

            self._create_update_worker(
                model_path=model_path,
                save_model_path=save_model_path,
                save_optimizer_path=save_optimizer_path,
                save_scheduler_path=save_scheduler_path,
            )

            update_metrics = ray.get(
                self.update_worker.update.remote(tokenized)
            )
            ray.get(self.update_worker.save_checkpoint.remote())

            # ----- 4) Log -----
            agg_metrics = {}
            from tir_extension.training.hierarchical_reward import aggregate_hierarchical_metrics
            agg_metrics = aggregate_hierarchical_metrics(all_meta)

            log_dict = {
                "reward_mean": reward_mean,
                "base_score_mean": base_score_mean,
                "accuracy_mean": accuracy_mean,
                "curriculum_hard_frac": hard_frac,
                **{f"update/{k}": v for k, v in update_metrics.items()},
                **{f"reward/{k}": v for k, v in agg_metrics.items()},
                "generation_table": generation_table,
            }
            self.wandb.log(log_dict, step=global_step)

            last_checkpoint_dir = save_dir

            # ----- 5) Self-critic -----
            if (
                self.self_critic
                and global_step > 0
                and global_step % self.self_critic_every_k == 0
                and len(self_critic_buffer) >= self.batch_size
            ):
                sc_prompts = random.sample(
                    self_critic_buffer, min(len(self_critic_buffer), self.batch_size)
                )
                dpo_metrics = self._run_self_critic(
                    model_path=save_model_path,
                    prompts_with_gt=sc_prompts,
                    global_step=global_step,
                )
                if dpo_metrics:
                    self.wandb.log(
                        {f"self_critic/{k}": v for k, v in dpo_metrics.items()},
                        step=global_step,
                    )

            # ----- 6) Periodic save -----
            if self.save_every_n_steps > 0 and global_step % self.save_every_n_steps == 0:
                step_save_dir = os.path.join(
                    self.save_dir, self.wandb_project, self.wandb_name,
                    f"step_{global_step}"
                )
                if os.path.exists(save_dir):
                    shutil.copytree(save_dir, step_save_dir, dirs_exist_ok=True)
                    print(f"[Curriculum] Saved step {global_step} checkpoint to {step_save_dir}")

            global_step += 1

        print(f"[Curriculum] Training complete at step {global_step}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Base RLOO args
    parser.add_argument("--model_name", type=str,
                        default="sbfisher/tir-sft-3tool_from_sft")
    parser.add_argument("--ref_model_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str,
                        default="mih123/RL_calc_training_set")
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
        **tir_kwargs,
        **rloo_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
