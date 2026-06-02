"""RLOO training loop with Tool-Star hierarchical reward + optional DPO self-critic.

Subclasses ``RLOOTrainer`` to:

1) Execute any ``<use_tool>...</use_tool>`` calls via multi-turn sampling.
2) Score rollouts with ``compute_hierarchical_reward`` (Tool-Star eq. 3).
3) Build a ``tool_result_mask`` so deterministic tool-output tokens
   are excluded from the policy-gradient / entropy / KL terms.
4) Optionally run a DPO self-critic phase every k steps (Tool-Star Algorithm 1).

Designed to run on Modal exactly like the baseline RLOO trainer:

    modal run modal_train.py tir_rloo -- --num_training_steps 250
    modal run modal_train.py tir_rloo -- --num_training_steps 250 --self_critic
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

from evaluation.countdown import compute_score
from rloo_trainer.rloo import RLOOTrainer
from tir_extension.tools.tool_pool import (
    execute_tool_calls,
    find_tool_result_spans,
    relevant_tool_names,
)
from tir_extension.training.hierarchical_reward import (
    compute_hierarchical_reward,
    aggregate_hierarchical_metrics,
)
from tir_extension.training.tir_sampling_worker import TIRSamplingWorker
from tir_extension.tools.system_prompt import build_tool_system_prompt


def _inject_tool_system_prompt(prompt: str, tool_system_prompt: str) -> str:
    """Replace the default system message in a chat-templated prompt with I^T."""
    old_system = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    new_system = f"<|im_start|>system\n{tool_system_prompt}<|im_end|>"
    if old_system in prompt:
        return prompt.replace(old_system, new_system, 1)
    return new_system + "\n" + prompt


class TIRRLOOTrainer(RLOOTrainer):
    """RLOO trainer with hierarchical reward + optional DPO self-critic."""

    def __init__(
        self,
        # Tool config
        initial_active_tools: str | None = None,
        max_tool_turns: int = 5,
        multi_tool_bonus: float = 0.1,
        min_tools_for_bonus: int = 2,
        # Reward mode
        use_vanilla_reward: bool = False,
        # Self-critic config (Tool-Star Algorithm 1)
        self_critic: bool = False,
        self_critic_every_k: int = 5,
        self_critic_n_samples: int = 8,
        self_critic_beta: float = 0.1,
        self_critic_threshold: float = 1.0,
        # Everything else forwarded to RLOOTrainer.
        **rloo_kwargs,
    ):
        super().__init__(**rloo_kwargs)

        self.max_tool_turns = max_tool_turns
        self.multi_tool_bonus = multi_tool_bonus
        self.min_tools_for_bonus = min_tools_for_bonus
        self.use_vanilla_reward = use_vanilla_reward
        self.self_critic = self_critic
        self.self_critic_every_k = self_critic_every_k
        self.self_critic_n_samples = self_critic_n_samples
        self.self_critic_beta = self_critic_beta
        self.self_critic_threshold = self_critic_threshold

        # Parse active tools
        if initial_active_tools:
            self.active_tools: set[str] = {
                name.strip()
                for name in initial_active_tools.split(",")
                if name.strip()
            }
        else:
            self.active_tools = relevant_tool_names()

        # Build tool-integrated instruction (I^T)
        self._tool_system_prompt = build_tool_system_prompt(
            active_tools=self.active_tools
        )
        print(f"[TIR] Tool system prompt (I^T):\n{self._tool_system_prompt}\n")
        print(f"[TIR] Active tools: {sorted(self.active_tools)}")
        print(f"[TIR] Reward mode: {'vanilla (compute_score)' if self.use_vanilla_reward else 'hierarchical'}")
        print(f"[TIR] Self-critic: {self.self_critic}")
        if self.self_critic:
            print(f"[TIR] Self-critic every {self.self_critic_every_k} steps, "
                  f"n_samples={self.self_critic_n_samples}, beta={self.self_critic_beta}")

        self.wandb.config.update({
            "tir_active_tools": sorted(self.active_tools),
            "tir_use_vanilla_reward": use_vanilla_reward,
            "tir_multi_tool_bonus": multi_tool_bonus,
            "tir_min_tools_for_bonus": min_tools_for_bonus,
            "tir_self_critic": self_critic,
            "tir_self_critic_every_k": self_critic_every_k,
            "tir_self_critic_n_samples": self_critic_n_samples,
            "tir_self_critic_beta": self_critic_beta,
        })

    # ------------------------------------------------------------------
    # Multi-turn sampling worker
    # ------------------------------------------------------------------

    def _create_sampling_worker(self, model_path):
        """Create a TIR sampling worker with multi-turn tool execution."""
        if self.update_worker is not None:
            ray.kill(self.update_worker)
            self.update_worker = None

        self.sampling_worker = TIRSamplingWorker.remote(
            model_path=model_path,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_batched_tokens=self.max_num_batched_tokens,
            enable_chunked_prefill=self.enable_chunked_prefill,
            max_num_seqs=self.max_num_seqs,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            group_size=self.group_size,
            max_tool_turns=self.max_tool_turns,
            active_tools=set(self.active_tools),
        )
        ray.get(self.sampling_worker.load_checkpoint.remote())
        return self.sampling_worker

    # ------------------------------------------------------------------
    # Tokenization with tool-result mask
    # ------------------------------------------------------------------

    def tokenize_batch_tir(
        self, batch: dict, group_size_override: int | None = None
    ) -> dict[str, np.ndarray]:
        """Tokenize with tool-result mask."""
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

        self.tokenizer.padding_side = "left"
        tokenized_prompts = self.tokenizer(
            prompts_repeated,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="np",
        )

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

    def _tokenize_single_sequences(
        self, prompts: list[str], responses: list[str]
    ) -> dict[str, np.ndarray]:
        """Tokenize prompt+response pairs (for DPO self-critic)."""
        self.tokenizer.padding_side = "left"
        tok_p = self.tokenizer(
            prompts, add_special_tokens=False, padding=True,
            truncation=True, max_length=self.max_prompt_length, return_tensors="np",
        )
        self.tokenizer.padding_side = "right"
        tok_r = self.tokenizer(
            responses, add_special_tokens=False, padding=True,
            truncation=True, max_length=self.max_response_length,
            return_tensors="np", return_offsets_mapping=True,
        )

        r_ids = tok_r["input_ids"]
        offsets = tok_r["offset_mapping"]
        r_tool_mask = np.zeros_like(r_ids, dtype=np.int64)
        for i, resp in enumerate(responses):
            spans = find_tool_result_spans(resp)
            if not spans:
                continue
            for tok_idx in range(r_ids.shape[1]):
                s, e = int(offsets[i, tok_idx, 0]), int(offsets[i, tok_idx, 1])
                if s == e:
                    continue
                for ss, se in spans:
                    if s < se and e > ss:
                        r_tool_mask[i, tok_idx] = 1
                        break

        input_ids = np.concatenate([tok_p["input_ids"], r_ids], axis=1)
        attention_mask = np.concatenate(
            [tok_p["attention_mask"], tok_r["attention_mask"]], axis=1
        )
        is_response = np.concatenate(
            [np.zeros_like(tok_p["input_ids"]), np.ones_like(r_ids)], axis=1
        )
        tool_mask = np.concatenate(
            [np.zeros_like(tok_p["input_ids"]), r_tool_mask], axis=1
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "is_response_token": is_response,
            "tool_result_mask": tool_mask,
        }

    # ------------------------------------------------------------------
    # DPO Self-Critic (Tool-Star Algorithm 1)
    # ------------------------------------------------------------------

    def _run_self_critic(
        self,
        model_path: str,
        prompts_with_gt: list[tuple[str, dict]],
        global_step: int,
    ) -> dict | None:
        """Self-sample, form preference pairs, run DPO update.

        For each prompt:
        1. Sample n responses with the current policy
        2. Score with hierarchical reward
        3. Responses with reward >= threshold are "chosen", < threshold are "rejected"
        4. Form all (chosen, rejected) pairs and run DPO update
        """
        n = self.self_critic_n_samples

        # Use a subset of prompts to keep compute reasonable
        prompts = [p for p, _ in prompts_with_gt]
        ground_truths = [gt for _, gt in prompts_with_gt]

        # Inject I^T
        prompts_it = [
            _inject_tool_system_prompt(p, self._tool_system_prompt) for p in prompts
        ]

        print(f"[TIR] Self-critic: sampling {n} responses for {len(prompts)} prompts",
              flush=True)

        # Sample from current policy
        self._create_sampling_worker(model_path)
        all_responses, all_logprobs = ray.get(
            self.sampling_worker.generate.remote(prompts_it, n=n)
        )

        # Score with hierarchical reward
        chosen_prompts, chosen_responses = [], []
        rejected_prompts, rejected_responses = [], []

        for prompt, responses, gt in zip(prompts_it, all_responses, ground_truths):
            pos, neg = [], []
            for resp in responses:
                reward, _ = compute_hierarchical_reward(
                    resp, gt,
                    active_tools=self.active_tools,
                    multi_tool_bonus=self.multi_tool_bonus,
                    min_tools_for_bonus=self.min_tools_for_bonus,
                )
                if reward >= self.self_critic_threshold:
                    pos.append(resp)
                else:
                    neg.append(resp)

            # Form pairs: each positive paired with each negative
            for p_resp in pos:
                for n_resp in neg:
                    chosen_prompts.append(prompt)
                    chosen_responses.append(p_resp)
                    rejected_prompts.append(prompt)
                    rejected_responses.append(n_resp)

        n_pairs = len(chosen_prompts)
        if n_pairs == 0:
            print("[TIR] Self-critic: no preference pairs formed (all same label)",
                  flush=True)
            return None

        # Cap pairs to avoid OOM — randomly subsample if too many
        max_pairs = 32
        if n_pairs > max_pairs:
            import random as rng
            indices = rng.sample(range(n_pairs), max_pairs)
            chosen_prompts = [chosen_prompts[i] for i in indices]
            chosen_responses = [chosen_responses[i] for i in indices]
            rejected_prompts = [rejected_prompts[i] for i in indices]
            rejected_responses = [rejected_responses[i] for i in indices]
            n_pairs = max_pairs

        print(f"[TIR] Self-critic: formed {n_pairs} preference pairs", flush=True)

        # Tokenize chosen and rejected
        chosen_tok = self._tokenize_single_sequences(chosen_prompts, chosen_responses)
        rejected_tok = self._tokenize_single_sequences(rejected_prompts, rejected_responses)

        # Need the update worker loaded with current model
        # The sampling worker must be killed first (single GPU)
        optimizer_path = os.path.join(
            self.save_dir, self.wandb_project, self.wandb_name,
            "latest_checkpoint", "optimizer.pt"
        )
        scheduler_path = os.path.join(
            self.save_dir, self.wandb_project, self.wandb_name,
            "latest_checkpoint", "scheduler.pt"
        )
        if not os.path.exists(optimizer_path):
            optimizer_path = None
            scheduler_path = None

        self._create_update_worker(model_path, optimizer_path, scheduler_path)

        dpo_metrics = ray.get(
            self.update_worker.dpo_update.remote(
                chosen_input_ids=chosen_tok["input_ids"],
                chosen_attention_mask=chosen_tok["attention_mask"],
                chosen_is_response_token=chosen_tok["is_response_token"],
                rejected_input_ids=rejected_tok["input_ids"],
                rejected_attention_mask=rejected_tok["attention_mask"],
                rejected_is_response_token=rejected_tok["is_response_token"],
                chosen_tool_result_mask=chosen_tok["tool_result_mask"],
                rejected_tool_result_mask=rejected_tok["tool_result_mask"],
                beta=self.self_critic_beta,
            )
        )

        # Save checkpoint after DPO update
        save_dir = os.path.join(
            self.save_dir, self.wandb_project, self.wandb_name, "latest_checkpoint"
        )
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

        print(f"[TIR] Self-critic DPO metrics: {dpo_metrics}", flush=True)
        return dpo_metrics

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):  # type: ignore[override]
        import random
        import shutil

        last_checkpoint_dir = None
        global_step = 0
        # Collect prompts+gt for self-critic sampling
        self_critic_buffer: list[tuple[str, dict]] = []

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
                            # Vanilla compute_score: 0.0 / 0.1 / 1.0
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
                            # Hierarchical reward (Tool-Star eq. 3)
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
                    # Keep buffer bounded
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
                    # Sample a batch of prompts from buffer
                    import random as rng
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
                # Vanilla-compatible accuracy: fraction of rollouts with
                # base_score == 1.0 (same as what vanilla RLOO tracks via
                # reward_mean when compute_score returns 0/0.1/1.0).
                rollout_accuracy = float(
                    np.mean([1.0 if m["base_score"] >= 1.0 else 0.0 for m in all_meta])
                )
                log_dict = {
                    "train/epoch": epoch,
                    "train/train_iter": train_iter,
                    "train/global_step": global_step,
                    "sampling/reward_mean": reward_mean,
                    "sampling/base_score_mean": base_score_mean,
                    "sampling/rollout_accuracy": rollout_accuracy,
                    **{f"train/{k}": v for k, v in all_metrics.items()},
                    **hier_metrics,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Base RLOO args
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
    parser.add_argument("--weight_decay", type=float, default=1e-4)
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

    # TIR-specific
    parser.add_argument("--initial_active_tools", type=str, default=None,
                        help="Comma-separated tool names. Defaults to all relevant.")
    parser.add_argument("--max_tool_turns", type=int, default=5)
    parser.add_argument("--multi_tool_bonus", type=float, default=0.1)
    parser.add_argument("--min_tools_for_bonus", type=int, default=2)
    parser.add_argument("--use_vanilla_reward", action="store_true",
                        help="Use vanilla compute_score (0/0.1/1.0) instead of hierarchical reward.")

    # Self-critic (Tool-Star Algorithm 1)
    parser.add_argument("--self_critic", action="store_true",
                        help="Enable DPO self-critic phase (Tool-Star Algorithm 1).")
    parser.add_argument("--self_critic_every_k", type=int, default=5,
                        help="Run self-critic every k RLOO steps.")
    parser.add_argument("--self_critic_n_samples", type=int, default=8,
                        help="Number of responses to sample per prompt for self-critic.")
    parser.add_argument("--self_critic_beta", type=float, default=0.1,
                        help="DPO temperature parameter for self-critic.")
    parser.add_argument("--self_critic_threshold", type=float, default=1.0,
                        help="Reward threshold for positive/negative labeling.")
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
    trainer = TIRRLOOTrainer(**tir_kwargs, **rloo_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
