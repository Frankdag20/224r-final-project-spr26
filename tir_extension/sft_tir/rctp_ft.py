"""Stage 1 of RC-GRPO: Reward-Conditioned Trajectory Policy Fine-Tuning (RCTP-FT).

Trains the model on a mixed-quality dataset (successes + failures) where each
prompt is augmented with a discrete reward token:

  [Reward Goal: <|high reward|>]   <- prepended to user turn for score=1.0
  [Reward Goal: <|low reward|>]    <- prepended to user turn for score<1.0

After training the model learns to produce qualitatively different trajectories
depending on the conditioning token.  This checkpoint then serves as the
starting point for Stage 2 (RC-GRPO in rc_grpo.py).

The mixed dataset is produced by:
    python tir_extension/data/generate_failure_trajectories.py \\
        --input_path <success_records.json> \\
        --output_path <mixed.json>

Run on Modal:
    modal run modal_train.py rctp_ft -- \\
        --trajectory_path /vol/checkpoints/tir_rctp_mixed.json \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --output_dir /vol/checkpoints/rctp_ft_pilot
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tir_extension.sft_tir.sft_tir_dataset import (
    SFTToolTrajectoryDataset,
    _tool_result_token_mask,
    _is_pre_formatted,
    get_dataloaders as _base_get_dataloaders,
)

# ---------------------------------------------------------------------------
# Reward token constants (must match rc_grpo.py exactly).
# ---------------------------------------------------------------------------

HIGH_REWARD_TOKEN = "<|high reward|>"
LOW_REWARD_TOKEN = "<|low reward|>"
_REWARD_GOAL_FMT = "\n[Reward Goal: {}]"

# Marker that separates the user turn content from the start of the assistant turn.
# Works for Qwen2.5-style chat templates.
_ASST_MARKER = "<|im_end|>\n<|im_start|>assistant\n"


def inject_reward_token(prompt: str, token: str) -> str:
    """Append the reward-goal string to the user turn of a pre-formatted prompt.

    Inserts "[Reward Goal: <token>]" immediately before the closing <|im_end|>
    of the last user turn, so the model sees it as part of the human message.

    Falls back to appending before the raw `assistant` generation-prompt prefix
    when the Qwen-style markers are absent.
    """
    goal = _REWARD_GOAL_FMT.format(token)
    idx = prompt.rfind(_ASST_MARKER)
    if idx != -1:
        return prompt[:idx] + goal + prompt[idx:]
    # Fallback for non-Qwen templates: insert right before "assistant\n" at end.
    if prompt.endswith("assistant\n"):
        return prompt[:-len("assistant\n")] + goal + "\nassistant\n"
    return prompt + goal


# ---------------------------------------------------------------------------
# RCTP-FT dataset: wraps SFTToolTrajectoryDataset with reward conditioning.
# ---------------------------------------------------------------------------

class RCTPDataset(SFTToolTrajectoryDataset):
    """Extends SFTToolTrajectoryDataset to inject reward tokens.

    Records are expected to have a ``reward_label`` field ("high"/"low") or
    will be assigned one based on their ``score`` field.
    """

    def __init__(self, json_path: str, tokenizer: Any, **kwargs: Any) -> None:
        # keep_only_full_score=False so we also load failure trajectories.
        super().__init__(
            json_path=json_path,
            tokenizer=tokenizer,
            keep_only_full_score=False,
            min_score=0.0,
            **kwargs,
        )
        # Re-inject reward tokens into the already-processed records.
        for rec in self.records:
            score = rec.get("score", 1.0)
            label = rec.get("reward_label") or ("high" if float(score) >= 1.0 else "low")
            token = HIGH_REWARD_TOKEN if label == "high" else LOW_REWARD_TOKEN
            rec["prompt"] = inject_reward_token(rec["prompt"], token)
            rec["reward_label"] = label


def get_dataloaders(
    json_path: str,
    tokenizer: Any,
    batch_size: int = 8,
    max_prompt_length: int = 512,
    max_response_length: int = 1024,
    num_workers: int = 0,
    val_fraction: float = 0.1,
    seed: int = 0,
    gradient_accumulation_steps: int = 1,
):
    """Build train/val DataLoaders over the mixed-quality RCTP dataset."""
    import torch.utils.data as D

    assert batch_size % gradient_accumulation_steps == 0
    micro_bs = batch_size // gradient_accumulation_steps

    full = RCTPDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
    )
    print(
        f"[RCTP] Loaded {len(full)} records "
        f"({sum(1 for r in full.records if r.get('reward_label') == 'high')} high, "
        f"{sum(1 for r in full.records if r.get('reward_label') == 'low')} low)",
        flush=True,
    )

    if val_fraction <= 0 or len(full) < 2:
        return {
            "train": D.DataLoader(
                full, batch_size=micro_bs, shuffle=True,
                collate_fn=full.collate_fn, num_workers=num_workers,
            )
        }

    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub = D.random_split(full, [n_train, n_val], generator=generator)
    return {
        "train": D.DataLoader(
            train_sub, batch_size=micro_bs, shuffle=True,
            collate_fn=full.collate_fn, num_workers=num_workers,
        ),
        "val": D.DataLoader(
            val_sub, batch_size=micro_bs, shuffle=False,
            collate_fn=full.collate_fn, num_workers=num_workers,
        ),
    }


# ---------------------------------------------------------------------------
# Trainer (mirrors sft_tir.py closely).
# ---------------------------------------------------------------------------

def get_model(model_name: str, device: str = "cuda", use_gradient_checkpointing: bool = True):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    return model, tokenizer


def _build_labels(batch: dict, device: str):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    is_response = batch["is_response_token"].to(device).bool()
    tool_mask = batch["tool_result_mask"].to(device).bool()

    labels = input_ids.clone()
    # Mask out prompt tokens and tool-result tokens from the loss.
    labels[~is_response] = -100
    labels[tool_mask] = -100
    return input_ids, attention_mask, labels


def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "model")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        os.path.join(output_dir, "train_states.pth"),
    )
    print(f"Saved checkpoint to {output_dir}", flush=True)


def train(
    model_name: str,
    trajectory_path: str,
    output_dir: str,
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    max_prompt_length: int = 512,
    max_response_length: int = 1024,
    warmup_ratio: float = 0.05,
    gradient_accumulation_steps: int = 1,
    val_fraction: float = 0.1,
    wandb_project: str = "rctp_ft",
    wandb_name: str = "rctp_ft_run",
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project=wandb_project, name=wandb_name, config=locals())

    model, tokenizer = get_model(model_name)
    dataloaders = get_dataloaders(
        json_path=trajectory_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        val_fraction=val_fraction,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("val")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        for micro_step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = _build_labels(batch, device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (micro_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                log = {"train/loss": loss.item() * gradient_accumulation_steps,
                       "train/lr": scheduler.get_last_lr()[0],
                       "train/epoch": epoch,
                       "train/global_step": global_step}

                if val_loader and global_step % 50 == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vb in val_loader:
                            vi, va, vl = _build_labels(vb, device)
                            vout = model(input_ids=vi, attention_mask=va, labels=vl)
                            val_losses.append(vout.loss.item())
                    log["val/loss"] = sum(val_losses) / len(val_losses)
                    model.train()

                wandb.log(log, step=global_step)
                if global_step % 20 == 0:
                    print(
                        f"[RCTP] epoch={epoch} step={global_step} "
                        f"loss={log['train/loss']:.4f}",
                        flush=True,
                    )
                global_step += 1

        gc.collect()
        torch.cuda.empty_cache()

    save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)
    wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RCTP-FT: Stage 1 of RC-GRPO.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trajectory_path", type=str, required=True,
                        help="Mixed-quality JSON from generate_failure_trajectories.py.")
    parser.add_argument("--output_dir", type=str,
                        default="/vol/checkpoints/rctp_ft_checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="rctp_ft")
    parser.add_argument("--wandb_name", type=str, default="rctp_ft_run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
