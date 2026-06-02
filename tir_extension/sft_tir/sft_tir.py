"""SFT warm-start over tool-using Countdown trajectories.

This script is a near-clone of ``sft_trainer/sft.py`` with two changes:

- The training data comes from a local JSON of tool-using trajectories
  produced by ``generate_trajectories.py`` rather than a HuggingFace
  dataset of plain ``(query, completion)`` pairs.
- Tokens inside ``<tool_result>...</tool_result>`` spans are excluded
  from the loss (in addition to the usual prompt-token masking) so the
  policy is never trained on tokens it didn't actually generate.

The resulting checkpoint is intended as the warm-start for the TIR RLOO
trainer (``tir_extension/training/rloo_tir.py``).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# Make the top-level project importable when this file is run as
# ``python tir_extension/sft_tir/sft_tir.py`` on Modal.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tir_extension.sft_tir.sft_tir_dataset import get_dataloaders


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
        print("Gradient checkpointing enabled")
    model.train()
    return model, tokenizer


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(output_dir, "train_states.pth"),
    )
    print(f"Saved checkpoint to {output_dir}")


def _build_labels(batch: dict[str, torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(input_ids, attention_mask, labels)`` with TIR masking applied."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    is_response = batch["is_response_token"].to(device).bool()
    tool_mask = batch["tool_result_mask"].to(device).bool()

    labels = input_ids.clone()
    # Standard prompt-token mask + new tool-result mask.
    ignore = (~is_response) | tool_mask
    labels[ignore] = -100
    return input_ids, attention_mask, labels


def evaluate(model, dataloader, device: str, global_step: int):
    if dataloader is None:
        return
    model.eval()
    loss_sum = 0.0
    loss_count = 0
    num_correct = 0
    total = 0
    with torch.inference_mode():
        for batch in dataloader:
            input_ids, attention_mask, labels = _build_labels(batch, device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            resp_mask = shifted_labels != -100
            n_response_tokens = resp_mask.sum().item()
            loss_sum += loss.item() * n_response_tokens
            loss_count += n_response_tokens
            preds = shifted_logits.argmax(dim=-1)
            num_correct += ((preds == shifted_labels) & resp_mask).sum().item()
            total += resp_mask.sum().item()
    avg_loss = loss_sum / max(loss_count, 1)
    avg_acc = num_correct / max(total, 1)
    print(f"Eval loss {avg_loss:.4f}. Eval accuracy {avg_acc:.4f}.")
    wandb.log(
        {"tir_sft_eval_loss": avg_loss, "tir_sft_eval_accuracy": avg_acc},
        step=global_step,
    )
    model.train()


def train(
    model,
    tokenizer,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    num_epochs: int,
    device: str = "cuda",
    save_model: int = 1,
    output_dir: str = "tir_sft_model",
    gradient_accumulation_steps: int = 1,
    gradient_clipping: float = 1.0,
):
    global_step = 0
    for e in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss_sum = 0.0
        loss_count = 0
        num_correct = 0
        total = 0
        tool_mask_tokens = 0

        for idx, batch in enumerate(train_dataloader):
            input_ids, attention_mask, labels = _build_labels(batch, device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            (loss / gradient_accumulation_steps).backward()

            with torch.no_grad():
                shifted_logits = logits[:, :-1, :]
                shifted_labels = labels[:, 1:]
                resp_mask = shifted_labels != -100
                n_response_tokens = resp_mask.sum().item()
                loss_sum += loss.item() * n_response_tokens
                loss_count += n_response_tokens
                preds = shifted_logits.argmax(dim=-1)
                num_correct += ((preds == shifted_labels) & resp_mask).sum().item()
                total += resp_mask.sum().item()
                tool_mask_tokens += batch["tool_result_mask"].sum().item()

            if (idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                wandb.log(
                    {
                        "tir_sft_train_loss_step": loss.item(),
                        "tir_sft_tool_mask_tokens_step": int(batch["tool_result_mask"].sum().item()),
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

        avg_loss = loss_sum / max(loss_count, 1)
        avg_acc = num_correct / max(total, 1)
        print(
            f"Epoch {e}/{num_epochs}. Train loss {avg_loss:.4f}. "
            f"Train accuracy {avg_acc:.4f}. Tool-result tokens skipped: {tool_mask_tokens}"
        )
        wandb.log(
            {
                "tir_sft_train_loss_epoch": avg_loss,
                "tir_sft_train_accuracy": avg_acc,
                "tir_sft_tool_mask_tokens_epoch": int(tool_mask_tokens),
            },
            step=global_step,
        )

        evaluate(model, test_dataloader, device, global_step)
        clear_cache()

    if save_model == 1:
        save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="asingh15/qwen-sft-countdown-defaultproj")
    parser.add_argument("--trajectory_path", type=str,
                        default="/vol/checkpoints/tir_trajectories.json")
    parser.add_argument("--output_dir", type=str,
                        default="/vol/checkpoints/tir_sft_checkpoints")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="tir_sft_project")
    parser.add_argument("--wandb_name", type=str, default="tir_sft_test")
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--keep_only_full_score", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--hf_repo", type=str, default="",
                        help="If set, push final checkpoint to this HuggingFace repo (e.g. 'username/model-name').")
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer = get_model(
        args.model_name,
        args.device,
        use_gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    dataloaders = get_dataloaders(
        json_path=args.trajectory_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        keep_only_full_score=args.keep_only_full_score,
        val_fraction=args.val_fraction,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dataloader = dataloaders["train"]
    test_dataloader = dataloaders.get("test")

    num_steps = (
        len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    )
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(num_steps, 1),
    )

    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_name)
    os.makedirs(full_output_dir, exist_ok=True)

    train(
        model,
        tokenizer,
        train_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        args.num_epochs,
        args.device,
        args.save_model,
        full_output_dir,
        args.gradient_accumulation_steps,
        args.gradient_clipping,
    )

    if args.hf_repo:
        print(f"Pushing checkpoint to HuggingFace: {args.hf_repo}")
        model_dir = os.path.join(full_output_dir, "model")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
        hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        hf_model.push_to_hub(args.hf_repo)
        hf_tokenizer.push_to_hub(args.hf_repo)
        print(f"Pushed to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
