"""Starter SFT training entrypoint for the class project.

This file is intentionally incomplete. Students are expected to implement
`train(...)` while reusing the data/model setup provided here.
"""

import sys
from pathlib import Path

# Allow `python sft_trainer/sft.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from sft_trainer.sft_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device='cuda', use_gradient_checkpointing=True):
    """Load policy model + tokenizer for SFT training."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enable gradient checkpointing to reduce memory (trades compute for memory)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.train()
    return model, tokenizer

def clear_cache(model):
    """Best-effort GPU/CPU cache cleanup between heavy steps."""
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir):
    """Save model/tokenizer plus optimizer/scheduler states."""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

    torch.save({
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(output_dir, 'train_states.pth'))
    print(f"Model saved to {output_dir}")

def evaluate(model, test_dataloader, device, global_step):
    model.eval()
    loss_sum_e = 0.0
    loss_count_e = 0
    num_correct_e = 0
    total_e = 0

    with torch.inference_mode():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            is_response_token = batch['is_response_token'].to(device).bool()

            # Build labels: ignore prompt tokens by setting them to -100 
            labels = input_ids.clone()
            labels[~is_response_token] = -100

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss

            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]

            resp_mask = shifted_labels != -100
            n_response_tokens = resp_mask.sum().item()
            loss_sum_e += loss.item() * n_response_tokens
            loss_count_e += n_response_tokens

            preds = shifted_logits.argmax(dim=-1)
            num_correct_e += ((preds == shifted_labels) & resp_mask).sum().item()
            total_e += resp_mask.sum().item()

    avg_eval_loss = loss_sum_e / max(loss_count_e, 1)
    avg_eval_accuracy = num_correct_e / max(total_e, 1)

    print(f'Eval loss {avg_eval_loss:.4f}. Eval accuracy {avg_eval_accuracy:.4f}.')

    wandb.log(
        {'sft_eval_loss': avg_eval_loss, 'sft_eval_accuracy': avg_eval_accuracy},
        step=global_step,
    )

def train(
    model, 
    tokenizer, 
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0
):
    # Expected high-level flow:
    # 1) Forward pass on `input_ids` and compute token-level log-probs.
    # 2) Mask loss to response tokens only using `is_response_token`.
    # 3) Backprop, optionally clip gradients, then optimizer/scheduler steps.
    # 4) Periodically evaluate on `test_dataloader` and log metrics to W&B.
    # 5) Save checkpoints under `output_dir` when requested.

    global_step = 0 # for tracking gradient update steps
    for e in tqdm.tqdm(range(num_epochs), desc="epoch"):
        model.train()
        optimizer.zero_grad()
        loss_sum = 0.0
        loss_count = 0
        num_correct = 0
        total = 0

        for idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            is_response_token = batch['is_response_token'].to(device).bool()

            # Build labels: ignore prompt tokens by setting them to -100 
            labels = input_ids.clone()
            labels[~is_response_token] = -100

            # pass in labels so HuggingFace CausalLMLoss can compute the loss
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits


            # Accumulate gradients and scale by gradient_accumulation_steps
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

            if (idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # log loss per step
                wandb.log(
                    {
                        'sft_train_loss_step': loss.item(),
                        'lr': scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

        # calculate average loss and accuracy per epoch
        avg_loss = loss_sum / max(loss_count, 1)
        avg_accuracy = num_correct / max(total, 1)

        print('================')
        print(f'Epoch {e}/{num_epochs}.')
        print(f'Train loss {avg_loss:.4f}. Train accuracy {avg_accuracy:.4f}.')

        # log to wandb, taken from HW2
        wandb.log(
            {'sft_train_loss_epoch': avg_loss, 'sft_train_accuracy': avg_accuracy},
            step=global_step,
        )

        ########## EVALUATION ##########
        # evaluate on test dataloader every epoch or on the final epoch
        # can change to evaluate every x epochs by changing the % 1 to % x
        if e % 1 == 0 or e == num_epochs - 1:
            evaluate(model, test_dataloader, device, global_step)

        clear_cache(model)

    # save model checkpoint after training
    if save_model == 1:
        save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)









def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='Asap7772/cog_behav_all_strategies')
    parser.add_argument('--output_dir', type=str, default='sft_model')
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_response_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='sft_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

    dataloaders = get_dataloaders(
        dataset_name=args.dataset_name, 
        tokenizer=tokenizer, 
        max_prompt_length=args.max_prompt_length, 
        max_response_length=args.max_response_length, 
        batch_size=args.batch_size, 
        splits=['train', 'test'],
        pin_memory=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dataloader, test_dataloader = dataloaders['train'], dataloaders['test']
    # Scheduler steps happen only after an optimizer step, so account for
    # gradient accumulation when estimating total training steps.
    num_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

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
        args.gradient_clipping
    )

if __name__ == "__main__":
    main()
