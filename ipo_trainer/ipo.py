"""Starter IPO training entrypoint for the class project.

This script wires model loading, data loading, and optimizer setup.
Students are expected to implement `train(...)` for the IPO objective.
"""

import sys
from pathlib import Path

# Allow `python ipo_trainer/ipo.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from ipo_trainer.ipo_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
import copy
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device, use_gradient_checkpointing=True):
    """Load trainable policy model and frozen reference model."""
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

    # IPO compares policy preferences to a fixed baseline policy.
    reference_model = copy.deepcopy(model)
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    return model, tokenizer, reference_model

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



def compute_logps(model, input_ids, attention_mask, is_response_token):
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:].clone()
    resp_mask = is_response_token[:, 1:].bool()

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, 
                        index=shifted_targets.unsqueeze(-1)).squeeze(-1)

    token_log_probs_masked = token_log_probs * resp_mask

    summed = token_log_probs_masked.sum(dim=-1)

    length = resp_mask.sum(dim=-1).float().clamp(min=1)
    
    return summed / length # OR SHOULD WE RETURN THE SUMMED AND THEN AVERAGE LATER?
    

def compute_loss(logps_w, logps_l, logps_ref_w, logps_ref_l, beta):
    term_1 = logps_w - logps_ref_w
    term_2 = logps_l - logps_ref_l
    h = term_1 - term_2
    loss = ((h-1/(2*beta))**2).mean()

    reward_margin = (beta*h).mean().item()

    return loss, reward_margin

def evaluate(model, reference_model, test_dataloader, device, beta, global_step):
    model.eval()
    loss_sum_e = 0.0
    loss_count_e = 0
    eval_reward_margin_sum = 0.0

    with torch.inference_mode():
        for batch in test_dataloader:
            input_ids_w = batch['input_ids_w'].to(device)
            attention_mask_w = batch['attention_mask_w'].to(device)
            is_response_token_w = batch['is_response_token_w'].to(device).bool()
            input_ids_l = batch['input_ids_l'].to(device)
            attention_mask_l = batch['attention_mask_l'].to(device)
            is_response_token_l = batch['is_response_token_l'].to(device).bool()

            logps_w = compute_logps(model, input_ids_w, attention_mask_w, is_response_token_w)
            logps_l = compute_logps(model, input_ids_l, attention_mask_l, is_response_token_l)

            logps_ref_w = compute_logps(reference_model, input_ids_w, attention_mask_w, is_response_token_w)
            logps_ref_l = compute_logps(reference_model, input_ids_l, attention_mask_l, is_response_token_l)

            loss_e, eval_reward_margin = compute_loss(logps_w, logps_l, logps_ref_w, logps_ref_l, beta)

            loss_sum_e += loss_e.item()
            loss_count_e += 1
            eval_reward_margin_sum += eval_reward_margin

    avg_eval_loss = loss_sum_e / max(loss_count_e, 1)
    avg_eval_reward_margin = eval_reward_margin_sum / max(loss_count_e, 1)
    if global_step % 500 == 0:
        print(f'Eval loss {avg_eval_loss:.4f}.')
        print(f'Eval reward margin {avg_eval_reward_margin:.4f}.')
        print('--------------------------------')

    wandb.log(
        {'ipo_eval_loss': avg_eval_loss, 
        'ipo_eval_reward_margin': avg_eval_reward_margin},
        step=global_step,
    ) 
    

def train(
    model, 
    tokenizer, 
    reference_model,
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0,
    beta=0.1,
    average_logps=False,
    loss_type='ipo',
):
    # TODO(student): implement IPO/DPO-style pairwise optimization.
    # Expected high-level flow:
    # 1) Compute policy log-probs for chosen/rejected responses.
    # 2) Compute frozen-reference log-probs for chosen/rejected responses.
    # 3) Build the pairwise objective (IPO or related variant).
    # 4) Apply gradient accumulation, clipping, logging, and checkpointing.
    # raise NotImplementedError("This function is not implemented")

    global_step = 0 # for tracking gradient update steps
    for e in tqdm.tqdm(range(num_epochs), desc="epoch"):
        model.train()
        optimizer.zero_grad()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch_reward_margin_sum = 0.0
        loss_sum = 0.0
        reward_margin_sum = 0.0
        loss_count = 0
        
        for idx, batch in enumerate(train_dataloader):
            
            if idx % 500 == 0:
                print(f'Batch {idx} of {len(train_dataloader)}')
                print('--------------------------------')

            input_ids_w = batch['input_ids_w'].to(device)
            attention_mask_w = batch['attention_mask_w'].to(device)
            is_response_token_w = batch['is_response_token_w'].to(device).bool()
            input_ids_l = batch['input_ids_l'].to(device)
            attention_mask_l = batch['attention_mask_l'].to(device)
            is_response_token_l = batch['is_response_token_l'].to(device).bool()
            
            #compute logps for chosen and rejected responses for both w and l
            logps_w = compute_logps(model, input_ids_w, attention_mask_w, is_response_token_w)
            logps_l = compute_logps(model, input_ids_l, attention_mask_l, is_response_token_l)

            #no grad for reference model
            with torch.no_grad():
                logps_ref_w = compute_logps(reference_model, input_ids_w, attention_mask_w, is_response_token_w)
                logps_ref_l = compute_logps(reference_model, input_ids_l, attention_mask_l, is_response_token_l)

            loss, reward_margin = compute_loss(logps_w, logps_l, logps_ref_w, logps_ref_l, beta)

            with torch.no_grad():
                loss_sum += loss.item()
                reward_margin_sum += reward_margin
                loss_count += 1

                epoch_loss_sum += loss.item()
                epoch_reward_margin_sum += reward_margin
                epoch_loss_count += 1

            # Accumulate gradients and scale by gradient_accumulation_steps
            (loss / gradient_accumulation_steps).backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # log loss per step
                wandb.log(
                    {
                        'ipo_train_loss_step': loss_sum/gradient_accumulation_steps,
                        'ipo_train_reward_margin_step': reward_margin_sum/gradient_accumulation_steps   ,
                        'lr': scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
                loss_sum = 0.0
                reward_margin_sum = 0.0
                loss_count = 0

                ########## EVALUATION ##########
                # evaluate on test dataloader every epoch or on the final epoch
                # can change to evaluate every x gradient accumulation steps by changing the % 1 to % x
                if (idx + 1) % 10 == 0:
                    evaluate(model, reference_model, test_dataloader, device, beta, global_step)
                    model.train()

        # NOT USING PER EPOCH LOGGING BECAUSE WE ARE LOGGING PER STEP ABOVE  

        # calculate average loss and accuracy per epoch
        epoch_avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        epoch_avg_reward_margin = epoch_reward_margin_sum / max(epoch_loss_count, 1)
        print('================')
        print(f'Epoch {e}/{num_epochs}.')
        print(f'Train loss {epoch_avg_loss:.4f}.')
        print(f'Train reward margin {epoch_avg_reward_margin:.4f}.')

        
        # # log to wandb, taken from HW2
        # wandb.log(
        #     {'ipo_train_loss_epoch': avg_loss, 
        #     'ipo_train_reward_margin_epoch': avg_reward_margin},
        #     step=e,
        # )

        # ########## EVALUATION ##########
        # # evaluate on test dataloader every epoch or on the final epoch
        # # can change to evaluate every x epochs by changing the % 1 to % x
        # if e % 1 == 0 or e == num_epochs - 1:
        #     evaluate(model, reference_model, test_dataloader, device, beta, e)

        clear_cache(model)

    # save model checkpoint after training
    if save_model == 1:
        save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)

    
            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='asingh15/countdown_tasks_3to4-dpo')
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
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--average_logps', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='dpo')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer, reference_model = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

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
        reference_model,
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
        args.beta,
        args.average_logps,
        args.loss_type
    )

if __name__ == "__main__":
    main()
