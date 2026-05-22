"""Ray actor that applies policy-gradient updates for RLOO.

The orchestrator (`rloo.py`) samples responses and computes rewards, then
calls this worker with tokenized sequences to perform gradient updates.

This file is intentionally incomplete. Students are expected to implement
`update(...)` while reusing the data/model/sampling setup provided here.
"""

import atexit
import os
import warnings
import ray
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")

@ray.remote(num_gpus=1)
class RLOOUpdateWorker:
    """Owns policy/ref models and optimizer state for RLOO updates."""
    def __init__(
        self, 
        model_path, 
        optimizer_path, 
        scheduler_path,
        tokenizer_path=None, 
        ref_model_path=None,
        batch_size=64,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        group_size=16, 
        entropy_coefficient=0.01, 
        kl_divergence_coefficient=0.0, 
        lr_schedule='constant',
        learning_rate=1e-5, 
        weight_decay=0.01, 
        warmup_ratio=0.0,
        num_training_steps=250,
    ):
        self.model_path = model_path
        self.ref_model_path = ref_model_path if ref_model_path is not None else model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.group_size = group_size
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2 for RLOO, got {self.group_size}")
        self.entropy_coefficient = entropy_coefficient
        self.kl_divergence_coefficient = kl_divergence_coefficient
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        if warmup_ratio > 0:
            raise NotImplementedError("Warmup ratio > 0 is not supported for constant learning rate schedule")
        self.num_training_steps = num_training_steps

    def tear_down(self):
        """Release model/optimizer objects and clear GPU memory."""
        import gc
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'ref_model'):
            del self.ref_model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'scheduler'):
            del self.scheduler
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_checkpoint_paths(self, model_path, optimizer_path, scheduler_path, load_checkpoint=False):
        """Update output paths (and optionally reload state immediately)."""
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        if load_checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load policy model, optional reference model, and optimizer/scheduler."""
        self.tear_down()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device="cuda")
        self.model.gradient_checkpointing_enable()

        if self.kl_divergence_coefficient > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.ref_model_path,
                torch_dtype=torch.bfloat16,
            ).to(device="cuda")
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        if self.optimizer_path and self.scheduler_path and os.path.exists(self.optimizer_path) and os.path.exists(self.scheduler_path):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")
            
            self.scheduler.load_state_dict(torch.load(self.scheduler_path))
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")

        self.model.train()

    def save_checkpoint(self):
        """Persist optimizer/scheduler state plus model+tokenizer weights."""
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
        torch.save(self.scheduler.state_dict(), self.scheduler_path)

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)


    def update_gradient_accumulation(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        device='cuda',
    ):
        """Split incoming batch into microbatches and call `update(...)`."""
        update_metrics = None
        if self.gradient_accumulation_steps > 1:
            curr_batch_size = input_ids.shape[0]
            assert curr_batch_size % self.gradient_accumulation_steps == 0, (
                f"Flattened batch size {curr_batch_size} must be divisible by gradient_accumulation_steps "
                f"{self.gradient_accumulation_steps}."
            )
            group_per_gradient_accumulation_step = curr_batch_size // self.gradient_accumulation_steps
            # Ensure each microbatch still contains full RLOO groups so the baseline is meaningful
            assert group_per_gradient_accumulation_step % self.group_size == 0, (
                f"Microbatch size {group_per_gradient_accumulation_step} must be divisible by group_size {self.group_size} "
                f"when using gradient_accumulation_steps={self.gradient_accumulation_steps}."
            )
            all_metrics = []
            for i in range(self.gradient_accumulation_steps):
                curr_input_ids = input_ids[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_attention_mask = attention_mask[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_is_response_token = is_response_token[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_rewards = rewards[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_sample_log_probs = None
                if sample_log_probs is not None:
                    curr_sample_log_probs = sample_log_probs[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                
                is_update_step = (i == self.gradient_accumulation_steps - 1)
                curr_update_metrics = self.update(
                    curr_input_ids,
                    curr_attention_mask,
                    curr_is_response_token,
                    curr_rewards,
                    curr_sample_log_probs,
                    is_update_step,
                    device,
                )
                all_metrics.append(curr_update_metrics)
            update_metrics = {}
            for metric_name in all_metrics[0].keys():
                update_metrics[metric_name] = np.mean([metric[metric_name] for metric in all_metrics]).item()
        else:
            update_metrics = self.update(
                input_ids,
                attention_mask,
                is_response_token,
                rewards,
                sample_log_probs,
                True,
                device,
            )

        return update_metrics

    # `is_update_step` is False on intermediate microbatches so we can
    # accumulate gradients before stepping optimizer/scheduler.
    def update(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        is_update_step: bool = True,
        device='cuda',
    ):
        # TODO(student): implement one RLOO policy update.
        # Inputs arrive flattened as [batch_size * group_size, seq_len].
        # convert inputs to tensors
        input_ids = torch.from_numpy(input_ids).to(device)
        attention_mask = torch.from_numpy(attention_mask).to(device)
        is_response_token = torch.from_numpy(is_response_token).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        sample_log_probs = torch.from_numpy(sample_log_probs).to(device) if sample_log_probs is not None else None
        # Required pieces:
        # 1) Compute per-token log-probs on target tokens under current policy.
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use classic causal LM trick, use logits shifted without the last logit
        # and targets as input_ids shifted w/o first id
        logits = out.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        response_mask = is_response_token[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        seq_log_probs = (target_log_probs * response_mask).sum(dim=-1)
        # 2) Build leave-one-out baseline within each response group.
        k = self.group_size
        N = rewards.shape[0]
        num_prompts = N // k

        grp_rewards = rewards.view(num_prompts, k)
        grp_mean_reward = grp_rewards.mean(dim=1, keepdim=True)
        A = (k / (k - 1)) * (grp_rewards - grp_mean_reward)
        A = A.view(N).detach()
        # 3) Compute policy-gradient loss using advantages (and importance weights
        #    if sample_log_probs are provided).
        if sample_log_probs is not None:
            with torch.no_grad():
                # using clamp values suggested on Ed
                log_prob_diff = torch.clamp(seq_log_probs - sample_log_probs, min=-20.0, max=20.0)
                iw = torch.clamp(torch.exp(log_prob_diff), max=100.0)
        else:
            iw = torch.ones_like(seq_log_probs)
        # 4) Add entropy regularization and optional KL penalty to ref model.
        # calculate shannon entropy (I think that's what entropy refers to)
        token_count = response_mask.sum()
        ps = log_probs.exp()
        entropy_per_token = -(ps * log_probs).sum(dim=-1)
        entropy = (entropy_per_token * response_mask).sum() / token_count

        # calculate kl divergence term
        if self.kl_divergence_coefficient > 0:
            with torch.no_grad():
                ref_model_out = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_model_logits = ref_model_out.logits[:, :-1, :]
                # gather the log ps for the selected tokens
                # unsqueeze target_ids so it matches the shape (N, T-1, V) with (N, T-1, 1)
                ref_log_ps = F.log_softmax(ref_model_logits, dim=-1).gather(
                    dim = -1,
                    index = target_ids.unsqueeze(-1)
                ).squeeze(-1)
            
            log_diff = ref_log_ps - target_log_probs
            kl_term = ((torch.exp(log_diff) - 1.0 - log_diff) * response_mask).sum() / token_count
        else:
            kl_term = torch.tensor(0.0, device=device)

        # calculate final loss as sum of policy gradient loss, entropy regularization, and kl divergence term
        policy_gradient_loss = -(iw * A * seq_log_probs).mean()
        entropy_loss = -self.entropy_coefficient * entropy
        kl_divergence_penalty = self.kl_divergence_coefficient * kl_term
        loss = policy_gradient_loss + entropy_loss + kl_divergence_penalty
        # 5) Backward pass; if `is_update_step`, clip and step optimizer/scheduler.
        # make sure average loss over gradient accumulation steps is used
        (loss / self.gradient_accumulation_steps).backward()

        if is_update_step:
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm = self.gradient_clipping,
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        # 6) Return scalar metrics used by trainer logging.
        metrics = {
            'policy_gradient_loss': policy_gradient_loss.item(),
            'entropy': entropy_loss.item(),
            'kl_loss': kl_divergence_penalty.item(),
            'loss': loss.item(),
            'iw_mean': iw.mean().item(),
        }
        return metrics
