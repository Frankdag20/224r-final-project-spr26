"""Ray actor that applies policy-gradient updates for RLOO.

The orchestrator (`rloo.py`) samples responses and computes rewards, then
calls this worker with tokenized sequences to perform gradient updates.

This file is intentionally incomplete. Students are expected to implement
`update(...)` while reusing the data/model/sampling setup provided here.
"""

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
        entropy_coefficient=0.05, 
        kl_divergence_coefficient=0.02, 
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
        tool_result_mask: Optional[np.ndarray] = None,
        device='cuda',
        group_size_override: Optional[int] = None,
    ):
        """Split incoming batch into microbatches and call `update(...)`.

        ``tool_result_mask`` (optional, same shape as ``input_ids``) marks
        tokens that came from a deterministic tool execution
        (``<tool_result>...</tool_result>``). Those tokens are excluded from
        the policy-gradient loss and from the entropy/KL bonuses inside
        ``update`` so we don't train the policy on its own tool outputs.
        """
        effective_group = group_size_override if group_size_override is not None else self.group_size
        update_metrics = None
        if self.gradient_accumulation_steps > 1:
            curr_batch_size = input_ids.shape[0]
            assert curr_batch_size % self.gradient_accumulation_steps == 0, (
                f"Flattened batch size {curr_batch_size} must be divisible by gradient_accumulation_steps "
                f"{self.gradient_accumulation_steps}."
            )
            group_per_gradient_accumulation_step = curr_batch_size // self.gradient_accumulation_steps
            assert group_per_gradient_accumulation_step % effective_group == 0, (
                f"Microbatch size {group_per_gradient_accumulation_step} must be divisible by group_size {effective_group} "
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
                curr_tool_result_mask = None
                if tool_result_mask is not None:
                    curr_tool_result_mask = tool_result_mask[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]

                is_update_step = (i == self.gradient_accumulation_steps - 1)
                curr_update_metrics = self.update(
                    curr_input_ids,
                    curr_attention_mask,
                    curr_is_response_token,
                    curr_rewards,
                    curr_sample_log_probs,
                    is_update_step,
                    device,
                    tool_result_mask=curr_tool_result_mask,
                    group_size_override=effective_group,
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
                tool_result_mask=tool_result_mask,
                group_size_override=effective_group,
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
        tool_result_mask: Optional[np.ndarray] = None,
        group_size_override: Optional[int] = None,
    ):
        # TODO(student): implement one RLOO policy update.
        # Inputs arrive flattened as [batch_size * group_size, seq_len].
        # Required pieces:
        # 1) Compute per-token log-probs on target tokens under current policy.
        # 2) Build leave-one-out baseline within each response group.
        # 3) Compute policy-gradient loss using advantages (and importance weights
        #    if sample_log_probs are provided).
        # 4) Add entropy regularization and optional KL penalty to ref model.
        # 5) Backward pass; if `is_update_step`, clip and step optimizer/scheduler.
        # 6) Return scalar metrics used by trainer logging.
        
        # Convert everything to tensors since they are all numpy arrays
        input_ids_torch = torch.from_numpy(input_ids).to(device)
        attention_mask_torch = torch.from_numpy(attention_mask).to(device)
        response_mask = torch.from_numpy(is_response_token).to(device).float() * attention_mask_torch.float()
        rewards_torch = torch.from_numpy(rewards).to(device).float()

        # TIR extension: exclude deterministic <tool_result>...</tool_result>
        # tokens from the policy gradient so we never train on tokens the
        # model didn't actually generate.
        tool_result_fraction = 0.0
        if tool_result_mask is not None:
            tool_result_mask_torch = torch.from_numpy(tool_result_mask).to(device).float()
            # Per-token "trainable" mask: response AND not tool-result.
            response_mask = response_mask * (1.0 - tool_result_mask_torch)
            tool_result_fraction = float(
                tool_result_mask_torch.sum().item()
                / max(tool_result_mask_torch.numel(), 1)
            )

        # Compute token log probs
        outputs = self.model(input_ids=input_ids_torch, attention_mask=attention_mask_torch)

        # Same logic as done in sft.py
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = input_ids_torch[:, 1:]
        shifted_mask = response_mask[:, 1:]

        #                                [batch*group, vocab, seq len]  [batch*group, seq len] => [batch*group, seq len]
        token_log_probs = -F.cross_entropy(shifted_logits.transpose(1,2), shifted_labels, reduction='none')

        # Build leave-one-out baseline
        group = group_size_override if group_size_override is not None else self.group_size
        batch_size = rewards_torch.shape[0] // group # [batch*group]

        rewards_g = rewards_torch.view(batch_size, group)
        group_sum = rewards_g.sum(dim=1, keepdim=True)
        baseline = (group_sum - rewards_g) / (group - 1)

        # Compute advantage
        adv = (rewards_g - baseline).view(-1)

        log_probs_filt = (token_log_probs * shifted_mask).sum(dim=-1)

        # Get importance weights and compute policy gradient loss
        if sample_log_probs is not None:
            with torch.no_grad():
                sample_log_probs_torch = torch.from_numpy(sample_log_probs).to(device)
                ratio = torch.exp(torch.clamp(log_probs_filt - sample_log_probs_torch, -20, 20))
                weights = torch.clamp(ratio, max=100)

            policy_loss = -(weights * log_probs_filt * adv.detach()).mean()
        else:
            weights = torch.ones_like(log_probs_filt)
            policy_loss = -(log_probs_filt * adv.detach()).mean()
        
        # Add entropy regularization
        token_count = shifted_mask.sum().clamp(min=1.0)
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy_token_lvl = -(probs * log_probs).sum(dim=-1)
        entropy = (entropy_token_lvl * shifted_mask).sum() / token_count

        # Add KL penalty with respect to reference model
        if self.kl_divergence_coefficient > 0:
            with torch.no_grad():
                ref_out = self.ref_model(input_ids=input_ids_torch, attention_mask=attention_mask_torch)

                ref_logits = ref_out.logits[:, :-1, :]

                ref_token_log_probs = -F.cross_entropy(ref_logits.transpose(1,2), shifted_labels, reduction='none')

            log_diff = ref_token_log_probs - token_log_probs
            kl = ((torch.exp(log_diff) - 1.0 - log_diff) * shifted_mask).sum() / token_count
        else:
            kl = torch.tensor(0.0, device=device)

        # Combine losses
        ent_loss = -self.entropy_coefficient * entropy
        kl_loss = self.kl_divergence_coefficient * kl

        loss = policy_loss + ent_loss + kl_loss

        # Backward pass & avg loss over gradient accumulation steps
        (loss / self.gradient_accumulation_steps).backward()

        # Take steps if it's an update step
        if is_update_step:
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping)
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Return scalar metrics
        metrics = {
            'policy_gradient_loss': policy_loss.item(),
            'entropy_loss': ent_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': loss.item(),
            'weight_mean': weights.mean().item(),
            'tool_result_mask_fraction': tool_result_fraction,
        }

        return metrics