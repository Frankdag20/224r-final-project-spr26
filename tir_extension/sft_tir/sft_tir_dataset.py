"""Dataset + collation for SFT over tool-using trajectories.

Mirrors ``sft_trainer/sft_dataset.SFTDataset`` but:

- Loads a local JSON file (the output of ``generate_trajectories.py``)
  with columns ``prompt`` and ``completion``.
- Builds an extra ``tool_result_mask`` per token so the SFT loss can be
  computed only on the policy's own tokens (i.e. excluding the
  deterministic ``<tool_result>...</tool_result>`` spans).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset

# Same span regex used by the tool pool, duplicated here to avoid pulling
# tool execution into the dataset code path.
_TOOL_RESULT_PATTERN = re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL)


def _load_records(json_path: str) -> list[dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Trajectory file not found: {json_path}")
    with open(json_path, "r") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {json_path}, got {type(records)}")
    return records


def _is_pre_formatted(prompt: str) -> bool:
    """Return True if the prompt is already a rendered chat template string."""
    return "<|im_start|>" in prompt or "<|im_end|>" in prompt or prompt.startswith("<s>")


def _apply_chat_template(tokenizer, prompt: str, completion: str) -> tuple[str, str]:
    """Format ``prompt``/``completion`` with the model's chat template.

    If the prompt is already a rendered chat-template string (e.g. from a
    dataset that stores Qwen2.5-formatted prompts), the template is not
    applied again to avoid double-wrapping.
    """
    if _is_pre_formatted(prompt):
        return prompt, completion

    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        add_generation_prompt=False,
        tokenize=False,
    )
    if prompt_only not in full:
        raise ValueError("Chat template did not preserve the prompt prefix")
    response_text = full[len(prompt_only):]
    return prompt_only, response_text


def _tool_result_token_mask(
    text: str,
    tokenizer,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Return ``(token_ids, mask)`` where ``mask[i] == 1`` for tool-result tokens.

    Uses offset-mapping from the fast tokenizer to align character spans to
    token spans without having to re-tokenize each span individually.
    """
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    mask = [0] * len(token_ids)
    spans = [(m.start(), m.end()) for m in _TOOL_RESULT_PATTERN.finditer(text)]
    if not spans:
        return token_ids, mask
    for tok_idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for span_start, span_end in spans:
            # Any overlap with a tool-result span -> exclude from loss.
            if start < span_end and end > span_start:
                mask[tok_idx] = 1
                break
    return token_ids, mask


class SFTToolTrajectoryDataset(TorchDataset):
    """Tokenises tool-using trajectories with a per-token tool-result mask."""

    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_prompt_length: int = 512,
        max_response_length: int = 1024,
        keep_only_full_score: bool = True,
        min_score: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

        raw = _load_records(json_path)
        cleaned: list[dict] = []
        for rec in raw:
            score = float(rec.get("score", 0.0))
            if keep_only_full_score and score < min_score:
                continue
            prompt_only, response_text = _apply_chat_template(
                tokenizer,
                rec["prompt"],
                rec["completion"],
            )
            cleaned.append(
                {
                    "prompt": prompt_only,
                    "response": response_text,
                    "score": score,
                }
            )
        if not cleaned:
            raise ValueError(
                f"No usable trajectories found in {json_path}"
                f" (keep_only_full_score={keep_only_full_score}, min_score={min_score})"
            )
        self.records = cleaned

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        prompt_token_lists: list[list[int]] = []
        response_token_lists: list[list[int]] = []
        response_tool_masks: list[list[int]] = []

        for item in batch:
            prompt_tokens = self.tokenizer(
                item["prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_prompt_length,
            )["input_ids"]

            response_tokens, response_tool_mask = _tool_result_token_mask(
                item["response"],
                self.tokenizer,
                max_length=self.max_response_length,
            )
            prompt_token_lists.append(prompt_tokens)
            response_token_lists.append(response_tokens)
            response_tool_masks.append(response_tool_mask)

        max_prompt_len = max(len(t) for t in prompt_token_lists)
        max_response_len = max(len(t) for t in response_token_lists)
        seq_len = max_prompt_len + max_response_len

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids = torch.full((len(batch), seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), seq_len), dtype=torch.long)
        is_response_token = torch.zeros((len(batch), seq_len), dtype=torch.long)
        tool_result_mask = torch.zeros((len(batch), seq_len), dtype=torch.long)

        for i, (p, r, m) in enumerate(
            zip(prompt_token_lists, response_token_lists, response_tool_masks)
        ):
            # Left-pad the prompt portion so the boundary between prompt and
            # response is aligned across the batch.
            prompt_offset = max_prompt_len - len(p)
            input_ids[i, prompt_offset : prompt_offset + len(p)] = torch.tensor(p)
            attention_mask[i, prompt_offset : prompt_offset + len(p)] = 1

            response_start = max_prompt_len
            response_end = response_start + len(r)
            input_ids[i, response_start:response_end] = torch.tensor(r)
            attention_mask[i, response_start:response_end] = 1
            is_response_token[i, response_start:response_end] = 1
            tool_result_mask[i, response_start:response_end] = torch.tensor(m)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "is_response_token": is_response_token,
            "tool_result_mask": tool_result_mask,
        }


def get_dataloaders(
    json_path: str,
    tokenizer,
    batch_size: int = 8,
    max_prompt_length: int = 512,
    max_response_length: int = 1024,
    num_workers: int = 0,
    keep_only_full_score: bool = True,
    val_fraction: float = 0.1,
    seed: int = 0,
    gradient_accumulation_steps: int = 1,
) -> dict[str, DataLoader]:
    """Split the trajectory file into train/test DataLoaders."""
    assert batch_size % gradient_accumulation_steps == 0
    micro_bs = batch_size // gradient_accumulation_steps

    full = SFTToolTrajectoryDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        keep_only_full_score=keep_only_full_score,
    )

    if val_fraction <= 0 or len(full) < 2:
        return {
            "train": DataLoader(
                full,
                batch_size=micro_bs,
                shuffle=True,
                collate_fn=full.collate_fn,
                num_workers=num_workers,
                drop_last=False,
            )
        }

    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full, [n_train, n_val], generator=generator
    )
    return {
        "train": DataLoader(
            train_subset,
            batch_size=micro_bs,
            shuffle=True,
            collate_fn=full.collate_fn,
            num_workers=num_workers,
            drop_last=False,
        ),
        "test": DataLoader(
            val_subset,
            batch_size=micro_bs,
            shuffle=False,
            collate_fn=full.collate_fn,
            num_workers=num_workers,
            drop_last=False,
        ),
    }
