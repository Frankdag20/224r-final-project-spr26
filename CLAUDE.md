# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS224R final project: post-training pipeline for LLMs on the **Countdown arithmetic reasoning task**. The pipeline has three stages: SFT -> IPO -> RLOO, plus evaluation. Base model is Qwen/Qwen2.5-0.5B. All training uses bfloat16 and requires a CUDA GPU.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

Required env vars: `WANDB_API_KEY`, `HF_TOKEN`. Optional: `WANDB_ENTITY`, `CUDA_VISIBLE_DEVICES`.

## Common Commands

```bash
# Training (local)
bash sft_trainer/train_sft.sh
bash ipo_trainer/train_ipo.sh
bash rloo_trainer/train_rloo.sh    # needs MODEL_NAME and DATASET_NAME env vars

# Training (Modal - remote GPU)
bash sft_trainer/train_sft_modal.sh
bash ipo_trainer/train_ipo_modal.sh
bash rloo_trainer/train_rloo_modal.sh

# Evaluation
python evaluation/countdown_eval.py --model_path <path> --eval_dataset asingh15/countdown_tasks_3to4 --output_dir evaluation/eval_results --output_name <name>
bash evaluation/sample_modal.sh    # multi-model eval on Modal

# Direct Python (example)
python sft_trainer/sft.py --model_name Qwen/Qwen2.5-0.5B --dataset_name Asap7772/cog_behav_all_strategies --output_dir checkpoints/sft --batch_size 64 --gradient_accumulation_steps 8
```

## Linting

```bash
ruff check .          # line-length=100, ignores E501
ruff check --fix .
```

## Architecture

### Training Stages

1. **SFT** (`sft_trainer/sft.py`): Supervised fine-tuning with masked LM loss on response tokens only. Uses `is_response_token` mask to ignore prompt tokens (labels set to -100). Dataset requires `query` + `completion` columns.

2. **IPO** (`ipo_trainer/ipo.py`): Pairwise preference optimization. Maintains a frozen reference model (deep copy). Computes log-probs for chosen (`_w`) and rejected (`_l`) responses, then applies IPO loss: `((h - 1/(2*beta))^2).mean()` where `h = (logp_w - logp_ref_w) - (logp_l - logp_ref_l)`. Dataset requires `query` + `response_ws` + `response_ls` columns.

3. **RLOO** (`rloo_trainer/rloo.py` + `rloo_update_worker.py` + `sampling_worker.py`): Online RL with leave-one-out baseline. Uses Ray actors to alternate between vLLM sampling and PyTorch gradient updates on a single GPU (workers are created/killed as needed since they can't coexist). The orchestrator samples `group_size` responses per prompt, scores them via `compute_score`, tokenizes, then sends to the update worker. Dataset requires `prompt` + `ground_truth` (dict with `target` and `numbers`).

### RLOO Worker Architecture

- **SamplingWorker** (Ray actor, 1 GPU): Loads model into vLLM for fast batched generation. Returns responses + per-sequence log-probs.
- **RLOOUpdateWorker** (Ray actor, 1 GPU): Loads policy + optional reference model in PyTorch. Computes REINFORCE loss with leave-one-out baseline, entropy bonus, and optional KL penalty to reference. Supports gradient accumulation via microbatching (each microbatch must contain full groups).
- Only one worker type runs at a time; `_create_sampling_worker` kills the update worker and vice versa.

### Evaluation & Scoring (`evaluation/countdown.py`)

Rewards: 0.0 (no `<answer>` tags) -> 0.1 (valid format, wrong result) -> 1.0 (correct). Validates that the equation uses exactly the provided numbers.

### Modal Integration (`modal_train.py`)

Wraps all trainers for remote execution on Modal (default H100 GPU, 24h timeout). Artifacts persist in Modal volume `default-proj-training` under `/vol/`. Configure via `MODAL_GPU`, `MODAL_TIMEOUT_SECONDS`, `MODAL_VOLUME_NAME` env vars.

## Key Patterns

- All trainer entrypoints add the project root to `sys.path` so they can be run as `python sft_trainer/sft.py` from the project root.
- Checkpoints save to `<output_dir>/model/` (HF format) + `train_states.pth` (optimizer/scheduler).
- RLOO checkpoints: `<save_dir>/model/` + `optimizer.pt` + `scheduler.pt`.
- W&B logging uses `step=global_step` where global_step counts optimizer updates (not batch iterations).
- `gradient_accumulation_steps` affects both the loss scaling (`loss / grad_accum_steps`) and the scheduler step count calculation.
- RLOO's `lr_schedule='constant'` requires `warmup_ratio=0.0`.
