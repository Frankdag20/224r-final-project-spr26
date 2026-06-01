#!/bin/bash
# TIR SFT Experiments — run both ablations on Modal.
#
# Prerequisites:
#   - tir_trajectories_calc_only.json uploaded to Modal volume
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...
#
# Usage:
#   bash tir_extension/scripts/02_sft_tir_experiments.sh [exp1|exp2|both]
#     exp1 = from vanilla SFT checkpoint (comparable to Mahmood's run)
#     exp2 = from base Qwen (cold start, matches literature)
#     both = run both sequentially (default)

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Shared config
TRAJECTORY_PATH="${TRAJECTORY_PATH:-/vol/checkpoints/tir_trajectories_calc_only.json}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-/vol/checkpoints/tir_sft_checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-10}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
WANDB_PROJECT="${WANDB_PROJECT:-tir_sft_project}"

EXPERIMENT="${1:-both}"

run_exp1() {
    echo "=== Experiment 1: TIR SFT from vanilla SFT checkpoint ==="
    echo "  Model: asingh15/qwen-sft-countdown-defaultproj"
    echo "  Trajectories: ${TRAJECTORY_PATH}"
    echo "  LR: ${LR}, Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}x${GRAD_ACCUM}"

    modal run "$PROJECT_ROOT/modal_train.py" tir_sft -- \
        --model_name "asingh15/qwen-sft-countdown-defaultproj" \
        --trajectory_path "$TRAJECTORY_PATH" \
        --output_dir "${OUTPUT_DIR_BASE}" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" \
        --num_epochs "$EPOCHS" \
        --max_response_length "$MAX_RESPONSE_LENGTH" \
        --warmup_ratio 0.05 \
        --gradient_checkpointing 1 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "calc_only_from_sft_lr${LR}_ep${EPOCHS}"

    echo "=== Experiment 1 done ==="
}

run_exp2() {
    echo "=== Experiment 2: TIR SFT from base Qwen (cold start) ==="
    echo "  Model: Qwen/Qwen2.5-0.5B"
    echo "  Trajectories: ${TRAJECTORY_PATH}"
    echo "  LR: ${LR}, Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}x${GRAD_ACCUM}"

    modal run "$PROJECT_ROOT/modal_train.py" tir_sft -- \
        --model_name "Qwen/Qwen2.5-0.5B" \
        --trajectory_path "$TRAJECTORY_PATH" \
        --output_dir "${OUTPUT_DIR_BASE}" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" \
        --num_epochs "$EPOCHS" \
        --max_response_length "$MAX_RESPONSE_LENGTH" \
        --warmup_ratio 0.05 \
        --gradient_checkpointing 1 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "calc_only_from_base_lr${LR}_ep${EPOCHS}"

    echo "=== Experiment 2 done ==="
}

case "$EXPERIMENT" in
    exp1) run_exp1 ;;
    exp2) run_exp2 ;;
    both) run_exp1; run_exp2 ;;
    *) echo "Usage: $0 [exp1|exp2|both]"; exit 1 ;;
esac
