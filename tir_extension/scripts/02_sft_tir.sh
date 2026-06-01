#!/bin/bash
# Stage 2: SFT warm-start on tool-using trajectories.
#
# Prerequisites:
#   - Stage 1 completed (tir_trajectories.json on Modal volume)
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...
#
# Trains Qwen2.5-0.5B on the generated trajectories with tool-result
# loss masking. Output checkpoint goes to /vol/checkpoints/tir_sft_checkpoints/

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-/vol/checkpoints/tir_trajectories.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/checkpoints/tir_sft_checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-6}"
WANDB_PROJECT="${WANDB_PROJECT:-tir_sft_project}"
WANDB_NAME="${WANDB_NAME:-tir_sft_lr${LR}_ep${EPOCHS}}"

echo "=== Stage 2: TIR SFT warm-start ==="
echo "Model: ${MODEL_NAME}"
echo "Trajectories: ${TRAJECTORY_PATH}"

modal run modal_train.py tir_sft -- \
    --model_name "$MODEL_NAME" \
    --trajectory_path "$TRAJECTORY_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --num_epochs "$EPOCHS" \
    --warmup_ratio 0.05 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME"

echo "=== Done. Checkpoint saved to ${OUTPUT_DIR} on Modal volume. ==="
echo "Verify with: modal volume ls default-proj-training /checkpoints/tir_sft_checkpoints/"
