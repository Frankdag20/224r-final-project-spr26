#!/bin/bash
# Stage 3: RLOO with multi-turn tool-integrated reasoning.
#
# Prerequisites:
#   - Stage 2 completed (TIR SFT checkpoint on Modal volume)
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

# Point this at the TIR SFT checkpoint
MODEL_NAME="${MODEL_NAME:-/vol/checkpoints/tir_sft_checkpoints/tir_sft_project/tir_sft_lr5e-5_ep6/model}"
DATASET_NAME="${DATASET_NAME:-asingh15/countdown_tasks_3to4}"
SAVE_DIR="${SAVE_DIR:-/vol/checkpoints/tir_rloo_checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-tir_rloo_project}"
WANDB_NAME="${WANDB_NAME:-tir_rloo_multi_turn}"

NUM_STEPS="${NUM_STEPS:-250}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
LR="${LR:-1e-5}"
ENTROPY="${ENTROPY:-0.01}"
KL="${KL:-0.0}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-5}"

echo "=== Stage 3: TIR RLOO with multi-turn tool execution ==="
echo "Model: ${MODEL_NAME}"
echo "Steps: ${NUM_STEPS}, Group: ${GROUP_SIZE}, LR: ${LR}"

modal run modal_train.py tir_rloo -- \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --save_dir "$SAVE_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME" \
    --num_training_steps "$NUM_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --group_size "$GROUP_SIZE" \
    --learning_rate "$LR" \
    --warmup_ratio 0.0 \
    --entropy_coefficient "$ENTROPY" \
    --kl_divergence_coefficient "$KL" \
    --max_tool_turns "$MAX_TOOL_TURNS" \
    --multi_turn_tools \
    --reanalyze_every_n_steps 0 \
    --save_every_n_steps 50 \
    --enable_chunked_prefill

echo "=== Done. Checkpoints at ${SAVE_DIR}/ ==="
