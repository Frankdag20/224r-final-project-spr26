#!/bin/bash
# Stage 8: Curriculum TIR RLOO training (medium → hard).
#
# Trains on mih123/RL_calc_training_set with tool-integrated prompts.
# Starts 100% medium problems, linearly shifts to 100% hard by step 110,
# then trains on hard only for the remaining steps (110-150).
#
# Usage:
#   source .env && bash tir_extension/scripts/08_curriculum_rloo.sh
#
# Uses vanilla reward (compute_score) by default since hierarchical
# reward didn't help in prior experiments.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-sbfisher/tir-sft-3tool_from_sft}"
DATASET_NAME="${DATASET_NAME:-mih123/RL_calc_training_set}"
WANDB_NAME="${WANDB_NAME:-curriculum_medium_to_hard_v1}"
NUM_STEPS="${NUM_STEPS:-150}"
CURRICULUM_END="${CURRICULUM_END:-110}"

echo "=== Curriculum TIR RLOO ==="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Steps: $NUM_STEPS (curriculum ends at step $CURRICULUM_END)"
echo ""

modal run --detach modal_train.py tir_curriculum -- \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --wandb_name "$WANDB_NAME" \
    --num_training_steps "$NUM_STEPS" \
    --curriculum_end_step "$CURRICULUM_END" \
    --use_vanilla_reward \
    --batch_size 4 \
    --group_size 2 \
    --gradient_accumulation_steps 4 \
    --save_every_n_steps 25

echo ""
echo "Submitted to Modal (detached)."
echo "Monitor at https://modal.com"
