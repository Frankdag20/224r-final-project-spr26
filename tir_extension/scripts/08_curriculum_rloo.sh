#!/bin/bash
# Stage 8: Curriculum TIR RLOO training — 5 parallel runs with DPO self-critic.
#
# Launches 5 experiments on separate Modal apps (on-demand H100):
#   A) numcount: 3-number=medium → 4-number=hard curriculum
#   B) solvability: teacher-labeled tool solvability curriculum
#   C) score: teacher difficulty score threshold ramp (3→10)
#   D) none: no curriculum baseline (uniform sampling)
#   E) hard_only: Tool-Star style — RL on hard (unsolved) problems only
#
# All use hierarchical reward + DPO self-critic (separate Adam, lr=5e-7, beta=0.3).
# Hyperparams match Run 13: batch=128, group=8, grad_accum=128, lr=1e-5,
# KL=0.01, weight_decay=1e-4, entropy=0.01.
# Dataset: sbfisher/countdown_curriculum (490k problems).
#
# Usage:
#   source .env && bash tir_extension/scripts/08_curriculum_rloo.sh
#
# To run a single experiment:
#   source .env && bash tir_extension/scripts/08_curriculum_rloo.sh numcount

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-sbfisher/tir-sft-3tool_from_sft}"
DATASET_NAME="${DATASET_NAME:-sbfisher/countdown_curriculum}"
NUM_STEPS="${NUM_STEPS:-150}"
CURRICULUM_END="${CURRICULUM_END:-60}"

run_experiment() {
    local strategy="$1"
    local wandb_name="$2"
    local app_name="$3"

    echo "=== Launching: $wandb_name (strategy=$strategy) ==="
    echo "Model: $MODEL_NAME"
    echo "Dataset: $DATASET_NAME"
    echo "Steps: $NUM_STEPS (curriculum ends at step $CURRICULUM_END)"
    echo ""

    MODAL_APP_NAME="$app_name" \
    MODAL_GPU="H100" \
    modal run --detach modal_train.py tir_curriculum -- \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --wandb_name "$wandb_name" \
        --curriculum_strategy "$strategy" \
        --num_training_steps "$NUM_STEPS" \
        --curriculum_end_step "$CURRICULUM_END" \
        --self_critic \
        --self_critic_every_k 5 \
        --self_critic_n_samples 8 \
        --self_critic_beta 0.3 \
        --batch_size 128 \
        --group_size 8 \
        --gradient_accumulation_steps 128 \
        --learning_rate 1e-5 \
        --kl_divergence_coefficient 0.01 \
        --weight_decay 1e-4 \
        --entropy_coefficient 0.01 \
        --save_every_n_steps 25

    echo "Submitted $wandb_name to Modal (detached)."
    echo ""
}

if [[ $# -ge 1 ]]; then
    case "$1" in
        numcount)
            run_experiment numcount curriculum_numcount curriculum-numcount
            ;;
        solvability)
            run_experiment solvability curriculum_solvability curriculum-solvability
            ;;
        score)
            run_experiment score curriculum_score curriculum-score
            ;;
        none)
            run_experiment none no_curriculum_baseline no-curriculum-baseline
            ;;
        hard_only)
            run_experiment hard_only curriculum_hard_only curriculum-hard-only
            ;;
        *)
            echo "Unknown strategy: $1"
            echo "Usage: $0 [numcount|solvability|score|none|hard_only]"
            exit 1
            ;;
    esac
else
    run_experiment numcount curriculum_numcount curriculum-numcount
    run_experiment solvability curriculum_solvability curriculum-solvability
    run_experiment score curriculum_score curriculum-score
    run_experiment none no_curriculum_baseline no-curriculum-baseline
    run_experiment hard_only curriculum_hard_only curriculum-hard-only
fi

echo "All experiments submitted. Monitor at https://modal.com"
