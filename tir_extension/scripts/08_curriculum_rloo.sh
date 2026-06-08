#!/bin/bash
# Stage 8: Curriculum TIR RLOO training — 3 parallel runs.
#
# Launches 3 experiments on separate Modal apps (on-demand H100):
#   A) numcount curriculum: 3-number=medium, 4-number=hard
#   B) solvability curriculum: teacher-labeled tool solvability
#   C) no curriculum baseline: uniform sampling from all problems
#
# All use hierarchical reward and the sbfisher/countdown_curriculum dataset.
#
# Usage:
#   source .env && bash tir_extension/scripts/08_curriculum_rloo.sh
#
# To run a single experiment:
#   source .env && bash tir_extension/scripts/08_curriculum_rloo.sh numcount
#   source .env && bash tir_extension/scripts/08_curriculum_rloo.sh solvability
#   source .env && bash tir_extension/scripts/08_curriculum_rloo.sh none

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-sbfisher/tir-sft-3tool_from_sft}"
DATASET_NAME="${DATASET_NAME:-sbfisher/countdown_curriculum}"
NUM_STEPS="${NUM_STEPS:-150}"
CURRICULUM_END="${CURRICULUM_END:-110}"

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
        --batch_size 4 \
        --group_size 2 \
        --gradient_accumulation_steps 4 \
        --save_every_n_steps 25

    echo "Submitted $wandb_name to Modal (detached)."
    echo ""
}

if [[ $# -ge 1 ]]; then
    # Run a single experiment
    case "$1" in
        numcount)
            run_experiment numcount curriculum_numcount curriculum-numcount
            ;;
        solvability)
            run_experiment solvability curriculum_solvability curriculum-solvability
            ;;
        none)
            run_experiment none no_curriculum_baseline no-curriculum-baseline
            ;;
        *)
            echo "Unknown strategy: $1"
            echo "Usage: $0 [numcount|solvability|none]"
            exit 1
            ;;
    esac
else
    # Run all 3 experiments
    run_experiment numcount curriculum_numcount curriculum-numcount
    run_experiment solvability curriculum_solvability curriculum-solvability
    run_experiment none no_curriculum_baseline no-curriculum-baseline
fi

echo "All experiments submitted. Monitor at https://modal.com"
