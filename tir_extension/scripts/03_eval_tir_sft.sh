#!/bin/bash
# Evaluate TIR SFT checkpoints — runs both with-tools and no-tools evals.
#
# Usage:
#   source .env && bash tir_extension/scripts/03_eval_tir_sft.sh                    # eval all runs
#   source .env && bash tir_extension/scripts/03_eval_tir_sft.sh 3tool_from_sft     # eval one specific run
#
# Prerequisites:
#   source .env  (needs HF_TOKEN, WANDB_API_KEY)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT_BASE="${CHECKPOINT_BASE:-/vol/checkpoints/tir_sft_checkpoints/tir_sft_project}"
EVAL_DATASET="${EVAL_DATASET:-asingh15/countdown_tasks_3to4}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/evaluation/eval_results}"

# All run names to evaluate (override with first arg for single run)
ALL_RUNS=(
    "3tool_from_sft"
    "3tool_from_base"
    "calc_only_from_sft"
    "calc_only_from_base"
)

if [[ $# -ge 1 ]]; then
    RUNS=("$1")
else
    RUNS=("${ALL_RUNS[@]}")
fi

for run in "${RUNS[@]}"; do
    model_path="${CHECKPOINT_BASE}/${run}/model"

    echo "=== Evaluating ${run} WITH tools ==="
    MODAL_APP_NAME="eval-${run}-tools" \
    modal run --detach "$PROJECT_ROOT/modal_train.py" eval_tir -- \
        --model_path "$model_path" \
        --eval_dataset "$EVAL_DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --output_name "${run}_with_tools"

    echo "=== Evaluating ${run} WITHOUT tools ==="
    MODAL_APP_NAME="eval-${run}-no-tools" \
    modal run --detach "$PROJECT_ROOT/modal_train.py" eval_tir -- \
        --model_path "$model_path" \
        --eval_dataset "$EVAL_DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --output_name "${run}_no_tools" \
        --no_tools
done

echo ""
echo "All evals launched. Monitor at https://modal.com"
echo "Once complete, download results with:"
echo "  modal volume get default-proj-training evaluation/eval_results/ ./eval_results/"
