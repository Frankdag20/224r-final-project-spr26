#!/bin/bash
# Evaluate the TIR RLOO checkpoint.

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

MODEL_PATH="${MODEL_PATH:-/vol/checkpoints/tir_rloo_checkpoints/tir_rloo_project/tir_rloo_multi_turn/latest_checkpoint/model}"
OUTPUT_NAME="${OUTPUT_NAME:-tir_rloo_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/evaluation/eval_results}"

echo "=== Evaluating TIR RLOO checkpoint ==="

# With tools
modal run modal_train.py eval_tir -- \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "${OUTPUT_NAME}_with_tools" \
    --num_responses 16 \
    --max_tool_turns 5

# Without tools
modal run modal_train.py eval_tir -- \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "${OUTPUT_NAME}_no_tools" \
    --num_responses 16 \
    --no_tools

echo "=== Done. Download results with: ==="
echo "modal volume get default-proj-training evaluation/eval_results ./eval_results"
