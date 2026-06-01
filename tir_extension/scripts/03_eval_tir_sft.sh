#!/bin/bash
# Evaluate the TIR SFT checkpoint with multi-turn tool execution.
#
# Run this after Stage 2 to verify the SFT model can use tools
# and solve Countdown problems (target: comparable to baseline ~31.5% pass@1).

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

# Point this at the SFT checkpoint model directory
MODEL_PATH="${MODEL_PATH:-/vol/checkpoints/tir_sft_checkpoints/tir_sft_project/tir_sft_lr5e-5_ep6/model}"
OUTPUT_NAME="${OUTPUT_NAME:-tir_sft_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/evaluation/eval_results}"

echo "=== Evaluating TIR SFT checkpoint ==="
echo "Model: ${MODEL_PATH}"

# With tools (multi-turn)
modal run modal_train.py eval_tir -- \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "${OUTPUT_NAME}_with_tools" \
    --num_responses 16 \
    --max_tool_turns 5

# Without tools (baseline comparison)
modal run modal_train.py eval_tir -- \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "${OUTPUT_NAME}_no_tools" \
    --num_responses 16 \
    --no_tools

echo "=== Done. Results at ${OUTPUT_DIR}/ ==="
echo "Download with: modal volume get default-proj-training evaluation/eval_results ./eval_results"
