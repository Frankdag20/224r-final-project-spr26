#!/bin/bash
# Stage 5: Evaluate TIR RLOO checkpoints (with and without tools).
#
# Usage (run each in a separate terminal):
#   source .env && bash tir_extension/scripts/05_eval_tir_rloo.sh vanilla_step50
#   source .env && bash tir_extension/scripts/05_eval_tir_rloo.sh vanilla_latest
#   source .env && bash tir_extension/scripts/05_eval_tir_rloo.sh nosc_step50
#   source .env && bash tir_extension/scripts/05_eval_tir_rloo.sh nosc_latest
#   source .env && bash tir_extension/scripts/05_eval_tir_rloo.sh sc_step50
#
# Download results when done:
#   modal volume get default-proj-training evaluation/eval_results ./eval_results

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/vol/evaluation/eval_results}"
CKPT_ROOT="/vol/checkpoints/tir_rloo_checkpoints/tir_rloo_project"

export MODAL_GPU="${MODAL_GPU:-H100}"

EVAL="${1:?Usage: $0 [vanilla_step50|vanilla_latest|nosc_step50|nosc_latest|sc_step50]}"

case "$EVAL" in
    vanilla_step50)
        MODEL="$CKPT_ROOT/rloo_vanilla/epoch_0_step_50/model"
        NAME="rloo_vanilla_step50" ;;
    vanilla_latest)
        MODEL="$CKPT_ROOT/rloo_vanilla/latest_checkpoint/model"
        NAME="rloo_vanilla_latest" ;;
    nosc_step50)
        MODEL="$CKPT_ROOT/rloo_nosc/epoch_0_step_50/model"
        NAME="rloo_nosc_step50" ;;
    nosc_latest)
        MODEL="$CKPT_ROOT/rloo_nosc/latest_checkpoint/model"
        NAME="rloo_nosc_latest" ;;
    sc_step50)
        MODEL="$CKPT_ROOT/rloo_sc/epoch_0_step_50/model"
        NAME="rloo_sc_step50" ;;
    *) echo "Usage: $0 [vanilla_step50|vanilla_latest|nosc_step50|nosc_latest|sc_step50]"; exit 1 ;;
esac

echo "=== Evaluating: $NAME (with tools) ==="
MODAL_APP_NAME="eval-${NAME}-tools" \
modal run --detach modal_train.py eval_tir -- \
    --model_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "${NAME}_with_tools" \
    --num_responses 16 \
    --max_tool_turns 5

echo "=== Evaluating: $NAME (no tools) ==="
MODAL_APP_NAME="eval-${NAME}-notools" \
modal run --detach modal_train.py eval_tir -- \
    --model_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "${NAME}_no_tools" \
    --num_responses 16 \
    --no_tools

echo ""
echo "Both eval jobs submitted for $NAME."
echo "Download results when done:"
echo "  modal volume get default-proj-training evaluation/eval_results ./eval_results"
