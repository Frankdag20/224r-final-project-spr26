#!/bin/bash
# Stage 1: Generate tool-using trajectories with gpt-4o-mini teacher.
#
# Prerequisites:
#   export OPENAI_API_KEY=...
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...
#
# This calls gpt-4o-mini to solve Countdown problems using tools,
# filters for correct solutions, and saves to /vol/checkpoints/tir_trajectories.json
#
# Cost: ~$0.50 for 500 problems.

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

N_PROBLEMS="${N_PROBLEMS:-500}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt-4o-mini}"
OUTPUT_PATH="${OUTPUT_PATH:-/vol/checkpoints/tir_trajectories.json}"

echo "=== Stage 1: Generating ${N_PROBLEMS} tool-use trajectories with ${TEACHER_MODEL} ==="

modal run modal_train.py tir_gen -- \
    --n_problems "$N_PROBLEMS" \
    --teacher_model "$TEACHER_MODEL" \
    --output_path "$OUTPUT_PATH" \
    --require_tool_use True \
    --max_attempts 5 \
    --temperature 0.7

echo "=== Done. Trajectories saved to ${OUTPUT_PATH} on Modal volume. ==="
echo "Verify with: modal volume ls default-proj-training /checkpoints/"
