#!/bin/bash
# TIR SFT Experiments — run all 4 ablations on Modal.
#
# 2x2 grid: {3-tool, calc-only} x {from vanilla SFT, from base Qwen}
#
# Prerequisites:
#   source .env  (needs WANDB_API_KEY, HF_TOKEN)
#   Trajectory files uploaded to Modal volume:
#     /vol/checkpoints/tir_trajectories_2000.json
#     /vol/checkpoints/tir_trajectories_calc_only.json
#
# Usage:
#   bash tir_extension/scripts/02_sft_tir_experiments.sh [run_name|all]
#     3tool_from_sft      = 3-tool, from vanilla SFT checkpoint
#     3tool_from_base     = 3-tool, from base Qwen (cold start)
#     calc_only_from_sft  = calc-only, from vanilla SFT checkpoint
#     calc_only_from_base = calc-only, from base Qwen (cold start)
#     all                 = run all 4 (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Shared config
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-/vol/checkpoints/tir_sft_checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-10}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
WANDB_PROJECT="${WANDB_PROJECT:-tir_sft_project}"
HF_USER="${HF_USER:-sbfisher}"

# Paths to trajectory datasets on Modal volume
TRAJ_3TOOL="/vol/checkpoints/tir_trajectories_2000.json"
TRAJ_CALC_ONLY="/vol/checkpoints/tir_trajectories_calc_only.json"

# Base models
MODEL_SFT="asingh15/qwen-sft-countdown-defaultproj"
MODEL_BASE="Qwen/Qwen2.5-0.5B"

EXPERIMENT="${1:-all}"

run_one() {
    local run_name="$1"
    local model_name="$2"
    local trajectory_path="$3"
    local hf_repo="${HF_USER}/tir-sft-${run_name}"

    echo "=== ${run_name}: model=${model_name}, data=$(basename ${trajectory_path}) ==="

    modal run --detach "$PROJECT_ROOT/modal_train.py" tir_sft -- \
        --model_name "$model_name" \
        --trajectory_path "$trajectory_path" \
        --output_dir "${OUTPUT_DIR_BASE}" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" \
        --num_epochs "$EPOCHS" \
        --max_response_length "$MAX_RESPONSE_LENGTH" \
        --warmup_ratio 0.05 \
        --gradient_checkpointing 1 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "$run_name" \
        --hf_repo "$hf_repo"

    echo "=== ${run_name} launched ==="
    echo ""
}

case "$EXPERIMENT" in
    3tool_from_sft)      run_one "3tool_from_sft"      "$MODEL_SFT"  "$TRAJ_3TOOL" ;;
    3tool_from_base)     run_one "3tool_from_base"      "$MODEL_BASE" "$TRAJ_3TOOL" ;;
    calc_only_from_sft)  run_one "calc_only_from_sft"   "$MODEL_SFT"  "$TRAJ_CALC_ONLY" ;;
    calc_only_from_base) run_one "calc_only_from_base"  "$MODEL_BASE" "$TRAJ_CALC_ONLY" ;;
    all)
        run_one "3tool_from_sft"      "$MODEL_SFT"  "$TRAJ_3TOOL"
        run_one "3tool_from_base"     "$MODEL_BASE" "$TRAJ_3TOOL"
        run_one "calc_only_from_sft"  "$MODEL_SFT"  "$TRAJ_CALC_ONLY"
        run_one "calc_only_from_base" "$MODEL_BASE" "$TRAJ_CALC_ONLY"
        ;;
    *) echo "Usage: $0 [3tool_from_sft|3tool_from_base|calc_only_from_sft|calc_only_from_base|all]"; exit 1 ;;
esac

echo "Monitor runs at https://modal.com"
