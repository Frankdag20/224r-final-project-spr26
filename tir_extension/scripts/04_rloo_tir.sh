#!/bin/bash
# Stage 3: RLOO with hierarchical reward + optional DPO self-critic.
#
# Usage:
#   source .env && bash tir_extension/scripts/04_rloo_tir.sh phase1          # vanilla RLOO + hierarchical reward
#   source .env && bash tir_extension/scripts/04_rloo_tir.sh phase2          # RLOO + hierarchical reward + DPO self-critic
#   source .env && bash tir_extension/scripts/04_rloo_tir.sh both            # launch both in parallel on Modal
#
# Prerequisites:
#   source .env  (needs WANDB_API_KEY, HF_TOKEN)
#   TIR SFT checkpoint available on Modal volume

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Best v2 SFT checkpoint — uses HF repo (volume path may not persist)
MODEL_NAME="${MODEL_NAME:-sbfisher/tir-sft-3tool_from_sft}"
DATASET_NAME="${DATASET_NAME:-asingh15/countdown_tasks_3to4}"
SAVE_DIR="${SAVE_DIR:-/vol/checkpoints/tir_rloo_checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-tir_rloo_project}"

NUM_STEPS="${NUM_STEPS:-250}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
LR="${LR:-1e-5}"
ENTROPY="${ENTROPY:-0.01}"
KL="${KL:-0.0}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-5}"
ACTIVE_TOOLS="${ACTIVE_TOOLS:-calculator,number_tracker,running_total}"

PHASE="${1:-both}"

run_phase1() {
    echo "=== Phase 1: RLOO + Hierarchical Reward (no self-critic) ==="
    MODAL_APP_NAME="rloo-phase1" \
    modal run --detach "$PROJECT_ROOT/modal_train.py" tir_rloo -- \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --save_dir "$SAVE_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "phase1_hier_reward" \
        --num_training_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --group_size "$GROUP_SIZE" \
        --learning_rate "$LR" \
        --warmup_ratio 0.0 \
        --entropy_coefficient "$ENTROPY" \
        --kl_divergence_coefficient "$KL" \
        --max_tool_turns "$MAX_TOOL_TURNS" \
        --initial_active_tools "$ACTIVE_TOOLS" \
        --save_every_n_steps 50 \
        --enable_chunked_prefill &
}

run_phase2() {
    echo "=== Phase 2: RLOO + Hierarchical Reward + DPO Self-Critic ==="
    MODAL_APP_NAME="rloo-phase2" \
    modal run --detach "$PROJECT_ROOT/modal_train.py" tir_rloo -- \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --save_dir "$SAVE_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "phase2_self_critic" \
        --num_training_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --group_size "$GROUP_SIZE" \
        --learning_rate "$LR" \
        --warmup_ratio 0.0 \
        --entropy_coefficient "$ENTROPY" \
        --kl_divergence_coefficient "$KL" \
        --max_tool_turns "$MAX_TOOL_TURNS" \
        --initial_active_tools "$ACTIVE_TOOLS" \
        --save_every_n_steps 50 \
        --enable_chunked_prefill \
        --self_critic \
        --self_critic_every_k 5 \
        --self_critic_n_samples 8 \
        --self_critic_beta 0.1 &
}

case "$PHASE" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    both)
        run_phase1
        run_phase2
        ;;
    *) echo "Usage: $0 [phase1|phase2|both]"; exit 1 ;;
esac

wait
echo ""
echo "All RLOO runs submitted to Modal."
echo "Monitor at https://modal.com"
