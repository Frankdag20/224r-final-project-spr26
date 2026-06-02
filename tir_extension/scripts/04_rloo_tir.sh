#!/bin/bash
# Stage 3: RLOO with hierarchical reward + optional DPO self-critic.
#
# Usage:
#   source .env && bash tir_extension/scripts/04_rloo_tir.sh vanilla         # vanilla reward, no self-critic (baseline)
#   source .env && bash tir_extension/scripts/04_rloo_tir.sh phase1          # hierarchical reward, no self-critic
#   source .env && bash tir_extension/scripts/04_rloo_tir.sh phase2          # hierarchical reward + DPO self-critic
#   source .env && bash tir_extension/scripts/04_rloo_tir.sh both            # phase1 + phase2 in parallel
#   source .env && bash tir_extension/scripts/04_rloo_tir.sh all             # all three in parallel
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

NUM_STEPS="${NUM_STEPS:-150}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-128}"
LR="${LR:-1e-5}"
ENTROPY="${ENTROPY:-0.01}"
KL="${KL:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-5}"
ACTIVE_TOOLS="${ACTIVE_TOOLS:-calculator,number_tracker,running_total}"

PHASE="${1:-both}"

run_vanilla() {
    echo "=== Vanilla: RLOO + compute_score reward (no hierarchical, no self-critic) ==="
    MODAL_APP_NAME="rloo-vanilla" \
    modal run --detach "$PROJECT_ROOT/modal_train.py" tir_rloo -- \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --save_dir "$SAVE_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "rloo_vanilla" \
        --num_training_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --group_size "$GROUP_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" \
        --warmup_ratio 0.0 \
        --weight_decay "$WEIGHT_DECAY" \
        --entropy_coefficient "$ENTROPY" \
        --kl_divergence_coefficient "$KL" \
        --max_tool_turns "$MAX_TOOL_TURNS" \
        --initial_active_tools "$ACTIVE_TOOLS" \
        --save_every_n_steps 50 \
        --enable_chunked_prefill \
        --use_vanilla_reward
}

run_phase1() {
    echo "=== Phase 1: RLOO + Hierarchical Reward (no self-critic) ==="
    MODAL_APP_NAME="rloo-nosc" \
    modal run --detach "$PROJECT_ROOT/modal_train.py" tir_rloo -- \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --save_dir "$SAVE_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "rloo_nosc" \
        --num_training_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --group_size "$GROUP_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" \
        --warmup_ratio 0.0 \
        --weight_decay "$WEIGHT_DECAY" \
        --entropy_coefficient "$ENTROPY" \
        --kl_divergence_coefficient "$KL" \
        --max_tool_turns "$MAX_TOOL_TURNS" \
        --initial_active_tools "$ACTIVE_TOOLS" \
        --save_every_n_steps 50 \
        --enable_chunked_prefill
}

run_phase2() {
    local suffix="${1:-}"
    local app_name="rloo-sc${suffix:+-$suffix}"
    local wandb_name="rloo_sc${suffix:+_$suffix}"
    echo "=== Phase 2: RLOO + Hierarchical Reward + DPO Self-Critic (${app_name}) ==="
    MODAL_APP_NAME="$app_name" \
    modal run --detach "$PROJECT_ROOT/modal_train.py" tir_rloo -- \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --save_dir "$SAVE_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "$wandb_name" \
        --num_training_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --group_size "$GROUP_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" \
        --warmup_ratio 0.0 \
        --weight_decay "$WEIGHT_DECAY" \
        --entropy_coefficient "$ENTROPY" \
        --kl_divergence_coefficient "$KL" \
        --max_tool_turns "$MAX_TOOL_TURNS" \
        --initial_active_tools "$ACTIVE_TOOLS" \
        --save_every_n_steps 50 \
        --enable_chunked_prefill \
        --self_critic \
        --self_critic_every_k 5 \
        --self_critic_n_samples 8 \
        --self_critic_beta 0.3
}

SUFFIX="${2:-}"

case "$PHASE" in
    vanilla) run_vanilla ;;
    phase1) run_phase1 ;;
    phase2) run_phase2 "$SUFFIX" ;;
    both)
        run_phase1
        run_phase2 "$SUFFIX"
        ;;
    all)
        run_vanilla
        run_phase1
        run_phase2 "$SUFFIX"
        ;;
    *) echo "Usage: $0 [vanilla|phase1|phase2|both|all] [suffix]"; exit 1 ;;
esac

echo ""
echo "All RLOO runs submitted to Modal (detached)."
echo "Monitor at https://modal.com"
