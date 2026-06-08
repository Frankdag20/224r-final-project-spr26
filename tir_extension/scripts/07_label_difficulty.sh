#!/bin/bash
# Stage 7: Label Countdown dataset with difficulty using Qwen2.5-72B-Instruct.
#
# Launches a SINGLE shard on Modal (2 H100s). Run each shard in a separate
# terminal to parallelize across 5 shards (10 H100s total).
#
# Usage (run each in a separate terminal):
#   source .env && bash tir_extension/scripts/07_label_difficulty.sh 0
#   source .env && bash tir_extension/scripts/07_label_difficulty.sh 1
#   source .env && bash tir_extension/scripts/07_label_difficulty.sh 2
#   source .env && bash tir_extension/scripts/07_label_difficulty.sh 3
#   source .env && bash tir_extension/scripts/07_label_difficulty.sh 4
#
# To run on more data later (e.g. remaining 300k):
#   DATASET_SPLIT="train[200000:490314]" bash tir_extension/scripts/07_label_difficulty.sh 0
#   (use a different OUTPUT_DIR or merge manually)
#
# Supports resume: if a shard is interrupted, re-run the same command and it
# will pick up where it left off (reads existing output lines to skip).
#
# Quick test (50 examples, single shard):
#   DATASET_SPLIT="train[:50]" NUM_SHARDS=1 BATCH_SIZE=16 \
#     bash tir_extension/scripts/07_label_difficulty.sh 0
#
# Download results when done:
#   modal volume get default-proj-training curriculum/difficulty_labels ./difficulty_labels
#   cat difficulty_labels/shard_*.jsonl > difficulty_labels/all.jsonl

set -euo pipefail

NUM_SHARDS="${NUM_SHARDS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/vol/curriculum/difficulty_labels}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
DATASET_SPLIT="${DATASET_SPLIT:-train[:200000]}"
BATCH_SIZE="${BATCH_SIZE:-512}"

SHARD_ID="${1:?Usage: $0 <shard_id>  (0-4)}"

echo "=== Launching difficulty labeling shard $SHARD_ID/$NUM_SHARDS ==="
echo "Model: $MODEL_NAME"
echo "Dataset split: $DATASET_SPLIT"
echo "Output: $OUTPUT_DIR/shard_${SHARD_ID}.jsonl"
echo ""

MODAL_APP_NAME="difficulty-shard-${SHARD_ID}" \
MODAL_GPU="H100:2" \
modal run --detach modal_train.py label_difficulty -- \
    --model_name "$MODEL_NAME" \
    --shard_id "$SHARD_ID" \
    --num_shards "$NUM_SHARDS" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_split "$DATASET_SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --temperature 0.0 \
    --max_tokens 1024 \
    --max_tool_turns 5

echo ""
echo "Shard $SHARD_ID submitted to Modal (detached)."
echo "Monitor at https://modal.com"
