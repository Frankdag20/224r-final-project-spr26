#!/bin/bash
# Upload latest RLOO checkpoints from Modal volume to HuggingFace.
#
# Usage:
#   source .env && bash tir_extension/scripts/06_upload_rloo_models.sh
#
# Prerequisites: HF_TOKEN set, huggingface-cli installed

set -euo pipefail

HF_USER="${HF_USER:-sbfisher}"
CKPT_ROOT="checkpoints/tir_rloo_checkpoints/tir_rloo_project"
LOCAL_TMP="/tmp/rloo_models"

MODELS=(
    "rloo_vanilla"
    "rloo_nosc"
    "rloo_sc"
)

mkdir -p "$LOCAL_TMP"

for model in "${MODELS[@]}"; do
    REMOTE_PATH="$CKPT_ROOT/$model/latest_checkpoint/model"
    LOCAL_PATH="$LOCAL_TMP/$model"
    REPO_NAME="$HF_USER/tir-$model"

    echo ""
    echo "=== Downloading $model from Modal volume ==="
    rm -rf "$LOCAL_PATH"
    mkdir -p "$LOCAL_PATH"
    modal volume get default-proj-training "$REMOTE_PATH" "$LOCAL_PATH"

    echo "=== Uploading $model to $REPO_NAME ==="
    hf upload "$REPO_NAME" "$LOCAL_PATH" . --repo-type model

    echo "Done: $REPO_NAME"
done

echo ""
echo "========================================="
echo "All models uploaded! Links:"
echo "========================================="
for model in "${MODELS[@]}"; do
    echo "  https://huggingface.co/$HF_USER/tir-$model"
done
echo ""
