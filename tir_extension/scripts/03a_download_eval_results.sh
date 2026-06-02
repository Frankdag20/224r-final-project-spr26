#!/bin/bash
# Download TIR SFT eval results from Modal volume and compute pass@k.
#
# Usage:
#   bash tir_extension/scripts/03a_download_eval_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_DIR="$PROJECT_ROOT/eval_results"

mkdir -p "$EVAL_DIR"

echo "Downloading eval results from Modal volume..."
modal volume get default-proj-training evaluation/eval_results/ "$EVAL_DIR/" --force

# Flatten nested eval_results/ if modal creates one
if [ -d "$EVAL_DIR/eval_results" ]; then
    cp "$EVAL_DIR/eval_results/"*.json "$EVAL_DIR/" 2>/dev/null || true
    rm -rf "$EVAL_DIR/eval_results"
fi

echo ""
echo "Computing pass@k for all eval results:"
echo ""

python3 -c "
import json, glob, os
import numpy as np

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

eval_dir = '$EVAL_DIR'
files = sorted(glob.glob(os.path.join(eval_dir, '*.json')))

tir_files = [f for f in files if any(tag in os.path.basename(f) for tag in [
    '3tool_from_sft', '3tool_from_base', 'calc_only_from_sft', 'calc_only_from_base'
])]

if not tir_files:
    print('No TIR eval result files found.')
    exit()

for fpath in tir_files:
    name = os.path.basename(fpath).replace('.json', '')
    with open(fpath) as f:
        data = [json.loads(line) for line in f]
    results = []
    for k in [1, 4, 8, 16]:
        vals = []
        for row in data:
            n = len(row['scores'])
            c = sum(1 for s in row['scores'] if s >= 1.0)
            vals.append(pass_at_k(n, c, k))
        results.append(f'pass@{k}={np.mean(vals):.1%}')
    print(f'{name}: {\"  \".join(results)}')
"
