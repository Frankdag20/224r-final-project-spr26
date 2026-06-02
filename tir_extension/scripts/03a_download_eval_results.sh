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
modal volume get default-proj-training evaluation/eval_results/ "$EVAL_DIR/"

echo ""
echo "Computing pass@k for all eval results:"
echo ""

python3 -c "
import json, glob, os

eval_dir = '$EVAL_DIR'
files = sorted(glob.glob(os.path.join(eval_dir, '*.json')))

# Only process TIR SFT eval files
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
    total = len(data)
    results = []
    for k in [1, 4, 8, 16]:
        count = sum(1 for row in data if any(s >= 1.0 for s in row['scores'][:k]))
        results.append(f'pass@{k}={count}/{total}={count/total:.1%}')
    print(f'{name}: {\"  \".join(results)}')
"
