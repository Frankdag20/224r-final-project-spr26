# Difficulty Labeling Plan

## Goal
Label all examples in the Countdown dataset with difficulty scores using a large teacher model (Qwen3-72B), to enable curriculum training for the small 0.5B learner.

## Model
- **Qwen2.5-72B-Instruct** (same model family as the 0.5B learner)
- 2x H100 per instance, tensor_parallel_size=2
- vLLM for batched inference
- Temperature 0.0 (greedy/deterministic)

## Dataset
- Full Countdown dataset: 490,314 examples
- Source: `asingh15/countdown_tasks_3to4` (train split)

## Procedure

For each problem, run the model **twice** (1 sample per mode):

### Run 1: No Tools
- System prompt: plain Countdown instructions (no tool descriptions) + difficulty instruction
- Model solves single-turn, then rates difficulty after its answer
- Score the answer for correctness

### Run 2: With Tools
- System prompt: full I^T prompt with all three tools (calculator, number_tracker, running_total) + difficulty instruction
- Multi-turn tool execution — model generates `<use_tool>` calls, system executes them and returns `<tool_result>`, model continues (max 5 turns)
- Model solves, then rates difficulty after its answer
- Score the answer for correctness

### Difficulty Rating (both modes)
- Appended to system prompt: after providing `<answer>`, rate the problem's difficulty
- Format: optional reasoning, then `<difficulty>N</difficulty>` where N is integer 1-10
- Comes after the answer so the model can factor in whether it succeeded
- One difficulty score per mode (so two per problem total)

## Extracted Features (per problem)

| Field | Description |
|-------|-------------|
| `no_tools_correct` | Did the model solve without tools? (bool) |
| `with_tools_correct` | Did the model solve with tools? (bool) |
| `no_tools_reasoning_length` | Chars before first `<answer>` tag (no tools) |
| `with_tools_reasoning_length` | Chars before first `<answer>` tag (with tools) |
| `no_tools_difficulty` | Self-assessed difficulty 1-10 (no tools) |
| `with_tools_difficulty` | Self-assessed difficulty 1-10 (with tools) |

## Output Schema (per problem)
```json
{
  "index": 12345,
  "target": 98,
  "nums": [44, 19, 35],
  "no_tools_correct": true,
  "no_tools_difficulty": 3,
  "no_tools_reasoning_length": 245,
  "no_tools_response": "...",
  "with_tools_correct": true,
  "with_tools_difficulty": 2,
  "with_tools_reasoning_length": 312,
  "with_tools_response": "..."
}
```

## Infrastructure

### Parallelization
- 5 shards, each on 2 H100s (10 H100s total = user's Modal limit)
- Each shard processes ~98k problems
- Batch size 512 for vLLM

### Files
- **Script**: `tir_extension/curriculum/label_difficulty.py`
- **Modal entry**: `modal_train.py` → `label_difficulty` (hardcoded `gpu="H100:2"`)
- **Launch**: `tir_extension/scripts/07_label_difficulty.sh`

### Running
```bash
source .env && bash tir_extension/scripts/07_label_difficulty.sh
```

### Downloading results
```bash
modal volume get default-proj-training curriculum/difficulty_labels ./difficulty_labels
cat difficulty_labels/shard_*.jsonl > difficulty_labels/all.jsonl
```

## Estimated Cost & Time
- 10 H100s × ~3-4 hours = ~30-40 H100-hours
- At ~$3.95/hr per H100 = **~$120-160**
- Wall-clock time: **~3-4 hours**

## Curriculum Categories (TBD)
Possible splits based on outputs:
- **Easy**: both runs correct, low difficulty scores
- **Hard**: one or both runs fail, high difficulty scores
- **Tool-beneficial**: with-tools correct but no-tools failed
- **Reasoning-intensive**: long reasoning traces even when correct
- Exact thresholds and categories to be determined after seeing the distribution
