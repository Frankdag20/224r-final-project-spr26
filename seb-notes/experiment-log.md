# Experiment Log

## Run 1: Generate Tool-Use Trajectories (2000 problems)
- **Date**: 2026-06-01
- **Command**: `uv run python tir_extension/sft_tir/generate_trajectories.py --n_problems 2000 --output_path tir_trajectories_2000.json`
- **Teacher model**: gpt-4o-mini
- **Dataset source**: asingh15/countdown_tasks_3to4 (train split)
- **Status**: RUNNING
- **Notes**: Mohammed already generated 414 trajectories (tir_trajectories.json) from 500 problems (~83% success rate). This run targets 2000 problems, expecting ~1600 usable trajectories.
- **Result**: (pending)

---

## Prior Data (from Mohammed's branch `tir_extension_mih`)
- `tir_trajectories.json`: 414 trajectories, all score=1.0, all with tool calls
  - Tool usage: calculator (559 calls), number_tracker (90), running_total (34)
  - Generated from 500 problems with gpt-4o-mini
