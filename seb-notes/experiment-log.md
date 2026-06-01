# Experiment Log

## Run 1: Generate Tool-Use Trajectories — 3-tool (2000 problems)
- **Date**: 2026-06-01
- **Command**: `uv run python tir_extension/sft_tir/generate_trajectories.py --n_problems 2000 --output_path tir_trajectories_2000.json --max_workers 16`
- **Teacher model**: gpt-4o-mini
- **Active tools**: calculator, number_tracker, running_total
- **Dataset source**: asingh15/countdown_tasks_3to4 (train split)
- **Status**: ABANDONED (hit OpenAI 10k requests/day limit at ~800/2000)
- **Progress at ~800/2000**: ~56% kept (451 kept, 349 failed). All failures were wrong_answer until 800, then rate_limit_exhausted appeared.
- **Notes**: OpenAI Tier 1 has 10k RPD limit. Multiple restarts + retries burned through daily quota. Switched to DeepSeek.

## Run 2: Generate Tool-Use Trajectories — 3-tool with DeepSeek (2000 problems)
- **Date**: 2026-06-01
- **Command**: `uv run python tir_extension/sft_tir/generate_trajectories.py --n_problems 2000 --output_path tir_trajectories_2000.json --max_workers 32 --teacher_model deepseek-chat`
- **Teacher model**: deepseek-chat (DeepSeek V3, via DeepSeek API)
- **Active tools**: calculator, number_tracker, running_total
- **Dataset source**: asingh15/countdown_tasks_3to4 (train split)
- **API**: DeepSeek (concurrency-based limits, no RPD cap — much better than OpenAI Tier 1)
- **Status**: COMPLETE
- **Result**: 1707 trajectories (85.4% success rate), all score=1.0, all with tool calls
  - Tool usage: calculator (2083), number_tracker (974), running_total (578)
  - Problem mix: 993 three-number, 714 four-number
  - Runtime: ~7 minutes with 32 workers
  - Output: `tir_trajectories_2000.json`

## Run 3 (planned): Generate Tool-Use Trajectories — calculator-only (2000 problems)
- **Command**: `uv run python tir_extension/sft_tir/generate_trajectories.py --n_problems 2000 --output_path tir_trajectories_calc_only.json --max_workers 32 --active_tools calculator --teacher_model deepseek-chat`
- **Teacher model**: deepseek-chat
- **Active tools**: calculator only (matches literature — Tool-Star, Understanding TIR)
- **Rationale**: number_tracker and running_total don't add real computational value. Cleaner baseline.
- **Status**: PENDING (after Run 2 finishes)

---

## Prior Data (from Mahmood's branch `tir_extension_mih`)
- `tir_trajectories.json`: 414 trajectories, all score=1.0, all with tool calls
  - Tool usage: calculator (559 calls), number_tracker (90), running_total (34)
  - Generated from 500 problems with gpt-4o-mini
