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

## Run 3: Generate Tool-Use Trajectories — calculator-only (2000 problems)
- **Date**: 2026-06-01
- **Command**: `uv run python tir_extension/sft_tir/generate_trajectories.py --n_problems 2000 --output_path tir_trajectories_calc_only.json --max_workers 32 --active_tools calculator --teacher_model deepseek-chat`
- **Teacher model**: deepseek-chat
- **Active tools**: calculator only (matches literature — Tool-Star, Understanding TIR)
- **Status**: COMPLETE
- **Result**: 1699 trajectories, all score=1.0, calculator only (1823 calls)
  - Output: `tir_trajectories_calc_only.json`

## Run 4: TIR SFT — 3-tool, from vanilla SFT checkpoint
- **Date**: 2026-06-01
- **Base model**: `asingh15/qwen-sft-countdown-defaultproj` (vanilla SFT checkpoint)
- **Dataset**: `tir_trajectories_2000.json` (1707 trajectories, 3 tools)
- **Config**: lr=1e-5, 10 epochs, batch=16 (8x2), max_response_length=2048, warmup_ratio=0.05, gradient_checkpointing=1
- **W&B name**: `3tool_from_sft`
- **W&B link**:  (I can't seem to share this link directly bc of the limited WANDB plan, might have to look into adding teammates to the WANDB team so they can see the plots)
- **Status**: COMPLETE 
- **Results**:
  - Final train loss: 0.3308, train accuracy: 0.8906
  - Final eval loss: 0.3908, eval accuracy: 0.8746
  - Tool-result tokens skipped per epoch: 42,759
  - Converged by ~epoch 6, no overfitting (eval loss plateaued, did not increase)
- **Checkpoint**: `/vol/checkpoints/tir_sft_checkpoints/tir_sft_project/3tool_from_sft/model/` (Modal volume)
- **HF repo**: not pushed (run completed before HF push was added)
- **Eval results (with tools)**: `eval_results/3tool_from_sft_with_tools.json`
  - pass@1 = 62.0%, pass@4 = 72.0%, pass@8 = 76.0%, pass@16 = 78.0%
- **Eval results (no tools)**: `eval_results/3tool_from_sft_no_tools.json`
  - pass@1 = 34.0%, pass@4 = 64.0%, pass@8 = 76.0%, pass@16 = 78.0%
- **Key finding**: Tool execution nearly doubles pass@1 (34% → 62%). At higher k they converge.
  - No-tools result (34%) comparable to Mahmood's 37.6% (same setup, larger dataset).
- **Rationale**: Most comparable to Mahmood's run (same base model, same tools) but 4x larger dataset (1707 vs 414).

## Run 5: TIR SFT — 3-tool, from base Qwen (cold start)
- **Date**: 2026-06-01
- **Base model**: `Qwen/Qwen2.5-0.5B` (pretrained, no prior SFT)
- **Dataset**: `tir_trajectories_2000.json` (1707 trajectories, 3 tools)
- **Config**: lr=1e-5, 10 epochs, batch=16 (8x2), max_response_length=2048, warmup_ratio=0.05, gradient_checkpointing=1
- **W&B name**: `3tool_from_base`
- **HF repo**: `sbfisher/tir-sft-3tool-from-base` (auto-push on completion)
- **Checkpoint**: `/vol/checkpoints/tir_sft_checkpoints/tir_sft_project/3tool_from_base/model/` (Modal volume)
- **Rationale**: Cold-start ablation. Does the model need prior Countdown SFT, or can it learn task + tool use together?
- **Status**: COMPLETE
- **Eval results (with tools)**: `eval_results/3tool_from_base_with_tools.json`
  - pass@1 = 44.0%, pass@4 = 66.0%, pass@8 = 70.0%, pass@16 = 70.0%
- **Eval results (no tools)**: `eval_results/3tool_from_base_no_tools.json`
  - pass@1 = 38.0%, pass@4 = 66.0%, pass@8 = 70.0%, pass@16 = 72.0%
- **Key finding**: Cold start hurts significantly vs warm start (44% vs 62% pass@1 with tools). Tool boost smaller (+6pp vs +28pp).

## Run 6: TIR SFT — calculator-only, from vanilla SFT checkpoint
- **Date**: 2026-06-01
- **Base model**: `asingh15/qwen-sft-countdown-defaultproj` (vanilla SFT checkpoint)
- **Dataset**: `tir_trajectories_calc_only.json` (1699 trajectories, calculator only)
- **Config**: lr=1e-5, 10 epochs, batch=16 (8x2), max_response_length=2048, warmup_ratio=0.05, gradient_checkpointing=1
- **W&B name**: `calc_only_from_sft`
- **HF repo**: `sbfisher/tir-sft-calc-only-from-sft` (auto-push on completion)
- **Checkpoint**: `/vol/checkpoints/tir_sft_checkpoints/tir_sft_project/calc_only_from_sft/model/` (Modal volume)
- **Rationale**: Cleaner tool setup matching literature. Tests whether single useful tool (calculator) outperforms 3 tools (2 of which are busywork).
- **Status**: COMPLETE
- **Eval results (with tools)**: `eval_results/calc_only_from_sft_with_tools.json`
  - pass@1 = 56.0%, pass@4 = 72.0%, pass@8 = 74.0%, pass@16 = 78.0%
- **Eval results (no tools)**: `eval_results/calc_only_from_sft_no_tools.json`
  - pass@1 = 46.0%, pass@4 = 68.0%, pass@8 = 74.0%, pass@16 = 78.0%
- **Key finding**: Calculator-only slightly below 3-tool with tools (56% vs 62% pass@1), but better without tools (46% vs 34%).

## Run 7: TIR SFT — calculator-only, from base Qwen (cold start)
- **Date**: 2026-06-01
- **Base model**: `Qwen/Qwen2.5-0.5B` (pretrained, no prior SFT)
- **Dataset**: `tir_trajectories_calc_only.json` (1699 trajectories, calculator only)
- **Config**: lr=1e-5, 10 epochs, batch=16 (8x2), max_response_length=2048, warmup_ratio=0.05, gradient_checkpointing=1
- **W&B name**: `calc_only_from_base`
- **HF repo**: `sbfisher/tir-sft-calc-only-from-base` (auto-push on completion)
- **Checkpoint**: `/vol/checkpoints/tir_sft_checkpoints/tir_sft_project/calc_only_from_base/model/` (Modal volume)
- **Rationale**: Cleanest experiment — cold start + single tool. Matches Tool-Star literature most closely.
- **Status**: COMPLETE
- **Eval results (with tools)**: `eval_results/calc_only_from_base_with_tools.json`
  - pass@1 = 52.0%, pass@4 = 62.0%, pass@8 = 66.0%, pass@16 = 68.0%
- **Eval results (no tools)**: `eval_results/calc_only_from_base_no_tools.json`
  - pass@1 = 38.0%, pass@4 = 62.0%, pass@8 = 66.0%, pass@16 = 68.0%
- **Key finding**: Cold start + calc-only has decent tool boost (+14pp pass@1). Lowest ceiling (68% pass@16).

---

## TIR SFT Eval Summary (Runs 4-7)

| Run | Base Model | Tools | pass@1 | pass@4 | pass@8 | pass@16 |
|-----|-----------|-------|--------|--------|--------|---------|
| 3tool_from_sft | Vanilla SFT | Yes | **62%** | 72% | 76% | 78% |
| 3tool_from_sft | Vanilla SFT | No | 34% | 64% | 76% | 78% |
| calc_only_from_sft | Vanilla SFT | Yes | 56% | 72% | 74% | 78% |
| calc_only_from_sft | Vanilla SFT | No | 46% | 68% | 74% | 78% |
| 3tool_from_base | Base Qwen | Yes | 44% | 66% | 70% | 70% |
| 3tool_from_base | Base Qwen | No | 38% | 66% | 70% | 72% |
| calc_only_from_base | Base Qwen | Yes | 52% | 62% | 66% | 68% |
| calc_only_from_base | Base Qwen | No | 38% | 62% | 66% | 68% |

**Key takeaways:**
1. **Warm start (vanilla SFT) >> cold start (base Qwen)** across the board. Prior task knowledge matters.
2. **Tool execution boosts pass@1 significantly** — biggest gain with 3-tool warm start (+28pp), smallest with 3-tool cold start (+6pp).
3. **3-tool vs calc-only**: 3-tool wins with tools (62% vs 56%), but calc-only is better without tools (46% vs 34%), suggesting extra tools may hurt when model must hallucinate results.
4. **Best overall: 3tool_from_sft with tools (62% pass@1, 78% pass@16)** — best candidate for RLOO.

---

## Prior Data (from Mahmood's branch `tir_extension_mih`)
- `tir_trajectories.json`: 414 trajectories, all score=1.0, all with tool calls
  - Tool usage: calculator (559 calls), number_tracker (90), running_total (34)
  - Generated from 500 problems with gpt-4o-mini
