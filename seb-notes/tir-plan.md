# TIR Extension Plan

## Goal
Add tool-integrated reasoning (TIR) to the Countdown RLOO pipeline, following the approach from "Understanding Tool-Integrated Reasoning" (Lin & Xu) and "Tool-Star" (Dong et al.).

## Three-Stage Pipeline

### Stage 1: Generate Synthetic Tool-Use Trajectories
- **Script**: `bash tir_extension/scripts/01_generate_trajectories.sh`
- Uses gpt-4o-mini via OpenAI API to solve Countdown problems with `<use_tool>` calls
- Same prompts from `asingh15/countdown_tasks_3to4`, just new completions with tool use
- Filters for correctness (score=1.0), targets ~500 examples
- Output: `/vol/checkpoints/tir_trajectories.json` on Modal volume
- Requires `OPENAI_API_KEY` env var

### Stage 2: Cold-Start SFT on Tool-Use Trajectories
- **Script**: `bash tir_extension/scripts/02_sft_tir.sh`
- SFT from base Qwen2.5-0.5B on the generated trajectories
- Masks `<tool_result>` tokens from loss (model didn't generate those)
- Output: checkpoint at `/vol/checkpoints/tir_sft_checkpoints/` on Modal volume
- **Verify**: `bash tir_extension/scripts/03_eval_tir_sft.sh` — target ~31.5% pass@1

### Stage 3: RLOO with Multi-Turn Tool Execution
- **Script**: `bash tir_extension/scripts/04_rloo_tir.sh`
- Starts from TIR-SFT checkpoint (Stage 2 output)
- **Multi-turn sampling**: model generates until `</use_tool>`, tool executes, result appended, generation resumes
- Same `compute_score` verifier and dataset as baseline RLOO
- `<tool_result>` tokens masked from policy gradient
- DSPy dynamic tool selection disabled (`--reanalyze_every_n_steps 0`)
- **Evaluate**: `bash tir_extension/scripts/05_eval_tir_rloo.sh`

## How to Run (step by step)

```bash
# 0. Ensure env vars are set
export OPENAI_API_KEY=...
export WANDB_API_KEY=...
export HF_TOKEN=...

# 1. Generate trajectories (~500 problems, <$1)
bash tir_extension/scripts/01_generate_trajectories.sh

# 2. SFT warm-start
bash tir_extension/scripts/02_sft_tir.sh

# 3. Evaluate SFT checkpoint (with and without tools)
bash tir_extension/scripts/03_eval_tir_sft.sh

# 4. RLOO training
bash tir_extension/scripts/04_rloo_tir.sh

# 5. Evaluate RLOO checkpoint
bash tir_extension/scripts/05_eval_tir_rloo.sh

# Download eval results locally
modal volume get default-proj-training evaluation/eval_results ./eval_results
```

## What Was Built (new code)
- `tir_extension/training/tir_sampling_worker.py` — multi-turn vLLM sampling with iterative tool execution
- `tir_extension/tools/system_prompt.py` — tool-integrated system prompt for student model
- `evaluation/countdown_eval_tir.py` — evaluation script with multi-turn tool support
- `tir_extension/scripts/*.sh` — shell scripts for each stage
- `modal_train.py` updated with `eval_tir` entrypoint
- `rloo_tir.py` updated to use `TIRSamplingWorker` and `--multi_turn_tools` flag

## What Already Existed (from Frank's branch)
- `tir_extension/tools/tool_pool.py` — tool registry, execution, parsing (used as-is)
- `tir_extension/sft_tir/generate_trajectories.py` — trajectory generation with gpt-4o-mini
- `tir_extension/sft_tir/sft_tir.py` + `sft_tir_dataset.py` — SFT with tool-result masking
- `tir_extension/training/rloo_tir.py` — TIR RLOO trainer (updated for multi-turn)
- `rloo_trainer/rloo_update_worker.py` — already supports `tool_result_mask`

## Not Using (Frank's extras)
- DSPy dynamic tool selection (disabled with `--reanalyze_every_n_steps 0`)
- Failure database analysis
- Self-critic GRPO (`rloo_self_critic.py`)
- Hierarchical reward (`hierarchical_reward.py`)

## Open Questions
- Do we need all 3 relevant tools or just calculator?
- How much does multi-turn vs post-hoc actually matter for Countdown?
- Should we start SFT from base Qwen or from the vanilla SFT checkpoint?
