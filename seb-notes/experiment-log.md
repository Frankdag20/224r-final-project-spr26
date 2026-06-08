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
  - pass@1 = 56.5%, pass@4 = 70.0%, pass@8 = 74.2%, pass@16 = 78.0%
- **Eval results (no tools)**: `eval_results/3tool_from_sft_no_tools.json`
  - pass@1 = 41.8%, pass@4 = 69.5%, pass@8 = 75.7%, pass@16 = 78.0%
- **Key finding**: Tool execution boosts pass@1 significantly (41.8% → 56.5%). At higher k they converge.
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
  - pass@1 = 50.0%, pass@4 = 65.3%, pass@8 = 68.3%, pass@16 = 70.0%
- **Eval results (no tools)**: `eval_results/3tool_from_base_no_tools.json`
  - pass@1 = 42.0%, pass@4 = 65.4%, pass@8 = 70.0%, pass@16 = 72.0%
- **Key finding**: Cold start hurts vs warm start (50.0% vs 56.5% pass@1 with tools). Tool boost smaller (+8pp vs +14.7pp).

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
  - pass@1 = 57.8%, pass@4 = 70.7%, pass@8 = 74.5%, pass@16 = 78.0%
- **Eval results (no tools)**: `eval_results/calc_only_from_sft_no_tools.json`
  - pass@1 = 50.4%, pass@4 = 68.7%, pass@8 = 73.9%, pass@16 = 78.0%
- **Key finding**: Calculator-only slightly better than 3-tool with tools (57.8% vs 56.5% pass@1), and notably better without tools (50.4% vs 41.8%). Supports hypothesis that calc-only teaches better arithmetic.

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
  - pass@1 = 51.5%, pass@4 = 63.0%, pass@8 = 65.4%, pass@16 = 68.0%
- **Eval results (no tools)**: `eval_results/calc_only_from_base_no_tools.json`
  - pass@1 = 38.5%, pass@4 = 62.6%, pass@8 = 66.8%, pass@16 = 68.0%
- **Key finding**: Cold start + calc-only has decent tool boost (+13pp pass@1). Lowest ceiling (68% pass@16).

---

## TIR SFT Eval Summary (Runs 4-7)

All pass@k values computed using the unbiased estimator: `pass@k = 1 - C(n-c, k) / C(n, k)`.

Vanilla SFT baseline (no TIR): pass@1 = 31.5%, pass@16 = 76.0% (`eval_results/sft_eval_run.json`)

| Run | Base Model | Tools | pass@1 | pass@4 | pass@8 | pass@16 |
|-----|-----------|-------|--------|--------|--------|---------|
| calc_only_from_sft | Vanilla SFT | Yes | **57.8%** | 70.7% | 74.5% | 78.0% |
| 3tool_from_sft | Vanilla SFT | Yes | 56.5% | 70.0% | 74.2% | 78.0% |
| calc_only_from_sft | Vanilla SFT | No | 50.4% | 68.7% | 73.9% | 78.0% |
| calc_only_from_base | Base Qwen | Yes | 51.5% | 63.0% | 65.4% | 68.0% |
| 3tool_from_base | Base Qwen | Yes | 50.0% | 65.3% | 68.3% | 70.0% |
| 3tool_from_base | Base Qwen | No | 42.0% | 65.4% | 70.0% | 72.0% |
| 3tool_from_sft | Vanilla SFT | No | 41.8% | 69.5% | 75.7% | 78.0% |
| calc_only_from_base | Base Qwen | No | 38.5% | 62.6% | 66.8% | 68.0% |

**Key takeaways:**
1. **All TIR models beat vanilla SFT** (31.5% pass@1) — tool-integrated training helps across the board.
2. **calc_only_from_sft is the best model** (57.8% with tools, 50.4% without) — calc-only teaches better arithmetic that transfers even without tool access.
3. **Warm start >> cold start** across the board. Prior task knowledge matters.
4. **Tool execution boosts pass@1** — typically +7-13pp for warm start, +8-13pp for cold start.
5. **3-tool vs calc-only**: calc-only slightly wins with tools (57.8% vs 56.5%) and notably wins without tools (50.4% vs 41.8%), suggesting extra tools add noise without computational value.

---

## Runs 8-11: TIR SFT v2 — with tool-integrated instruction (I^T)

**Motivation:** Runs 4-7 were missing the tool-integrated system prompt (I^T) from the Tool-Star paper (eq. 1). The model was trained with a generic "You are a helpful assistant." system message, meaning it never received explicit instructions about what tools are available or how to call them. This is inconsistent with the Tool-Star formalization where I^T conditions generation at all stages.

**What changed:**
- System prompt now describes the Countdown task, lists available tools with descriptions, shows the `<use_tool>` / `<tool_result>` format with a worked example
- I^T is injected consistently across SFT, RLOO, and eval (previously only eval had it, and even that was unused)
- Calc-only runs get a calc-only I^T (via `--active_tools calculator`); 3-tool runs get the full 3-tool I^T
- Also fixed: SFT dataset was double-wrapping chat template tokens — now handles already-templated prompts correctly

**W&B project:** `tir_sft_project_v2` (old results preserved in `tir_sft_project`)
**Checkpoints:** `/vol/checkpoints/tir_sft_checkpoints/tir_sft_project_v2/` (old preserved in `tir_sft_project/`)

### Run 8: TIR SFT v2 — 3-tool, from vanilla SFT checkpoint
- **Date**: 2026-06-01
- **Base model**: `asingh15/qwen-sft-countdown-defaultproj`
- **Dataset**: `tir_trajectories_2000.json` (1698 trajectories after cleaning 9 malformed, 3 tools)
- **Config**: lr=1e-5, 10 epochs, batch=16 (8x2), max_response_length=2048, warmup_ratio=0.05
- **I^T**: 3-tool system prompt (calculator, number_tracker, running_total)
- **W&B**: `tir_sft_project_v2` / `3tool_from_sft`
- **HF repo**: `sbfisher/tir-sft-3tool_from_sft`
- **Status**: COMPLETE
- **Eval results (with tools)**: pass@1=58.0%, pass@4=71.3%, pass@8=75.0%, pass@16=78.0%
- **Eval results (no tools)**: pass@1=50.6%, pass@4=69.1%, pass@8=76.0%, pass@16=82.0%

### Run 9: TIR SFT v2 — 3-tool, from base Qwen (cold start)
- **Date**: 2026-06-01
- **Base model**: `Qwen/Qwen2.5-0.5B`
- **Dataset**: `tir_trajectories_2000.json` (1707 trajectories, 3 tools)
- **Config**: lr=1e-5, 10 epochs, batch=16 (8x2), max_response_length=2048, warmup_ratio=0.05
- **I^T**: 3-tool system prompt
- **W&B**: `tir_sft_project_v2` / `3tool_from_base`
- **HF repo**: `sbfisher/tir-sft-3tool_from_base`
- **Status**: COMPLETE
- **Eval results (with tools)**: pass@1=53.6%, pass@4=67.7%, pass@8=72.9%, pass@16=78.0%
- **Eval results (no tools)**: pass@1=15.1%, pass@4=41.0%, pass@8=54.5%, pass@16=62.0%

### Run 10: TIR SFT v2 — calculator-only, from vanilla SFT checkpoint
- **Date**: 2026-06-01
- **Base model**: `asingh15/qwen-sft-countdown-defaultproj`
- **Dataset**: `tir_trajectories_calc_only.json` (1699 trajectories, calculator only)
- **Config**: lr=1e-5, 10 epochs, batch=16 (8x2), max_response_length=2048, warmup_ratio=0.05
- **I^T**: calculator-only system prompt (`--active_tools calculator`)
- **W&B**: `tir_sft_project_v2` / `calc_only_from_sft`
- **HF repo**: `sbfisher/tir-sft-calc_only_from_sft`
- **Status**: COMPLETE
- **Eval results (with tools)**: pass@1=57.0%, pass@4=69.9%, pass@8=74.3%, pass@16=78.0%
- **Eval results (no tools)**: pass@1=46.4%, pass@4=66.3%, pass@8=71.1%, pass@16=74.0%

### Run 11: TIR SFT v2 — calculator-only, from base Qwen (cold start)
- **Date**: 2026-06-01
- **Base model**: `Qwen/Qwen2.5-0.5B`
- **Dataset**: `tir_trajectories_calc_only.json` (1699 trajectories, calculator only)
- **Config**: lr=1e-5, 10 epochs, batch=16 (8x2), max_response_length=2048, warmup_ratio=0.05
- **I^T**: calculator-only system prompt (`--active_tools calculator`)
- **W&B**: `tir_sft_project_v2` / `calc_only_from_base`
- **HF repo**: `sbfisher/tir-sft-calc_only_from_base`
- **Status**: COMPLETE
- **Eval results (with tools)**: pass@1=47.9%, pass@4=65.0%, pass@8=68.8%, pass@16=72.0%
- **Eval results (no tools)**: pass@1=15.0%, pass@4=39.8%, pass@8=52.9%, pass@16=62.0%

### TIR SFT v2 Eval Summary (Runs 8-11)

Vanilla SFT baseline (no TIR): pass@1 = 31.5%, pass@16 = 76.0%

| Run | Base Model | Tools | pass@1 | pass@4 | pass@8 | pass@16 |
|-----|-----------|-------|--------|--------|--------|---------|
| 3tool_from_sft v2 | Vanilla SFT | Yes | **58.0%** | **71.3%** | 75.0% | 78.0% |
| calc_only_from_sft v2 | Vanilla SFT | Yes | 57.0% | 69.9% | 74.3% | 78.0% |
| 3tool_from_base v2 | Base Qwen | Yes | 53.6% | 67.7% | 72.9% | 78.0% |
| 3tool_from_sft v2 | Vanilla SFT | No | 50.6% | 69.1% | 76.0% | **82.0%** |
| calc_only_from_base v2 | Base Qwen | Yes | 47.9% | 65.0% | 68.8% | 72.0% |
| calc_only_from_sft v2 | Vanilla SFT | No | 46.4% | 66.3% | 71.1% | 74.0% |
| 3tool_from_base v2 | Base Qwen | No | 15.1% | 41.0% | 54.5% | 62.0% |
| calc_only_from_base v2 | Base Qwen | No | 15.0% | 39.8% | 52.9% | 62.0% |

**v2 vs v1 comparison (best with-tools pass@1):**
- v2 3tool_from_sft: 58.0% vs v1: 56.5% (+1.5pp)
- v2 calc_only_from_sft: 57.0% vs v1: 57.8% (-0.8pp)
- v2 3tool_from_base: 53.6% vs v1: 50.0% (+3.6pp)
- v2 calc_only_from_base: 47.9% vs v1: 51.5% (-3.6pp)

**Key takeaways (v2):**
1. **3tool_from_sft is now the best model** (58.0% pass@1 with tools) — I^T helps the 3-tool model leverage all tools more effectively.
2. **I^T dramatically hurts cold-start models without tools** — base Qwen models drop to ~15% pass@1 without tools (vs 38-42% in v1). The model becomes tool-dependent when trained with I^T from scratch.
3. **Warm-start models retain no-tools ability** — 3tool_from_sft actually *improves* without tools (50.6% vs 41.8% in v1), suggesting I^T teaches better structured reasoning even without tool execution.
4. **3tool_from_sft v2 has highest pass@16 without tools** (82.0%) — best ceiling of any model.
5. **Best RLOO candidate: `sbfisher/tir-sft-3tool_from_sft`** — highest pass@1 with tools, strong no-tools fallback, highest ceiling.

---

## Runs 12-13: TIR RLOO — Hierarchical Reward +/- DPO Self-Critic

**Motivation:** After TIR SFT, apply RLOO with the Tool-Star hierarchical reward (eq. 3) and optionally the DPO self-critic (Algorithm 1). Two runs in parallel:
- **Phase 1**: Vanilla RLOO + hierarchical reward (no self-critic) — clean ablation
- **Phase 2**: RLOO + hierarchical reward + DPO self-critic every 5 steps

**Key changes from Frank's implementation:**
- Replaced `compute_score_with_tools` (clipped [0,1] shaping) with `compute_hierarchical_reward` (Tool-Star eq. 3: -1/0/1.0/1.1)
- Removed DSPy failure analysis / FailureDatabase complexity
- Added true DPO self-critic (eq. 5) with preference pairs from self-sampling
- Added `--self_critic` flag — when disabled, runs vanilla RLOO + hierarchical reward

### Run 12: RLOO No Self-Critic — RLOO + Hierarchical Reward
- **Date**: 2026-06-02
- **Base model**: `sbfisher/tir-sft-3tool_from_sft` (best v2 SFT, 58.0% pass@1 with tools)
- **Reward**: Hierarchical (Tool-Star eq. 3): -1 (bad format), 0 (format ok, wrong), 1.0 (correct), 1.1 (correct + multi-tool)
- **Config**: 250 steps, batch=4, group=8, grad_accum=4, lr=1e-5, entropy=0.01, kl=0.0, 3-tool (calculator,number_tracker,running_total), max_tool_turns=5
- **W&B**: `tir_rloo_project` / `rloo_nosc`
- **Modal**: https://modal.com/apps/sebastianfisher/main/ap-f8JY9gb4Hl9TfjwtGZ7Zct
- **Status**: RUNNING (step 0-1 completed, reward_mean=-0.525 at step 0)
- **Notes**: First launch OOM'd with grad_accum=1 (32 seqs in one forward pass). Fixed with grad_accum=4.

### Run 13: RLOO Self-Critic — RLOO + Hierarchical Reward + DPO Self-Critic
- **Date**: 2026-06-02
- **Base model**: `sbfisher/tir-sft-3tool_from_sft` (same as Run 12)
- **Reward**: Same hierarchical reward
- **Self-critic**: Every 5 steps, sample 8 responses, threshold=1.0, DPO beta=0.1
- **Config**: Same as Run 12 + `--self_critic --self_critic_every_k 5 --self_critic_n_samples 8 --self_critic_beta 0.1`
- **W&B**: `tir_rloo_project` / `rloo_sc`
- **Modal**: https://modal.com/apps/sebastianfisher/main/ap-QoGx68LS7vFYkS5P53pRNE
- **Status**: RUNNING (step 9, DPO self-critic triggering every 5 steps)
- **DPO Bug (FIXED in code, not yet deployed)**: DPO loss was always log(2) because the ref model was a deepcopy of the current policy (ref==policy→logits=0). Fixed to load original SFT model as ref instead. Current run effectively identical to nosc (DPO is a no-op). Will need relaunch for proper self-critic.

---

## Run 14: Difficulty Labeling — Qwen3-32B Teacher on Full Dataset

**Motivation:** Label all ~490k Countdown problems with difficulty scores to enable curriculum learning for RLOO. The teacher model solves each problem twice (without tools, with tools) and optionally assigns a 1-10 difficulty rating. This produces three potential curriculum strategies:
1. **Tool-based solvability**: Easy (solved without tools), Medium (solved only with tools), Hard (unsolved even with tools)
2. **Teacher difficulty scores**: 1-10 rating from `<difficulty>N</difficulty>` tags
3. **Number count**: 3-number = easy, 4-number = hard (trivial heuristic baseline)

- **Date**: 2026-06-08
- **Teacher model**: Qwen/Qwen3-32B (with `enable_thinking=False` via `tokenizer.apply_chat_template`)
- **Infrastructure**: 5 shards × 2×H100 (tensor parallelism (TP)=2), Modal
- **Dataset**: `Jiayi-Pan/Countdown` (train split, 490,314 problems)
- **Config**: `max_tokens=2048`, `max_model_len=4096`, `batch_size=512`, `temperature=0.0`
- **No-tools system prompt**: `"You are a helpful assistant."` + difficulty instruction (matches vanilla RLOO student prompt exactly)
- **With-tools system prompt**: `build_tool_system_prompt()` (full I^T with tool catalogue + worked example) + difficulty instruction
- **Script**: `tir_extension/curriculum/label_difficulty.py`
- **Output**: `difficulty_labels_full/shard_{0-4}.jsonl` (490,314 total lines)
- **Status**: COMPLETE

### Results

| Metric | Value |
|--------|-------|
| Total problems | 490,314 |
| No-tools correct | 210,929 (43.0%) |
| With-tools correct | 230,792 (47.1%) |
| Tools helped (solved only with tools) | 50,316 (10.3%) |
| Tools hurt (solved only without tools) | 30,453 (6.2%) |
| Difficulty parsed (no-tools) | 41,286 (8.4%) |
| Difficulty parsed (with-tools) | 77,182 (15.7%) |
| 3-number problems | 240,608 (49.1%) |
| 4-number problems | 249,706 (50.9%) |

### Strategy 1 Split (Tool-Based Solvability)

| Difficulty | Count | Fraction |
|-----------|-------|----------|
| Easy (solved w/o tools) | 210,929 | 43.0% |
| Medium (solved only w/ tools) | 50,316 | 10.3% |
| Hard (unsolved) | 229,069 | 46.7% |

### Difficulty Parsing Rate by Solvability Category

| Category | Total | No-tools parsed | With-tools parsed |
|----------|-------|----------------|-------------------|
| Easy (solved w/o tools) | 210,929 | 16.4% | 24.8% |
| Medium (solved only w/ tools) | 50,316 | 6.7% | 29.2% |
| Hard (unsolved) | 229,069 | 1.4% | 4.5% |

Hard problems almost never produce a `<difficulty>` tag — the model exhausts its 2048 token budget trying (and failing) to solve them. Strategy 2 (teacher difficulty scores) is not viable as-is.

### Solvability × Number Count Cross-Tabulation

|  | 3-number | 4-number |
|--|----------|----------|
| Easy | 165,848 (68.9% of 3-num) | 45,081 (18.1% of 4-num) |
| Medium | 22,659 (9.4%) | 27,657 (11.1%) |
| Hard | 52,101 (21.7%) | 176,968 (70.9%) |

Number count is a decent but imperfect proxy: 3-number problems are mostly easy (69%) but 22% are hard even for the teacher. 4-number problems are mostly hard (71%) but 18% are easy. The medium bucket (where tools actually help) is spread evenly across both (~10%).

### Notes
- Low difficulty parsing rates (8-16%) because the model runs out of tokens on hard problems before outputting `<difficulty>` tags. Strategy 2 (teacher difficulty scores) may not be viable as-is.
- The medium bucket is small (10.3%) — curriculum that trains medium→hard will have limited medium diversity. May need to oversample or combine with number-count heuristic.
- Tools provide a net benefit (+4.1pp accuracy), but also hurt on 6.2% of problems (likely due to incorrect tool calls or wasted token budget).

---

## Prior Data (from Mahmood's branch `tir_extension_mih`)
- `tir_trajectories.json`: 414 trajectories, all score=1.0, all with tool calls
  - Tool usage: calculator (559 calls), number_tracker (90), running_total (34)
  - Generated from 500 problems with gpt-4o-mini
