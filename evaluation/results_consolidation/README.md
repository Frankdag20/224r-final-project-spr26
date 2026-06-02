# Results Consolidation

Self-contained analysis of all Countdown TIR methods. Every metric is computed
directly from the real evaluation JSONs in `data/` — nothing is hand-entered.

## Contents
- `results_consolidation.ipynb` — the analysis notebook (run top-to-bottom)
- `build_nb.py` — regenerates the notebook from scratch (`python build_nb.py`)
- `data/` — evaluation rollout JSONs (16 samples/prompt) from `countdown_eval.py`
- `figures/` — generated plots (regenerated on notebook execution)

## Reproduce
```bash
cd evaluation/results_consolidation
python build_nb.py                       # writes the notebook
jupyter nbconvert --to notebook --execute --inplace results_consolidation.ipynb
```

## Headline results (multi-turn Pass@1, Countdown test split)

| Method | Pass@1 | vs SFT | Tool-use |
|---|---|---|---|
| TIR SFT (baseline) | 51.2% | — | 65% |
| IPO (tool-contrastive) | 50.4% | -0.8 | 64% |
| ReST round 1 | 55.8% | +4.5 | 69% |
| **ReST round 2 (ReST-EM)** | **57.0%** | **+5.8** | 68% |
| TIR RLOO (best ckpt) | 50.4% | -0.8 | 66% |
| RLOO + Self-Critic (best ckpt) | 52.6% | +1.4 | 67% |
| RLOO + SC + Curriculum (best ckpt) | 50.2% | -1.0 | 66% |

All methods build on the same SFT warm-start (`tir_sft_run2`, trained on
`tir_trajectories_2000.json`).

## Findings
1. **Multi-turn tool execution** is the dominant inference lever (~+10pp over single-turn).
2. **IPO** plateaus at the SFT ceiling — offline preference learning cannot exceed its generator.
3. **ReST-EM** breaks the ceiling (+5.8pp) by distilling pass@k capability into pass@1.
4. **Online RL** is competitive at its best checkpoint (self-critic beats SFT) but over-trains/collapses on long runs.
5. **Tool selection works** — the 3 distractor tools are essentially never invoked.

## Data file → method map
| File | Method / regime |
|---|---|
| `tir_sft_eval.json` | SFT, single-turn |
| `tir_sft_eval_multiturn.json` | SFT, multi-turn |
| `ipo_v3_eval_multiturn.json` | IPO tool-contrastive |
| `rest_v3_eval_multiturn.json` | ReST round 1 |
| `rest_v4_eval_multiturn.json` | ReST round 2 (ReST-EM) |
| `rl_nosc_*.json` | TIR RLOO (no self-critic) |
| `rl_sc_*.json` | RLOO + self-critic |
| `rl_sc_curr_*.json` | RLOO + self-critic + curriculum |

(`*_s45` / `*_v2` suffixes = best earlier checkpoint; the notebook auto-selects the higher-scoring one.)
