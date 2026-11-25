# Hyperparameter tuning (validation split only, seed 42)

Produced by `oxiii_pet.utils.tune` via `scripts/tune_and_rerun.sh`. The test split is never loaded (`--no_test`).

| Exp | Trials done | Best trial | Best val acc | Baseline (t00) val acc |
|-----|-------------|------------|--------------|------------------------|
| B (transfer, 37 cls, 10 ep) | 18 / 18 | `t01` SGD lr=1e-2 wd=1e-4 ls=0.1 | 0.9620 | 0.9552 |
| A (scratch, 37 cls, 20 ep)  | **8 / 12** | `t07` AdamW lr=1e-3 wd=5e-2 | 0.3546 | 0.2024 |

**Status (2026-11-25): in progress.** The A search was interrupted during trial `t08`, so
`A/best.json` only covers `t00`-`t07` and does not yet include the four trials added later in the grid
(`t08`-`t11`). The final A/B/C re-runs with the tuned hyperparameters (seeds 42-44) have not been run yet.
Re-run `scripts/tune_and_rerun.sh` to resume: finished trials are skipped.

C (cat vs dog) reuses B's hyperparameters (its validation accuracy is saturated at about 99.9%).
