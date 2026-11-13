# Superseded: single-seed runs with checkpoint selection on the test split

> **Do not cite these numbers.** They are kept only for reference. The current results are in
> [`../summary.md`](../summary.md).

These are the runs of 2026-11-13 (seed 42, commit `0cfedcf`). In those runs the training scripts
evaluated on the official `test.txt` split after every epoch and saved `best_model.pth` by that test
accuracy. There was no validation split, so the "best test accuracy" was selected on the test set and
is optimistically biased. The final-epoch numbers didn't involve selection, but every run had a single
seed only.

In these files, "val" / "Validation" means the official **test** split.

The current protocol holds out a stratified 20% of `trainval.txt` as a validation split
(`splits/trainval_80_20_seed0.json`), selects checkpoints on it, evaluates on the test split once per
run and repeats every experiment with 3 seeds.
