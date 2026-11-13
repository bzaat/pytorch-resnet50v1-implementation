# Superseded: A seed 43, first run (crashed at epoch 37, resumed)

This run is **not** used in the results tables. It was replaced by an uninterrupted re-run with the same seed, split and settings in [`../seed43/`](../seed43/).

The run crashed during epoch 37 and was resumed from its epoch-36 checkpoint. The resume code of that commit (`8ba4cee`) did not restore the random-number state, so the first resumed epoch replayed the data order and augmentations of epoch 1, and training after epoch 36 differs from an uninterrupted run. Details are in [`../../summary.md`](../../summary.md#a-seed-43-re-run).

Its results: test accuracy 29.38% (best-val checkpoint, epoch 48) and 29.87% (last epoch).

Files:
- `config_initial.json`: configuration of the original start.
- `config.json`: configuration of the resume.
- `train_until_crash.log`: the log up to the crash.
- `train.log`: the log including the resume.
- `resume_stdout_stderr.txt`: output of the resume.
