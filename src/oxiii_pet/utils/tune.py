"""Hyperparameter search on the validation split (the test split is never used here).

Each trial is a normal training run started with --no_test, so it only sees train + validation.
Trials are ranked by best validation accuracy; ties keep the earlier trial in the grid (each grid
lists the original, untuned config first).

Outputs, per experiment, in results/tuning/<exp>/:
  trials.csv      one row per trial (hyperparameters, best val acc, epoch, time)
  best.json       the winning hyperparameters (read by scripts/tune_and_rerun.sh)
  <trial>/        metrics.csv, config.json, summary.json, train.log of every trial

Usage:
  uv run python -m oxiii_pet.utils.tune --exp B            # run the B grid
  uv run python -m oxiii_pet.utils.tune --exp A --dry_run  # print the trials only
  uv run python -m oxiii_pet.utils.tune --best_args B      # print the CLI flags of B's best trial
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser

# Fixed (not tuned) arguments per experiment. A is tuned on a 20-epoch budget to keep the search
# affordable on one laptop GPU; the final runs use the full 50 epochs.
BASE = {
    "A": {"module": "oxiii_pet.model.train", "args": ["-e", "20", "-b", "32", "-n", "37"]},
    "B": {"module": "oxiii_pet.transfer_learning.train", "args": ["-e", "10", "-b", "32", "-n", "37", "-v", "2"]},
}
# Search spaces. The first entry of each grid is the original (untuned) configuration.
GRIDS = {
    "A": [
        {"optimizer": "sgd", "lr": 1e-2, "weight_decay": 1e-4, "label_smoothing": 0.0},
        {"optimizer": "sgd", "lr": 1e-2, "weight_decay": 5e-4, "label_smoothing": 0.0},
        {"optimizer": "sgd", "lr": 5e-2, "weight_decay": 1e-4, "label_smoothing": 0.0},
        {"optimizer": "sgd", "lr": 5e-2, "weight_decay": 5e-4, "label_smoothing": 0.0},
        {"optimizer": "sgd", "lr": 1e-1, "weight_decay": 1e-4, "label_smoothing": 0.0},
        {"optimizer": "sgd", "lr": 1e-1, "weight_decay": 5e-4, "label_smoothing": 0.0},
        {"optimizer": "sgd", "lr": 5e-2, "weight_decay": 5e-4, "label_smoothing": 0.1},
        {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 5e-2, "label_smoothing": 0.0},
        # Added after trials t00-t02 showed that lr above 1e-2 hurts: also search below it.
        {"optimizer": "sgd", "lr": 5e-3, "weight_decay": 1e-4, "label_smoothing": 0.0},
        {"optimizer": "sgd", "lr": 1e-2, "weight_decay": 1e-4, "label_smoothing": 0.1},
        # Added after t07 (AdamW) clearly beat every SGD trial: check a lower AdamW lr and label smoothing.
        {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 5e-2, "label_smoothing": 0.0},
        {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 5e-2, "label_smoothing": 0.1},
    ],
    "B": [
        {"optimizer": opt, "lr": lr, "weight_decay": wd, "label_smoothing": ls}
        for opt, lr, wd in [("sgd", 1e-2, 1e-4), ("sgd", 3e-3, 1e-4), ("sgd", 3e-2, 1e-4),
                            ("sgd", 1e-2, 1e-3), ("sgd", 3e-3, 1e-3), ("sgd", 3e-2, 1e-3),
                            ("adamw", 1e-4, 1e-2), ("adamw", 3e-4, 1e-2), ("adamw", 1e-3, 1e-2)]
        for ls in (0.0, 0.1)
    ],
}
HP_KEYS = ["optimizer", "lr", "weight_decay", "label_smoothing"]


def trial_name(i, hp):
    return f"t{i:02d}_{hp['optimizer']}_lr{hp['lr']:g}_wd{hp['weight_decay']:g}_ls{hp['label_smoothing']:g}"


def to_flags(hp):
    return [f"--{k}={hp[k]}" for k in HP_KEYS]


def main():
    parser = ArgumentParser(description="Grid search on the validation split")
    parser.add_argument("--exp", choices=sorted(GRIDS))
    parser.add_argument("--best_args", choices=sorted(GRIDS), help="print the best trial's flags and exit")
    parser.add_argument("--data", "-d", default=os.getenv("DATA_DIR", "data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_file", default="splits/trainval_80_20_seed0.json")
    parser.add_argument("--runs", default="runs/tuning")
    parser.add_argument("--results", default="results/tuning")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.best_args:
        with open(os.path.join(args.results, args.best_args, "best.json")) as f:
            print(" ".join(to_flags(json.load(f)["hparams"])))
        return
    if not args.exp:
        parser.error("--exp or --best_args is required")

    exp, grid = args.exp, GRIDS[args.exp]
    out_dir = os.path.join(args.results, exp)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, hp in enumerate(grid):
        name = trial_name(i, hp)
        run_dir = os.path.join(args.runs, exp, name)
        cmd = [sys.executable, "-m", BASE[exp]["module"], *BASE[exp]["args"], "-d", args.data,
               "--seed", str(args.seed), "--split_file", args.split_file, "--run_dir", run_dir,
               "--no_test", *to_flags(hp)]
        print(f"[{exp} {i + 1}/{len(grid)}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):  # finished trials are skipped, so the search can be resumed
            shutil.rmtree(run_dir, ignore_errors=True)
            os.makedirs(run_dir)
            start = time.time()
            with open(os.path.join(run_dir, "stdout_stderr.txt"), "w") as log:
                code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(f"    exit={code} in {(time.time() - start) / 60:.1f} min", flush=True)
        row = {"trial": name, **hp, "status": "failed", "val_best_acc": "", "val_best_epoch": "",
               "val_acc_at_last": "", "train_time_min": ""}
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                s = json.load(f)
            row.update(status="ok", val_best_acc=s["val_best_acc"], val_best_epoch=s["val_best_epoch"],
                       val_acc_at_last=s["val_acc_at_last"], train_time_min=round(s["train_time_s"] / 60, 2))
            trial_out = os.path.join(out_dir, name)
            os.makedirs(trial_out, exist_ok=True)
            for fname in ("metrics.csv", "config.json", "summary.json", "train.log"):
                if os.path.exists(os.path.join(run_dir, fname)):
                    shutil.copy2(os.path.join(run_dir, fname), trial_out)
        print(f"    {row['status']} val_best_acc={row['val_best_acc']}", flush=True)
        rows.append(row)
    if args.dry_run:
        return

    with open(os.path.join(out_dir, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        sys.exit(f"all {exp} trials failed")
    best = max(ok, key=lambda r: r["val_best_acc"])  # max() keeps the first of equal values
    result = {"exp": exp, "selection": "highest best-epoch validation accuracy, seed "
              f"{args.seed}; ties keep the earlier trial", "trial": best["trial"],
              "val_best_acc": best["val_best_acc"], "hparams": {k: best[k] for k in HP_KEYS},
              "baseline_trial": rows[0]["trial"], "baseline_val_best_acc": rows[0]["val_best_acc"],
              "fixed_args": BASE[exp]["args"]}
    with open(os.path.join(out_dir, "best.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
