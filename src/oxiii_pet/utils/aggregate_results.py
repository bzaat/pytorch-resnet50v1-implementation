"""Collect the multi-seed runs into results/: copy small artifacts, plot curves, write tables.

For each experiment and seed it reads runs/<date>_<exp>_seed<k>/ and
  - copies config.json, metrics.csv, summary.json, train.log and test_confusion_matrix_best_val.png
    to results/<exp>/seed<k>/ (no checkpoints, no TensorBoard files),
  - plots train vs validation curves per experiment (mean +/- 1 sample std over seeds),
  - plots validation accuracy of A vs B,
  - writes results/aggregate.json and results/tables.md (per-seed and aggregate tables).
Missing or failed runs are listed, never silently dropped.

Usage: uv run python -m oxiii_pet.utils.aggregate_results --date 2026-11-13
"""
import json
import os
import shutil
import statistics
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oxiii_pet.utils.plot_metrics import load_runs, plot_band

EXPERIMENTS = {
    "A_scratch_37cls": "A. ResNet-50 from scratch, 37 breeds, 50 epochs",
    "B_transfer_37cls": "B. Pretrained ResNet-50 (IMAGENET1K_V2), 37 breeds, 10 epochs",
    "C_transfer_2cls": "C. Pretrained ResNet-50 (IMAGENET1K_V2), cat vs dog, 10 epochs",
}
ARTIFACTS = ["config.json", "metrics.csv", "summary.json", "train.log", "test_confusion_matrix_best_val.png"]


def pct(x):
    return f"{x * 100:.2f}"


def mean_std(values, scale=100.0):
    """'mean ± sample std' (n-1) with two decimals; std is 'n/a' for fewer than 2 values."""
    values = [v * scale for v in values]
    if not values:
        return "n/a"
    std = f"{statistics.stdev(values):.2f}" if len(values) > 1 else "n/a"
    return f"{statistics.mean(values):.2f} ± {std}"


def main():
    parser = ArgumentParser(description="Aggregate multi-seed runs into results/")
    parser.add_argument("--date", required=True, help="date prefix of the run folders, e.g. 2026-11-13")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    per_seed, missing, run_dirs = {}, [], {}
    for exp in EXPERIMENTS:
        per_seed[exp], run_dirs[exp] = {}, []
        for seed in seeds:
            run_dir = os.path.join(args.runs, f"{args.date}_{exp}_seed{seed}")
            summary_path = os.path.join(run_dir, "summary.json")
            if not os.path.exists(summary_path):
                missing.append(f"{exp} seed {seed}: no summary.json in {run_dir} (run failed or incomplete)")
                continue
            with open(summary_path) as f:
                per_seed[exp][seed] = json.load(f)
            run_dirs[exp].append(run_dir)
            out_dir = os.path.join(args.results, exp, f"seed{seed}")
            os.makedirs(out_dir, exist_ok=True)
            for name in ARTIFACTS:
                src = os.path.join(run_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(out_dir, name))
                else:
                    missing.append(f"{exp} seed {seed}: {name} missing")

    # Curves per experiment (mean +/- std over seeds)
    for exp, dirs in run_dirs.items():
        if not dirs:
            continue
        df = load_runs(dirs)
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
        plot_band(ax_loss, df, "train_loss", "train (augmented)")
        plot_band(ax_loss, df, "val_loss", "validation")
        plot_band(ax_acc, df, "train_acc", "train (augmented)", scale=100)
        plot_band(ax_acc, df, "val_acc", "validation", scale=100)
        ax_loss.set_title("Loss"); ax_loss.set_ylabel("cross-entropy loss")
        losses = df[["train_loss", "val_loss"]]
        if losses.max().max() / max(losses.min().min(), 1e-6) > 50:
            ax_loss.set_yscale("log"); ax_loss.set_ylabel("cross-entropy loss (log scale)")
        ax_acc.set_title("Accuracy"); ax_acc.set_ylabel("accuracy (%)")
        for ax in (ax_loss, ax_acc):
            ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
        fig.suptitle(f"{EXPERIMENTS[exp]}: seeds {', '.join(str(s) for s in per_seed[exp])} "
                     f"(line = mean, band = ±1 std)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, exp, "curves.png"))
        plt.close(fig)

    # A vs B on validation accuracy
    if run_dirs["A_scratch_37cls"] and run_dirs["B_transfer_37cls"]:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
        plot_band(ax, load_runs(run_dirs["A_scratch_37cls"]), "val_acc", "A: from scratch (50 epochs)", scale=100)
        plot_band(ax, load_runs(run_dirs["B_transfer_37cls"]), "val_acc", "B: ImageNet-pretrained (10 epochs)",
                  scale=100)
        ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy (%)"); ax.set_ylim(0, 100)
        ax.set_title("37 breeds: validation accuracy, mean ± 1 std over seeds")
        ax.grid(alpha=0.3); ax.legend(loc="center right")
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, "A_vs_B_val_accuracy.png"))
        plt.close(fig)

    # Tables
    lines = ["## Per-seed results", "",
             "| Experiment | Seed | Val best epoch / last | Val acc at best (%) | Test acc at best-val (%) | Test loss at best-val "
             "| Test acc at last epoch (%) | Test loss at last | Train acc at last (augmented, %) | Train time (min) |",
             "|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    aggregate = {}
    for exp, runs in per_seed.items():
        for seed, s in runs.items():
            lines.append(
                f"| {exp[0]} | {seed} | {s['val_best_epoch']} / {s['last_epoch']} | {pct(s['val_best_acc'])} | "
                f"{pct(s['test_acc_at_best_val'])} | {s['test_loss_at_best_val']:.4f} | {pct(s['test_acc_at_last'])} | "
                f"{s['test_loss_at_last']:.4f} | {pct(s['train_acc_at_last'])} | {s['train_time_s'] / 60:.2f} |")
        vals = list(runs.values())
        aggregate[exp] = {
            "n_seeds": len(vals), "seeds": list(runs),
            "test_acc_at_best_val": [v["test_acc_at_best_val"] for v in vals],
            "test_acc_at_last": [v["test_acc_at_last"] for v in vals],
            "val_best_acc": [v["val_best_acc"] for v in vals],
            "val_best_epoch": [v["val_best_epoch"] for v in vals],
            "train_acc_at_last": [v["train_acc_at_last"] for v in vals],
            "train_acc_at_best_val": [v["train_acc_at_best_val"] for v in vals],
            "train_time_min": [v["train_time_s"] / 60 for v in vals],
        }
    lines += ["", "## Aggregate (mean ± sample std, n = number of seeds)", "",
              "| Experiment | n | Test acc, best-val checkpoint (%) | Test acc, last epoch (%) | Val acc at best (%) "
              "| Train acc at last (augmented, %) | Train time (min) |",
              "|---|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for exp, a in aggregate.items():
        lines.append(
            f"| {EXPERIMENTS[exp]} | {a['n_seeds']} | {mean_std(a['test_acc_at_best_val'])} | "
            f"{mean_std(a['test_acc_at_last'])} | {mean_std(a['val_best_acc'])} | "
            f"{mean_std(a['train_acc_at_last'])} | {mean_std(a['train_time_min'], scale=1.0)} |")
    if missing:
        lines += ["", "## Missing or failed", ""] + [f"- {m}" for m in missing]

    with open(os.path.join(args.results, "aggregate.json"), "w") as f:
        json.dump({"date": args.date, "seeds": seeds, "aggregate": aggregate, "missing": missing}, f, indent=2)
    with open(os.path.join(args.results, "tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
