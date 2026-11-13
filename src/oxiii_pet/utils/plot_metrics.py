"""Plot train vs validation loss and accuracy curves from one or more runs' metrics.csv.

One run: plain curves. Several runs (e.g. seeds): mean curve with a +/- 1 sample-std band.

Usage:
  uv run python -m oxiii_pet.utils.plot_metrics runs/<run> -o results/<name>_curves.png
  uv run python -m oxiii_pet.utils.plot_metrics runs/<exp>_seed42 runs/<exp>_seed43 runs/<exp>_seed44 -o results/<exp>/curves.png
"""
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_runs(run_dirs):
    """Stack metrics.csv of several runs; keeps the last row per epoch (a resumed run may repeat one)."""
    frames = []
    for run_dir in run_dirs:
        df = pd.read_csv(os.path.join(run_dir, "metrics.csv")).drop_duplicates("epoch", keep="last")
        df["run"] = os.path.basename(os.path.normpath(run_dir))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_band(ax, df, column, label, scale=1.0, color=None):
    """Mean over runs per epoch, with a +/- 1 sample-std band when there is more than one run."""
    grouped = df.groupby("epoch")[column]
    mean, std, n = grouped.mean() * scale, grouped.std(ddof=1) * scale, grouped.count()
    n_runs = int(n.max())
    line, = ax.plot(mean.index, mean.values, label=label + (f" (mean of {n_runs})" if n_runs > 1 else ""),
                    color=color)
    if n_runs > 1:
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values, alpha=0.2, color=line.get_color())
    return mean


def main():
    parser = ArgumentParser(description="Plot metrics.csv curves (train vs validation)")
    parser.add_argument("run_dirs", nargs="+", help="run folder(s) containing metrics.csv")
    parser.add_argument("--output", "-o", required=True, help="output PNG path")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    df = load_runs(args.run_dirs)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    plot_band(ax_loss, df, "train_loss", "train (epoch mean, augmented)")
    plot_band(ax_loss, df, "val_loss", "validation")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("cross-entropy loss"); ax_loss.set_title("Loss")
    plot_band(ax_acc, df, "train_acc", "train (augmented)", scale=100)
    plot_band(ax_acc, df, "val_acc", "validation", scale=100)
    if len(args.run_dirs) == 1:
        best = df.loc[df["val_acc"].idxmax()]  # idxmax returns the first maximum = earliest epoch on ties
        ax_acc.scatter([best["epoch"]], [best["val_acc"] * 100], color="black", zorder=3,
                       label=f"best val {best['val_acc'] * 100:.2f}% (ep {int(best['epoch'])})")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy (%)"); ax_acc.set_title("Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    band = " (band = ±1 std over runs)" if len(args.run_dirs) > 1 else ""
    fig.suptitle((args.title or os.path.basename(os.path.normpath(args.run_dirs[0]))) + band)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
