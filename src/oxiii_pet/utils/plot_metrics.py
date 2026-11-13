"""Plot loss and accuracy curves from a run's metrics.csv.

Usage: uv run python -m oxiii_pet.utils.plot_metrics runs/<run> -o results/<name>_curves.png
"""
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = ArgumentParser(description="Plot metrics.csv curves")
    parser.add_argument("run_dir", help="run folder containing metrics.csv")
    parser.add_argument("--output", "-o", required=True, help="output PNG path")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.run_dir, "metrics.csv"))
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    ax_loss.plot(df["epoch"], df["train_loss"], label="train (epoch mean)")
    ax_loss.plot(df["epoch"], df["val_loss"], label="test")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("cross-entropy loss"); ax_loss.set_title("Loss")
    ax_acc.plot(df["epoch"], df["train_acc"] * 100, label="train (augmented)")
    ax_acc.plot(df["epoch"], df["val_acc"] * 100, label="test")
    best = df.loc[df["val_acc"].idxmax()]
    ax_acc.scatter([best["epoch"]], [best["val_acc"] * 100], color="black", zorder=3,
                   label=f"best test {best['val_acc'] * 100:.2f}% (ep {int(best['epoch'])})")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy (%)"); ax_acc.set_title("Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(args.title or os.path.basename(os.path.normpath(args.run_dir)))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
