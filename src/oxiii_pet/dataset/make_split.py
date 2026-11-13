"""Create the fixed, stratified train/validation split of the official trainval.txt.

The split depends only on --split_seed (default 0), not on the training seed, so every
training run (and both the 37-breed and the cat-vs-dog task) uses the same images.
The official test.txt split is not touched here.

Usage: uv run python -m oxiii_pet.dataset.make_split
"""
import json
import os
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = ArgumentParser(description="Stratified train/val split of trainval.txt")
    parser.add_argument("--dir", "-d", type=str, default=os.getenv("DATA_DIR", "data"), help="dataset root")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="default splits/trainval_<train>_<val>_seed<split_seed>.json")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.dir, "annotations", "trainval.txt"), sep=r"\s+", header=None)
    names, class_ids = df[0].astype(str).tolist(), df[1].astype(int).tolist()
    train_names, val_names = train_test_split(
        names, test_size=args.val_fraction, stratify=class_ids, random_state=args.split_seed, shuffle=True)

    pct_val = round(args.val_fraction * 100)
    output = args.output or os.path.join("splits", f"trainval_{100 - pct_val}_{pct_val}_seed{args.split_seed}.json")
    split = {
        "source": "annotations/trainval.txt",
        "method": "sklearn.model_selection.train_test_split, stratified by breed class_id",
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "n_train": len(train_names),
        "n_val": len(val_names),
        "train": sorted(train_names),
        "val": sorted(val_names),
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(split, f, indent=1)
    print(f"Saved {output}: {len(train_names)} train / {len(val_names)} val")


if __name__ == "__main__":
    main()
