"""Shared helpers for training runs: seeding, run folders, logging, metrics, class names."""
import csv
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm


def set_seed(seed):
    """Seed python, numpy and torch. Returns (generator, worker_init_fn) for DataLoaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return generator, worker_init_fn


def default_run_dir(tag):
    return os.path.join("runs", f"{datetime.now():%Y-%m-%d_%H%M}_{tag}")


def _git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def setup_run(run_dir, args, experiment):
    """Create the run folder, a logger writing to train.log and the console, and config.json."""
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("oxiii_pet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(run_dir / "train.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    status = _git("status", "--porcelain", "--untracked-files=no")
    config = {
        "experiment": experiment,
        "args": vars(args),
        "seed": getattr(args, "seed", None),
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "deterministic_note": "Seeded, but cuDNN kernels are not forced deterministic; GPU runs are not bit-exact.",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Run dir: %s", run_dir)
    logger.info("Config: %s", json.dumps(config))
    return logger, config


class MetricsCSV:
    FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s", "is_best"]

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def append(self, **row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)


def get_class_names(num_class, root):
    """Class names in label-index order, read from the dataset annotations."""
    if num_class == 2:
        return ["cat", "dog"]  # species 1 = Cat, 2 = Dog
    df = pd.read_csv(os.path.join(root, "annotations", "trainval.txt"), sep=r"\s+", header=None)
    names = {}
    for image_name, class_id in zip(df[0], df[1]):
        names[int(class_id)] = str(image_name).rsplit("_", 1)[0]
    return [names[i] for i in sorted(names)]


def plot_confusion_matrix(writer, cm, class_names, epoch, tag="Validation/confusion_matrix",
                          title="Confusion matrix (validation split)"):
    """Log a row-normalized confusion matrix figure to TensorBoard."""
    figure = confusion_matrix_figure(cm, class_names, title)
    writer.add_figure(tag, figure, epoch)
    plt.close(figure)


def save_confusion_matrix_png(cm, class_names, path, title):
    figure = confusion_matrix_figure(cm, class_names, title)
    figure.savefig(path, dpi=60 if len(class_names) > 10 else 100)
    plt.close(figure)


def confusion_matrix_figure(cm, class_names, title="Confusion matrix"):
    """Row-normalized confusion matrix: rows = true class, columns = predicted class."""
    size = 20 if len(class_names) > 10 else 6
    figure = plt.figure(figsize=(size, size))
    # Colors and numbers both show the row-normalized value (fraction of the true class).
    cm = np.around(cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1), decimals=2)
    plt.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title(title, fontsize=16 if size == 20 else 10)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > 0.5 else "black"
            plt.text(j, i, f"{cm[i, j]:.2f}", horizontalalignment="center", verticalalignment="center",
                     color=color)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return figure


def final_test_evaluation(model, test_loader, criterion, device, ckpt_dir, run_dir, num_class,
                          class_names, logger, extra=None):
    """Evaluate best_model.pth (best validation accuracy) and last_model.pth on the test split,
    exactly once each, and write summary.json. This is the only place the test split is used."""
    from sklearn.metrics import confusion_matrix

    results = {}
    for name in ("best_model", "last_model"):
        ckpt = torch.load(os.path.join(ckpt_dir, f"{name}.pth"), map_location=device)
        model.load_state_dict(ckpt["model"])
        loss, acc, preds, labels = run_epoch(model, test_loader, criterion, device, desc=f"Test  {name}")
        results[name] = {"epoch": ckpt["epoch"], "test_acc": acc, "test_loss": loss}
        logger.info("Test (%s, epoch %d): loss %.4f acc %.4f", name, ckpt["epoch"], loss, acc)
        if name == "best_model":
            cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(num_class)))
            save_confusion_matrix_png(
                cm, class_names, os.path.join(run_dir, "test_confusion_matrix_best_val.png"),
                f"Test split, checkpoint with best validation accuracy (epoch {ckpt['epoch']})")

    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    best_epoch, last_epoch = results["best_model"]["epoch"], results["last_model"]["epoch"]
    row_best = df[df["epoch"] == best_epoch].iloc[-1]
    row_last = df[df["epoch"] == last_epoch].iloc[-1]
    summary = {
        "val_best_epoch": int(best_epoch),
        "val_best_acc": float(row_best["val_acc"]),
        "val_best_loss": float(row_best["val_loss"]),
        "test_acc_at_best_val": results["best_model"]["test_acc"],
        "test_loss_at_best_val": results["best_model"]["test_loss"],
        "last_epoch": int(last_epoch),
        "val_acc_at_last": float(row_last["val_acc"]),
        "test_acc_at_last": results["last_model"]["test_acc"],
        "test_loss_at_last": results["last_model"]["test_loss"],
        "train_acc_at_best_val": float(row_best["train_acc"]),
        "train_acc_at_last": float(row_last["train_acc"]),
        "train_time_s": round(float(df["epoch_time_s"].sum()), 1),
        "n_train": None, "n_val": None, "n_test": len(test_loader.dataset),
        "note": ("Checkpoint selected on the validation split (20% of trainval.txt). The official test split "
                 "is evaluated once per checkpoint after training. train_acc is the running accuracy on "
                 "augmented training batches in train mode. train_time_s sums the per-epoch times "
                 "(training + validation) and excludes the final test evaluation."),
    }
    summary.update(extra or {})
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary))
    return summary


def run_epoch(model, loader, criterion, device, optimizer=None, desc="", clip=0.0,
              writer=None, global_step=0):
    """One pass over loader. Trains if optimizer is given, else evaluates.
    Returns (mean_loss, accuracy, all_predictions, all_labels)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    all_preds, all_labels = [], []
    lr = optimizer.param_groups[0]["lr"] if training else None
    bar = tqdm(loader, desc=desc, colour="green" if training else "cyan", leave=False, dynamic_ncols=True)
    with torch.set_grad_enabled(training):
        for i, (images, labels) in enumerate(bar):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip and clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                if writer is not None:
                    writer.add_scalar("Train/Loss", loss.item(), global_step + i)
            preds = output.argmax(dim=1)
            n = labels.size(0)
            total_loss += loss.item() * n
            correct += (preds == labels).sum().item()
            seen += n
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            postfix = {"loss": f"{total_loss / seen:.4f}", "acc": f"{correct / seen:.4f}"}
            if lr is not None:
                postfix["lr"] = f"{lr:.2e}"
            bar.set_postfix(postfix)
    return total_loss / seen, correct / seen, torch.cat(all_preds), torch.cat(all_labels)
