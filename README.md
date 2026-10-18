<h1 align="center">ResNet-50 in PyTorch</h1>

<p align="center">
  A from-scratch implementation of ResNet-50 for fine-grained pet breed classification on the Oxford-IIIT Pet dataset.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="License" src="https://img.shields.io/github/license/bzaat/pytorch-resnet50v1-implementation">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Roadmap](#roadmap)
- [License](#license)
- [References](#references)

## Overview

This project implements the ResNet-50 architecture from scratch in PyTorch and trains it on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (37 cat and dog breeds). It is a learning-oriented codebase focused on three things: understanding CNN architectures, building a clean data pipeline, and setting up a reproducible training workflow.

## Features

- **ResNet-50 from scratch:** bottleneck residual blocks, projection shortcuts, and a configurable classification head, with no dependency on `torchvision.models`.
- **Custom `Dataset`:** reads the official annotation files, builds image paths, converts labels to zero-based indices, and loads images as RGB with Pillow.
- **Train / evaluation splits:** `trainval.txt` for training and `test.txt` for evaluation.
- **Two task modes:** 37-breed classification (default) or binary cat-vs-dog classification, selected with `--numclass`.
- **Training script with CLI:** configurable epochs, batch size, data directory and log directory.
- **Experiment tracking:** loss and accuracy curves, plus a per-epoch confusion matrix, logged to TensorBoard.
- **Checkpointing and resume:** the latest and the best checkpoints (model, optimizer and scheduler state) are saved automatically, and training can be resumed from either.
- **CPU and GPU support:** training and evaluation run on CUDA when available and fall back to the CPU otherwise.
- **Installable package:** `pip install -e .` makes `oxiii_pet` importable from anywhere.

## Project Structure

```text
.
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── src/
    └── oxiii_pet/
        ├── dataset/
        │   ├── dataset_oxiiit_pet.py     # OxfordData: custom Dataset
        │   └── dataloader_oxiii_pet.py   # DataLoader smoke test
        └── model/
            ├── models.py                 # Bottleneck block and Resnet50
            └── train.py                  # Training and evaluation entry point
```

## Model Architecture

The network follows the standard ResNet-50 layout: a convolutional stem, four stages of bottleneck blocks (3-4-6-3), and a global-pooling classification head. Shapes below assume a 224 × 224 input.

| Stage | Block (1×1 → 3×3 → 1×1) | Repeats | Output channels | Output size |
|-------|-------------------------|:-------:|:---------------:|:-----------:|
| `stem` | 7×7 conv (stride 2) + 3×3 max-pool (stride 2) | 1 | 64 | 56 × 56 |
| `conv2_x` | 64 → 64 → 256 | 3 | 256 | 56 × 56 |
| `conv3_x` | 128 → 128 → 512 | 4 | 512 | 28 × 28 |
| `conv4_x` | 256 → 256 → 1024 | 6 | 1024 | 14 × 14 |
| `conv5_x` | 512 → 512 → 2048 | 3 | 2048 | 7 × 7 |
| `fc` | AdaptiveAvgPool → Flatten → Dropout(0.5) → Linear | 1 | 37 | – |

Each bottleneck block uses a residual (identity) connection. A 1×1 projection shortcut is applied whenever the spatial size or channel count changes. Downsampling is done by the stride of the 3×3 convolution.

## Dataset

The project uses the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/): roughly 7,400 images of 37 cat and dog breeds.

Each line of an annotation file has the format:

```text
<image_name> <class_id> <species> <breed_id>
```

Expected directory structure (the dataset root defaults to `data/`):

```text
data/
├── annotations/
│   ├── trainval.txt
│   └── test.txt
└── images/
    ├── Abyssinian_1.jpg
    ├── Abyssinian_2.jpg
    └── ...
```

### Data pipeline

The `OxfordData` class (`src/oxiii_pet/dataset/dataset_oxiiit_pet.py`) inherits from `torch.utils.data.Dataset`:

1. **Load annotations:** reads `trainval.txt` when `train=True`, or `test.txt` when `train=False`.
2. **Select the label column:** keeps the image name plus one label column, chosen by `num_class`: the breed class ID for 37 classes, or the species (cat/dog) for 2 classes. Any other value is rejected.
3. **Build paths:** joins `root/images/<image_name>.jpg`.
4. **Map labels:** converts 1-based class IDs to 0-based indices (`label - 1`).
5. **Load images:** opens each image with Pillow and converts it to RGB.
6. **Transform:** applies the optional `transform`.
7. **Return samples:** each item is an `(image, label)` pair.

## Getting Started

### Prerequisites

- Python 3.10 or newer
- (Optional) An NVIDIA GPU with CUDA for faster training

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/bzaat/pytorch-resnet50v1-implementation.git
cd pytorch-resnet50v1-implementation

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the project in editable mode
pip install -e .
```

> **GPU users:** if you want a specific CUDA build of PyTorch, install `torch` and `torchvision` first using the command from the [official selector](https://pytorch.org/get-started/locally/), then run the steps above.

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Download the dataset

```bash
mkdir -p data && cd data

curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
cd ..
```

By default the code looks for the dataset in `data/`. To use another location, set the `DATA_DIR` environment variable or pass `--dir` to the training script:

```bash
export DATA_DIR=/path/to/oxford-iiit-pet     # Windows (PowerShell): $env:DATA_DIR = "C:\path\to\oxford-iiit-pet"
```

## Usage

### 1. Check the data pipeline

Run the DataLoader smoke test to confirm the dataset is found and batches have the expected shape:

```bash
python src/oxiii_pet/dataset/dataloader_oxiii_pet.py
```

Or use the dataset directly in Python. The first argument of `OxfordData` is the number of classes: `37` for breeds or `2` for cat vs dog:

```python
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from oxiii_pet.dataset.dataset_oxiiit_pet import OxfordData

transform = Compose([Resize((224, 224)), ToImage(), ToDtype(torch.float32, scale=True)])

train_set = OxfordData(37, "data", train=True, transform=transform)
loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, drop_last=True)

images, labels = next(iter(loader))
print(images.shape)   # torch.Size([16, 3, 224, 224])
print(labels.shape)   # torch.Size([16])
```

### 2. Train the model

```bash
python -m oxiii_pet.model.train --epochs 50 --batch_size 32 --dir data
```

| Argument | Short | Default | Description |
|----------|:-----:|---------|-------------|
| `--epochs` | `-e` | `10` | Number of training epochs |
| `--batch_size` | `-b` | `8` | Batch size for both splits |
| `--dir` | `-d` | `$DATA_DIR` or `data` | Dataset root directory |
| `--logging` | `-l` | `tensorboard` | Directory for TensorBoard logs |
| `--trained_model` | `-t` | `trained_model` | Directory where checkpoints are saved |
| `--checkpoint` | `-c` | none | Checkpoint to resume training from |
| `--numclass` | `-n` | `37` | Number of classes: `37` for breeds, `2` for cat vs dog |

After every epoch the model is evaluated on the test split. Two checkpoints are written to the `--trained_model` directory: `last_model.pth` after every epoch, and `best_model.pth` whenever test accuracy improves. Each checkpoint stores the epoch, model, optimizer and scheduler state, and the best accuracy.

To resume an interrupted run, pass the checkpoint and keep the same `--epochs` value so the learning-rate schedule stays consistent:

```bash
python -m oxiii_pet.model.train --epochs 50 --batch_size 32 --checkpoint trained_model/last_model.pth
```

To train the binary cat-vs-dog task instead of the 37 breeds, pass `--numclass 2`:

```bash
python -m oxiii_pet.model.train --epochs 50 --batch_size 32 --numclass 2
```

### 3. Monitor training

```bash
tensorboard --logdir tensorboard
```

Logged scalars: `Train/Loss`, `Validation/Accuracy`.

Logged images: `confusion_matrix`, the confusion matrix of the test split, written once per epoch by `plot_confusion_matrix` in `train.py` (scikit-learn computes the matrix and matplotlib draws it). Open the **Images** tab and drag the slider to see how the errors change during training. Rows are the true classes and columns are the predicted classes; each cell shows the fraction of that true class predicted as the column class, so the diagonal is the per-class recall.

### 4. Load a trained checkpoint

```python
import torch
from oxiii_pet.model.models import Resnet50

model = Resnet50(num_classes=37)  # use 2 if you trained with --numclass 2
checkpoint = torch.load("trained_model/best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
```

## Training Configuration

| Component | Setting |
|-----------|---------|
| Loss | `CrossEntropyLoss` |
| Optimizer | SGD, learning rate `1e-2`, momentum `0.9`, weight decay `1e-4` |
| LR schedule | `CosineAnnealingLR` over all epochs (`T_max` = number of epochs), stepped once per epoch |
| Weight init | Kaiming normal (convolutions), BatchNorm weight 1 / bias 0, Linear `N(0, 0.01)` |
| Train transforms | `RandomResizedCrop(224, scale=(0.6, 1.0))`, `RandomHorizontalFlip`, ImageNet mean/std normalization |
| Test transforms | `Resize((224, 224))`, ImageNet mean/std normalization |
| DataLoader | 4 workers, `drop_last=True` for training |
| Evaluation | Test split, once per epoch |
| Output classes | 37 breeds by default, or 2 (cat vs dog) with `--numclass 2` |

## Results

ResNet-50 trained from scratch (no pretrained weights) for 50 epochs with batch size 32, using the configuration above. Evaluation is on the official `test.txt` split.

| Metric | Value |
|--------|-------|
| Best test accuracy | **38.95%** (epoch 49) |
| Final test accuracy (epoch 50) | 38.76% |
| Lowest test loss | 2.481 (epoch 28) |
| Final test loss | 2.670 |
| Random-guess reference | 2.7% accuracy, loss ≈ 3.61 |

Effect of the training setup, measured at epoch 10:

| Setup | Test loss | Test accuracy |
|-------|:---------:|:-------------:|
| Baseline: no normalization or augmentation, constant learning rate `1e-3` | 3.639 | 6.65% |
| Current: normalization, augmentation, learning rate `1e-2` with cosine decay, weight decay `1e-4` | 3.663 | 11.97% |

**Notes**

- The model overfits: the logged training loss (last batch of each epoch) drops to about 0.3–0.6, while the test loss stops improving after roughly epoch 28 and stays around 2.6–2.7.
- The test split is used both to pick the best checkpoint and to report accuracy, so the numbers are slightly optimistic. A dedicated validation split is on the roadmap.
- Training a 25M-parameter network from scratch on about 3.7k images is data-limited; transfer learning from pretrained weights is expected to score far higher.

## Roadmap

**Current status:** the full pipeline (data → model → training → logging → checkpointing) works end to end for 37-class breed classification, and a full 50-epoch run reaches about 39% test accuracy. The main open problem is overfitting.

**Done**

- [x] Custom `Dataset` and `DataLoader` for Oxford-IIIT Pet (37 classes)
- [x] ResNet-50 implemented from scratch
- [x] Training loop with TensorBoard logging and best-model checkpointing
- [x] Installable package (`pip install -e .`) and minimal `requirements.txt`
- [x] Input normalization and data augmentation (ImageNet mean/std, random resized crop, horizontal flip)
- [x] Cosine learning-rate schedule and weight decay
- [x] Full 50-epoch training run with normalization and augmentation (about 39% test accuracy)
- [x] Device-agnostic training and evaluation (CPU fallback)
- [x] Resumable training (latest and best checkpoints with optimizer and scheduler state)
- [x] Per-epoch confusion matrix logged to TensorBoard
- [x] Binary (cat vs dog) and 37-breed modes selectable with `--numclass`

**Next**

- [ ] Reduce overfitting (stronger augmentation, label smoothing, Mixup/CutMix)
- [ ] Dedicated validation split
- [ ] Show breed names instead of class indices on the confusion matrix axes
- [ ] Inference script for single images
- [ ] Benchmark against `torchvision` pretrained ResNet-50
- [ ] Unit tests

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2015
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — Parkhi et al., 2012
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Documentation](https://pytorch.org/vision/stable/index.html)
