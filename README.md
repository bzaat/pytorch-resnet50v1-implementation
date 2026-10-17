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
- [Roadmap](#roadmap)
- [License](#license)
- [References](#references)

## Overview

This project implements the ResNet-50 architecture from scratch in PyTorch and trains it on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (37 cat and dog breeds). It is a learning-oriented codebase focused on three things: understanding CNN architectures, building a clean data pipeline, and setting up a reproducible training workflow.

## Features

- **ResNet-50 from scratch:** bottleneck residual blocks, projection shortcuts, and a configurable classification head, with no dependency on `torchvision.models`.
- **Custom `Dataset`:** reads the official annotation files, builds image paths, converts labels to zero-based indices, and loads images as RGB with Pillow.
- **Train / evaluation splits:** `trainval.txt` for training and `test.txt` for evaluation.
- **Training script with CLI:** configurable epochs, batch size, data directory and log directory.
- **Experiment tracking:** loss and accuracy curves logged to TensorBoard.
- **Best-model checkpointing:** the checkpoint with the highest evaluation accuracy is saved automatically.
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
2. **Select columns:** keeps the image name and the breed class ID, and drops the remaining columns.
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

Or use the dataset directly in Python:

```python
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from oxiii_pet.dataset.dataset_oxiiit_pet import OxfordData

transform = Compose([Resize((224, 224)), ToImage(), ToDtype(torch.float32, scale=True)])

train_set = OxfordData("data", train=True, transform=transform)
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

After every epoch the model is evaluated on the test split. Whenever accuracy improves, a checkpoint is written to `best_model.pth`.

### 3. Monitor training

```bash
tensorboard --logdir tensorboard
```

Logged scalars: `Train/Loss`, `Validation/Loss`, `Validation/Accuracy`.

### 4. Load a trained checkpoint

```python
import torch
from oxiii_pet.model.models import Resnet50

model = Resnet50(num_classes=37)
checkpoint = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Best accuracy {checkpoint['best_val_accuracy']:.4f} at epoch {checkpoint['epoch']}")
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
| Output classes | 37 |

## Roadmap

**Current status:** the full pipeline (data → model → training → logging → checkpointing) works end to end for 37-class breed classification. A first baseline run without input normalization or augmentation was recorded; the training script now includes both, plus a higher initial learning rate with cosine decay and weight decay. The next step is to re-train and compare against that baseline.

Baseline run without normalization or augmentation (batch size 32, learning rate `1e-3`, first 10 epochs of a partial run):

| Epoch | Val loss | Val accuracy |
|:-----:|:--------:|:------------:|
| 1 | 3.955 | 3.1% |
| 5 | 3.738 | 3.5% |
| 10 | 3.639 | 6.7% |

Random guessing over 37 classes gives about 2.7% accuracy and a loss of about 3.61.

**Done**

- [x] Custom `Dataset` and `DataLoader` for Oxford-IIIT Pet (37 classes)
- [x] ResNet-50 implemented from scratch
- [x] Training loop with TensorBoard logging and best-model checkpointing
- [x] Installable package (`pip install -e .`) and minimal `requirements.txt`
- [x] Input normalization and data augmentation (ImageNet mean/std, random resized crop, horizontal flip)
- [x] Cosine learning-rate schedule and weight decay

**In progress**

- [ ] Re-train with normalization and augmentation and compare against the baseline

**Next**

- [ ] Device-agnostic training and evaluation (CPU fallback)
- [ ] Dedicated validation split
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
