<h1 align="center">ResNet-50 in PyTorch</h1>

<p align="center">
  ResNet-50 for pet breed classification on the Oxford-IIIT Pet dataset: a from-scratch implementation and an ImageNet-pretrained transfer-learning variant.
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
- [Transfer Learning](#transfer-learning)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Project Status](#project-status)
- [License](#license)
- [References](#references)

## Overview

This project implements the ResNet-50 architecture from scratch in PyTorch and trains it on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (37 cat and dog breeds). It is a learning-oriented codebase focused on three things: understanding CNN architectures, building a clean data pipeline, and setting up a reproducible training workflow.

Two models are included so they can be compared on the same data: a ResNet-50 trained from scratch (`oxiii_pet.model`), and a `torchvision` ResNet-50 pretrained on ImageNet and fine-tuned on the pet dataset (`oxiii_pet.transfer_learning`).

| Model (37 breeds, official test split) | Best test accuracy | Final epoch |
|----------------------------------------|:------------------:|:-----------:|
| ResNet-50 from scratch, 50 epochs | 38.95% | 38.76% |
| Pretrained ResNet-50, fine-tuned, 10 epochs | **93.19%** | 93.13% |

## Features

- **ResNet-50 from scratch:** bottleneck residual blocks, projection shortcuts, and a configurable classification head, with no dependency on `torchvision.models` (`oxiii_pet/model/models.py`).
- **Transfer learning:** an ImageNet-pretrained `torchvision` ResNet-50 (V1 or V2 weights) with a new classification head; only the head and `layer4` are trained, and the rest of the backbone stays frozen.
- **Custom `Dataset`:** reads the official annotation files, builds image paths, converts labels to zero-based indices, and loads images as RGB with Pillow.
- **Train / evaluation splits:** `trainval.txt` for training and `test.txt` for evaluation.
- **Two task modes:** 37-breed classification (default) or binary cat-vs-dog classification, selected with `--num_class`.
- **Training script with CLI:** configurable epochs, batch size, data directory and log directory.
- **Experiment tracking:** loss and accuracy curves, plus a per-epoch confusion matrix, logged to TensorBoard.
- **Checkpointing and resume:** the latest and the best checkpoints (model, optimizer and scheduler state) are saved automatically, and training can be resumed from either.
- **CPU and GPU support:** training and evaluation run on CUDA when available and fall back to the CPU otherwise.
- **Installable package managed with [uv](https://docs.astral.sh/uv/):** `uv sync` installs the locked dependencies and the project in editable mode, so `oxiii_pet` is importable from anywhere.

## Project Structure

```text
.
├── LICENSE
├── pyproject.toml
├── uv.lock
└── src/
    └── oxiii_pet/
        ├── dataset/
        │   ├── dataset_oxiiit_pet.py     # OxfordData: custom Dataset
        │   └── dataloader_oxiii_pet.py   # DataLoader smoke test
        ├── model/                        # ResNet-50 from scratch
        │   ├── models.py                 # Bottleneck block and Resnet50
        │   ├── train.py                  # Training and evaluation entry point
        │   └── test.py                   # Single-image inference (experimental)
        └── transfer_learning/            # Pretrained ResNet-50 fine-tuning
            ├── transfer_learning.py      # MyModel: pretrained backbone + new head
            ├── train.py                  # Fine-tuning and evaluation entry point
            └── test.py                   # Single-image inference (experimental)
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

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (it can also download a suitable Python for you)
- Python 3.10 or newer
- (Optional) An NVIDIA GPU with CUDA for faster training

### Installation

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh     # Windows (PowerShell): powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone the repository
git clone https://github.com/bzaat/pytorch-resnet50v1-implementation.git
cd pytorch-resnet50v1-implementation

# 3. Create .venv and install the locked dependencies and the project (editable)
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates a `.venv` in the project folder and installs everything into it. There is no need to activate the environment: prefix commands with `uv run` (as in the examples below), or activate `.venv` yourself if you prefer.

> **GPU users:** the default PyPI build of PyTorch on Linux includes CUDA support. For a CPU-only build or a specific CUDA version, follow uv's [PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

Verify the installation:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
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
uv run python src/oxiii_pet/dataset/dataloader_oxiii_pet.py
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
uv run python -m oxiii_pet.model.train --epochs 50 --batch_size 32 --dir data
```

| Argument | Short | Default | Description |
|----------|:-----:|---------|-------------|
| `--epochs` | `-e` | `10` | Number of training epochs |
| `--batch_size` | `-b` | `8` | Batch size for both splits |
| `--images_size` | `-i` | `224` | Input image size |
| `--dir` | `-d` | `$DATA_DIR` or `data` | Dataset root directory |
| `--logging` | `-l` | `tensorboard` | Directory for TensorBoard logs |
| `--trained_model` | `-t` | `trained_model` | Directory where checkpoints are saved |
| `--checkpoint` | `-c` | none | Checkpoint to resume training from |
| `--num_class` | `-n` | `37` | Number of classes: `37` for breeds, `2` for cat vs dog |

After every epoch the model is evaluated on the test split. Two checkpoints are written to the `--trained_model` directory: `last_model.pth` after every epoch, and `best_model.pth` whenever test accuracy improves. Each checkpoint stores the epoch, model, optimizer and scheduler state, and the best accuracy.

To resume an interrupted run, pass the checkpoint and keep the same `--epochs` value so the learning-rate schedule stays consistent:

```bash
uv run python -m oxiii_pet.model.train --epochs 50 --batch_size 32 --checkpoint trained_model/last_model.pth
```

To train the binary cat-vs-dog task instead of the 37 breeds, pass `--num_class 2`:

```bash
uv run python -m oxiii_pet.model.train --epochs 50 --batch_size 32 --num_class 2
```

### 3. Monitor training

```bash
uv run tensorboard --logdir tensorboard
```

Logged scalars: `Train/Loss`, `Validation/Accuracy`.

Logged images: `confusion_matrix`, the confusion matrix of the test split, written once per epoch by `plot_confusion_matrix` in `train.py` (scikit-learn computes the matrix and matplotlib draws it). Open the **Images** tab and drag the slider to see how the errors change during training. Rows are the true classes and columns are the predicted classes; each cell shows the fraction of that true class predicted as the column class, so the diagonal is the per-class recall.

### 4. Load a trained checkpoint

```python
import torch
from oxiii_pet.model.models import Resnet50

model = Resnet50(num_classes=37)  # use 2 if you trained with --num_class 2
checkpoint = torch.load("trained_model/best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
```

## Transfer Learning

`oxiii_pet.transfer_learning` fine-tunes a `torchvision` ResNet-50 that was pretrained on ImageNet-1K, using the same data pipeline, transforms and optimizer settings as the from-scratch model.

**Design** (`transfer_learning/transfer_learning.py`)

| Part | Setting |
|------|---------|
| Backbone | `torchvision.models.resnet50` with `IMAGENET1K_V2` weights (default) or `IMAGENET1K_V1` (`--version 1`) |
| Head | the original `fc` is removed and replaced by `Dropout(0.5)` → `Linear(2048, 1024)` → `ReLU` → `Linear(1024, num_classes)` |
| Trainable | the new head and `layer4`; the stem and `layer1`–`layer3` are frozen (`requires_grad=False`) |
| BatchNorm | `MyModel.train()` keeps the whole pretrained backbone, including `layer4`, in eval mode, so its BatchNorm layers keep using the ImageNet running statistics instead of re-estimating them from batches of the pet data; only their scale and shift are trained in `layer4` |

Freezing parameters with `requires_grad=False` does not stop BatchNorm from updating its running mean and variance, because those are buffers updated in the forward pass whenever the layer is in train mode. This is why `train()` is overridden.

**Train**

The script saves to `--trained_model` but does not create the folder, so create it first:

```bash
mkdir -p trained_model
uv run python src/oxiii_pet/transfer_learning/train.py -n 37 --epoch 10 --batch_size 32
```

| Argument | Short | Default | Description |
|----------|:-----:|---------|-------------|
| `--data_path` | `-d` | `$DATA_DIR` or `data` | Dataset root directory |
| `--epoch` | `-e` | `10` | Number of training epochs |
| `--image_size` | `-s` | `224` | Input image size |
| `--num_classes` | `-n` | `37` | Number of classes: `37` for breeds, `2` for cat vs dog |
| `--batch_size` | `-b` | `32` | Batch size for both splits |
| `--version` | `-v` | `2` | ImageNet-1K weights: `1` = V1, `2` = V2 |
| `--trained_model` | `-t` | `trained_model` | Directory for the saved weights |

After every epoch the model is evaluated on the test split and `fine_tuning_model.pth` (the model `state_dict` only, overwritten each epoch) is saved. Unlike the from-scratch script, this one does not log to TensorBoard, keep a best checkpoint or support resuming.

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
| Output classes | 37 breeds by default, or 2 (cat vs dog) with `--num_class 2` |

The transfer-learning script uses the same optimizer, schedule, transforms and DataLoader settings (see [Transfer Learning](#transfer-learning)).

## Results

### From scratch (37 breeds)

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
- The test split is used both to pick the best checkpoint and to report accuracy, so the numbers are slightly optimistic; there is no separate validation split.
- Training a 25M-parameter network from scratch on about 3.7k images is data-limited; starting from pretrained weights removes most of that limit (see the transfer-learning results below).

### Transfer learning (37 breeds)

Fine-tuning the pretrained ResNet-50 (`IMAGENET1K_V2`, trainable head + `layer4`) for 10 epochs with batch size 32 and the default settings of the transfer-learning script, on the same official `test.txt` split.

| Metric | Value |
|--------|-------|
| Best test accuracy | **93.19%** (epoch 8) |
| Final test accuracy (epoch 10) | 93.13% |
| Lowest test loss | 0.249 (epoch 8) |
| Final test loss | 0.268 |
| From-scratch reference | 38.95% best, 2.481 lowest test loss |

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|
| Test accuracy (%) | 77.68 | 86.73 | 88.03 | 88.23 | 87.00 | 91.03 | 92.45 | 93.19 | 93.16 | 93.13 |
| Test loss | 0.722 | 0.408 | 0.375 | 0.438 | 0.493 | 0.320 | 0.259 | 0.249 | 0.269 | 0.268 |

One epoch takes about 17–19 s on the machine used for this run.

**Notes**

- Pretraining on ImageNet raises 37-breed test accuracy from about 39% to about 93%, and one epoch already beats the 50-epoch from-scratch run (77.7% vs 38.95%).
- The best epoch (8) is chosen by test accuracy, which is slightly optimistic. The script saves only the last epoch, so the saved `fine_tuning_model.pth` is the epoch-10 model (93.13%).
- The training loss (last batch of each epoch) falls to about 0.003–0.03 while the test loss stays near 0.25–0.27 from epoch 8 on, so some overfitting remains.
- The same setup also reaches 99.8–99.95% on the binary cat-vs-dog task (`-n 2`, logged for epochs 2–6). That task is easy for ImageNet features, so it mostly serves as a sanity check.

## Project Status

The project is complete: data pipeline, ResNet-50 from scratch, training with TensorBoard logging and checkpointing, and a transfer-learning variant. On the 37-breed task the from-scratch model reaches about 39% test accuracy and the fine-tuned pretrained model about 93%.

**Done**

- [x] Custom `Dataset` and `DataLoader` for Oxford-IIIT Pet (37 classes)
- [x] ResNet-50 implemented from scratch
- [x] Training loop with TensorBoard logging and best-model checkpointing
- [x] Installable package with dependencies declared in `pyproject.toml` and locked with uv (`uv.lock`)
- [x] Input normalization and data augmentation (ImageNet mean/std, random resized crop, horizontal flip)
- [x] Cosine learning-rate schedule and weight decay
- [x] Full 50-epoch training run with normalization and augmentation (about 39% test accuracy)
- [x] Device-agnostic training and evaluation (CPU fallback)
- [x] Resumable training (latest and best checkpoints with optimizer and scheduler state)
- [x] Per-epoch confusion matrix logged to TensorBoard
- [x] Binary (cat vs dog) and 37-breed modes selectable with `--num_class`
- [x] Transfer learning from ImageNet-pretrained ResNet-50 (frozen backbone except `layer4`, BatchNorm kept in eval mode): 93.19% best test accuracy on 37 breeds versus 38.95% from scratch

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2015
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — Parkhi et al., 2012
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Documentation](https://pytorch.org/vision/stable/index.html)
- [Torchvision ResNet-50 weights](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
