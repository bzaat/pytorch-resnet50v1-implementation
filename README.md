# ResNet-50 Image Classification with PyTorch

An image classification project built with PyTorch, using the Oxford-IIIT Pet dataset to practice CNN architecture implementation, dataset preparation, and model training.

## Overview

This project implements a custom PyTorch data pipeline for the Oxford-IIIT Pet dataset. The dataset class loads images, reads annotations, converts labels to zero-based indices, and optionally applies image transformations.

**Current features**
- Custom PyTorch `Dataset` implementation.
- Image loading and RGB conversion using Pillow.
- Annotation parsing with Pandas.
- Optional image transformations.
- Separate dataset splits for training and evaluation.

## Tech Stack

- PyTorch
- Torchvision
- Pillow (PIL)

## Dataset

This project uses the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which contains pet images and annotation files.

Expected dataset directory structure:

```text
data/oxford-iiit-pet/
├── annotations/
│   ├── trainval.txt
│   └── test.txt
└── images/
    ├── Abyssinian_1.jpg
    ├── Abyssinian_2.jpg
    └── ...
```

The `root` argument should point to the directory containing both `annotations/` and `images/`.

> Update `data/oxford-iiit-pet/` if your actual dataset location differs.

## Dataset Pipeline

The custom `OxfordData` class inherits from `torch.utils.data.Dataset`.

1. **Load annotations:** Read `trainval.txt` when `train=True`, or `test.txt` when `train=False`.
2. **Select relevant columns:** Remove the annotation columns that are not required for image loading and classification.
3. **Build image paths:** Construct each image path using the image identifier and the `.jpg` extension.
4. **Map labels:** Convert the original class labels from 1-based indices to 0-based indices using `label - 1`.
5. **Load images:** Open each image with Pillow and convert it to RGB.
6. **Apply transformations:** Apply the optional `transform` if one is provided.
7. **Return samples:** Each sample is returned as an `(image, label)` pair.


**Note:** The import path assumes that your project package is installed or configured correctly. Adjust it to match your repository structure if necessary.

## Current Status

The project is under development. The current focus is on understanding CNN architectures and building a reproducible image classification pipeline with PyTorch.

Planned improvements may include:
- Training and evaluation metrics.
- Model checkpointing.
- Experiment tracking.
- Data augmentation.
- Comparison between custom and pretrained CNN models.

## References

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Documentation](https://pytorch.org/vision/stable/index.html)
