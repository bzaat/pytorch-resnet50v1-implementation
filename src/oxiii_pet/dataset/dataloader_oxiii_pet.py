"""DataLoader smoke test: uv run python -m oxiii_pet.dataset.dataloader_oxiii_pet"""
import os

from torch.utils.data import DataLoader
import torch
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from oxiii_pet.dataset.dataset_oxiiit_pet import OxfordData

if __name__ == '__main__':
    DATA_DIR = os.getenv("DATA_DIR", "data")
    transform = Compose([Resize((200, 200)), ToImage(), ToDtype(torch.float32, scale=True)])
    for split in ("train", "val", "test"):
        data = OxfordData(37, DATA_DIR, split=split, transform=transform)
        loader = DataLoader(dataset=data, batch_size=16, num_workers=4, drop_last=True, shuffle=True)
        images, labels = next(iter(loader))
        print(f"{split}: {len(data)} images, batch {tuple(images.shape)}, labels {labels.tolist()}")
