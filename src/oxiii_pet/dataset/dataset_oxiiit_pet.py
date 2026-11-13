import json
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_SPLIT_FILE = os.path.join("splits", "trainval_80_20_seed0.json")


class OxfordData(Dataset):
    """Oxford-IIIT Pet.

    split = "train" / "val": the two parts of the official trainval.txt, as listed in split_file
            (a fixed stratified 80/20 split, see oxiii_pet.dataset.make_split).
    split = "test": the official test.txt.
    num_class = 37 (breed) or 2 (species: cat / dog).
    """

    def __init__(self, num_class, root, split="train", transform=None, split_file=DEFAULT_SPLIT_FILE):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test'")
        self.transform = transform
        self.split = split
        ann = "test.txt" if split == "test" else "trainval.txt"
        df = pd.read_csv(os.path.join(root, "annotations", ann), sep=r"\s+", header=None)
        df[0] = df[0].astype(str)
        if split != "test":
            with open(split_file) as f:
                keep = set(json.load(f)[split])
            df = df[df[0].isin(keep)].reset_index(drop=True)
            if len(df) != len(keep):
                raise ValueError(f"{len(keep) - len(df)} images of the {split} split are missing from {ann}")
        self.image_path = [os.path.join(root, "images", name + ".jpg") for name in df[0].tolist()]
        if num_class == 2:
            self.labels = (df[2] - 1).tolist()  # species: 1 = cat, 2 = dog
        elif num_class == 37:
            self.labels = (df[1] - 1).tolist()  # breed class id 1..37
        else:
            raise ValueError("num_class must be 2 or 37")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[item]


def main():
    for split in ("train", "val", "test"):
        ds = OxfordData(37, os.getenv("DATA_DIR", "data"), split=split)
        print(split, len(ds), min(ds.labels), max(ds.labels), len(set(ds.labels)))


if __name__ == '__main__':
    main()
