from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class OxfordData(Dataset):
    def __init__(self, num_class, root, train= True, transform = None):
        self.transform = transform
        # print(os.path.join(root,'annotations', 'trainval.txt'))
        if train:
            df = pd.read_csv(os.path.join(root,'annotations', 'trainval.txt'),sep=r'\s+', header= None)
        else:
            df = pd.read_csv(os.path.join(root, 'annotations', 'test.txt'),sep=r'\s+', header= None)
        self.image_path = [os.path.join(root, "images", str(name) + ".jpg") for name in df[0].tolist()]
        if num_class ==2:
            df = df.drop(df.columns[[1, 3]], axis=1)
            self.labels = (df[2] - 1).tolist()
        elif num_class ==37:
            df = df.drop(df.columns[[2, 3]], axis=1)
            self.labels = (df[1] - 1).tolist()
        else:
            raise "Enter 2 or 37 class number"


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = self.image_path[item]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label

def main():
    ds = OxfordData("../../../data", train=True)
    print(min(ds.labels), max(ds.labels), len(set(ds.labels)))  # 0 36 37
if __name__ == '__main__':
    main()
