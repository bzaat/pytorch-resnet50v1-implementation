from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ToTensor, Resize, Compose
import os
from dataset_oxiiit_pet import OxfordData

if __name__ == '__main__':
    DATA_DIR = os.getenv("DATA_DIR", "data")

    transform = Compose([
        Resize((200,200)),
        ToTensor()
    ])
    training_data = OxfordData(DATA_DIR, train= True, transform = transform)
    testing_data = OxfordData(DATA_DIR, train= False, transform = transform)
    training_dataloader = DataLoader(
        dataset=training_data,
        batch_size=16,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    for images, labels in training_dataloader:
        print(images.shape)
        print(labels)