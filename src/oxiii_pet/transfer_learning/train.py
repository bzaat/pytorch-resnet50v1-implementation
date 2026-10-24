from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from oxiii_pet.dataset.dataset_oxiiit_pet import OxfordData
from torchvision.transforms.v2 import (
    Compose, Resize, RandomResizedCrop, RandomHorizontalFlip,
    ToImage, ToDtype, Normalize,
)
from tqdm import tqdm


from oxiii_pet.transfer_learning.transfer_learning import MyModel

def get_args():
    parser = ArgumentParser("TRANSFER LEARNING")
    parser.add_argument('--data_path','-d',type=str,default=os.getenv("DATA_DIR", "data"),help='Data path')
    parser.add_argument('--epoch','-e',type=int,default=10,help='Epoch')
    parser.add_argument('--image_size','-s',type=int,default=224,help='Image size')
    parser.add_argument('--num_classes','-n',type=int,default=37,help='Number of class')
    parser.add_argument('--batch_size','-b',type=int,default=32,help='Batch size')
    parser.add_argument("--version", "-v",type=int, default=2,help="Version of ImageNet1K Dataset: 1 = V1, 2 = V2")
    parser.add_argument("--trained_model","-t", type=str, default="trained_model")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    DATA_DIR = args.data_path
    num_classes = args.num_classes
    image_size = args.image_size
    num_epochs = args.epoch
    train_transform = None
    
    model = MyModel(args.num_classes, args.version).to(device)
    for name, param in model.named_parameters():
        if not "fc1." in name and not "fc2." in name and not "layer4."  in name:
            param.requires_grad = False
        print(name, param.requires_grad)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = Compose([
        RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        RandomHorizontalFlip(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        normalize
    ])
    test_transform = Compose([
        Resize((image_size, image_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        normalize
    ])
    train_dataset = OxfordData(num_classes,DATA_DIR, train=True, transform=train_transform)
    test_dataset = OxfordData(num_classes,DATA_DIR, train=False, transform=test_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=False
    )
    num_iters = len(train_dataloader)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        # progress_bar = tqdm(train_dataloader,desc=desc_text,colour="green")
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", colour="green")
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            output = model(images)
            loss_value = criterion(output, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {}".format(epoch + 1, num_epochs, i, num_iters, loss_value))
            # writer.add_scalar("Train/Loss", loss_value.item(), i + (epoch * num_iters))
            # backward & optimize
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        scheduler.step()
        model.eval()
        all_predictions = []
        all_labels = []
        total_val_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                loss_value = criterion(predictions, labels)

                batch_size = labels.size(0)
                total_val_loss += loss_value.item() * batch_size
                num_samples += batch_size

                all_predictions.append(predictions.argmax(dim=1))
                all_labels.append(labels)

        val_loss = total_val_loss / num_samples
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        val_accuracy = (all_predictions == all_labels).float().mean().item()
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Val Loss: {val_loss}"
        )
        torch.save(model.state_dict(), "{}/fine_tuning_model.pth".format(args.trained_model))
if __name__ == '__main__':
    main()