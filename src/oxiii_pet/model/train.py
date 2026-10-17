import os
from argparse import ArgumentParser

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose, Resize, RandomResizedCrop, RandomHorizontalFlip,
    ToImage, ToDtype, Normalize,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from oxiii_pet.model.models import Resnet50
from oxiii_pet.dataset.dataset_oxiiit_pet import OxfordData


def get_args():
    parser = ArgumentParser(description="CNN IN Oxiii Pet")
    parser.add_argument("--epochs","-e", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size","-b", type=int, default=8, help="batch size")
    parser.add_argument("--dir","-d", type=str, default=os.getenv("DATA_DIR", "data"), help="data path")
    parser.add_argument("--logging","-l", type=str, default="tensorboard")
    args = parser.parse_args()
    return args

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu"
            )

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

def main():
    args = get_args()
    num_epochs = args.epochs
    DATA_DIR = args.dir

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = Compose([
        RandomResizedCrop(224, scale=(0.6, 1.0)),
        RandomHorizontalFlip(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        normalize,
    ])
    test_transform = Compose([
        Resize((224, 224)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        normalize,
    ])
    train_data = OxfordData(DATA_DIR, train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    test_data = OxfordData(DATA_DIR, train=False, transform=test_transform)
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=False
    )
    writer = SummaryWriter(args.logging)
    model = Resnet50()
    initialize_weights(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # desc_text = "\033[1;36m[Epoch 1/10] Training Model...\033[0m"
    num_iters = len(train_dataloader)
    if torch.cuda.is_available():
        model.cuda()
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        # progress_bar = tqdm(train_dataloader,desc=desc_text,colour="green")
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", colour="green")
        for i, (images, labels) in enumerate(progress_bar):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # forward pass
            output = model(images)
            loss_value = criterion(output, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {}".format(epoch + 1, num_epochs, i, num_iters, loss_value))
            writer.add_scalar("Train/Loss", loss_value, i + (epoch * num_iters))
            # backward & optimize
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        # model.eval()
        # all_predictions = []
        # all_labels = []
        # for i, (images, labels) in enumerate(test_dataloader):
        #     all_labels.append(labels)
        #     if torch.cuda.is_available():
        #         images = images.cuda()
        #         labels = labels.cuda()
        #     with torch.no_grad():
        #         predictions = model(images)
        #         all_predictions.append(predictions)
        #         loss_value = criterion(predictions, labels)
        scheduler.step()
        model.eval()

        all_predictions = []
        all_labels = []
        total_val_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.cuda()
                labels = labels.cuda()

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
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_val_accuracy": best_val_accuracy,
            }, "best_model.pth")
            print(f"Saved best model! Val Accuracy: {best_val_accuracy:.4f}")

        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )
    writer.close()

if __name__ == '__main__':
    main()