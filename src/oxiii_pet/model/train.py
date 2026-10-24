import os
from argparse import ArgumentParser
from pathlib import Path

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose, Resize, RandomResizedCrop, RandomHorizontalFlip,
    ToImage, ToDtype, Normalize,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from oxiii_pet.model.models import Resnet50
from oxiii_pet.dataset.dataset_oxiiit_pet import OxfordData


def get_args():
    parser = ArgumentParser(description="CNN IN Oxiii Pet")
    parser.add_argument("--epochs","-e", type=int, default=10, help="number of epochs")
    parser.add_argument("--images_size","-i", type=int, default=224, help="size of images")
    parser.add_argument("--batch_size","-b", type=int, default=8, help="batch size")
    parser.add_argument("--dir","-d", type=str, default=os.getenv("DATA_DIR", "data"), help="data path")
    parser.add_argument("--logging","-l", type=str, default="tensorboard")
    parser.add_argument("--trained_model","-t", type=str, default="trained_model")
    parser.add_argument("--checkpoint","-c", type=str, default=None)
    parser.add_argument("--num_class","-n", type=int, default=37)
    parser.add_argument("--clip","-cl", type=float, default=0.1)
    args = parser.parse_args()
    return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    writer.add_figure('confusion_matrix', figure, epoch)

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # torch.autograd.set_detect_anomaly(True)
    num_epochs = args.epochs
    num_class = args.num_class
    DATA_DIR = args.dir
    image_size = args.images_size
    # Specify your target directory path
    folder_path = Path(args.trained_model)
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = Compose([
        RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        RandomHorizontalFlip(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        normalize,
    ])
    test_transform = Compose([
        Resize((image_size, image_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        normalize,
    ])
    train_data = OxfordData(num_class,DATA_DIR, train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    test_data = OxfordData(num_class,DATA_DIR, train=False, transform=test_transform)
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=False
    )
    writer = SummaryWriter(args.logging)
    model = Resnet50(num_class).to(device)
    initialize_weights(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # desc_text = "\033[1;36m[Epoch 1/10] Training Model...\033[0m"

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_val_accuracy = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_val_accuracy = 0.0
    num_iters = len(train_dataloader)
    for epoch in range(start_epoch, num_epochs):
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
            writer.add_scalar("Train/Loss", loss_value.item(), i + (epoch * num_iters))
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
        class_names = ['cat', 'dog'] if num_class == 2 else [str(i) for i in range(num_class)]
        plot_confusion_matrix(writer, confusion_matrix(all_labels.cpu().numpy(), all_predictions.cpu().numpy()), class_names, epoch)

        val_accuracy = (all_predictions == all_labels).float().mean().item()
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_val_accuracy
        }
        torch.save(checkpoint, "{}/last_model.pth".format(args.trained_model))
        if is_best:
            torch.save(checkpoint, "{}/best_model.pth".format(args.trained_model))
            print(f"Saved best model! Val Accuracy: {best_val_accuracy:.4f}")

        writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Val Accuracy: {val_accuracy:.4f}"
            f"Val Loss: {val_loss}"
        )
    writer.close()

if __name__ == '__main__':
    main()