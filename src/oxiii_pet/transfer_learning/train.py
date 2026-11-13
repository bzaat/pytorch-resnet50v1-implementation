from argparse import ArgumentParser
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from oxiii_pet.dataset.dataset_oxiiit_pet import OxfordData
from torchvision.transforms.v2 import (
    Compose, Resize, RandomResizedCrop, RandomHorizontalFlip,
    ToImage, ToDtype, Normalize,
)

from oxiii_pet.transfer_learning.transfer_learning import MyModel
from oxiii_pet.utils.run_utils import (
    MetricsCSV, default_run_dir, get_class_names, plot_confusion_matrix,
    run_epoch, set_seed, setup_run,
)

def get_args():
    parser = ArgumentParser("TRANSFER LEARNING")
    parser.add_argument('--data_path','-d',type=str,default=os.getenv("DATA_DIR", "data"),help='Data path')
    parser.add_argument('--epoch','-e',type=int,default=10,help='Epoch')
    parser.add_argument('--image_size','-s',type=int,default=224,help='Image size')
    parser.add_argument('--num_classes','-n',type=int,default=37,help='Number of class')
    parser.add_argument('--batch_size','-b',type=int,default=32,help='Batch size')
    parser.add_argument("--version", "-v",type=int, default=2,help="Version of ImageNet1K Dataset: 1 = V1, 2 = V2")
    parser.add_argument("--run_dir", "-r", type=str, default=None,
                        help="run folder for logs and checkpoints (default runs/<date>_transfer_<n>cls)")
    parser.add_argument("--trained_model","-t", type=str, default=None, help="checkpoint dir (default <run_dir>/checkpoints)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", "-w", type=int, default=4)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args.run_dir = args.run_dir or default_run_dir(f"transfer_{args.num_classes}cls")
    args.trained_model = args.trained_model or os.path.join(args.run_dir, "checkpoints")
    Path(args.trained_model).mkdir(parents=True, exist_ok=True)
    logger, _ = setup_run(args.run_dir, args, "transfer")
    generator, worker_init_fn = set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    DATA_DIR = args.data_path
    num_classes = args.num_classes
    image_size = args.image_size
    num_epochs = args.epoch

    model = MyModel(args.num_classes, args.version).to(device)
    for name, param in model.named_parameters():
        if not "fc1." in name and not "fc2." in name and not "layer4."  in name:
            param.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d", n_trainable, n_total)
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
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False
    )
    class_names = get_class_names(num_classes, DATA_DIR)
    logger.info("Train samples: %d | Test samples: %d", len(train_dataset), len(test_dataset))
    writer = SummaryWriter(os.path.join(args.run_dir, "tensorboard"))
    metrics = MetricsCSV(os.path.join(args.run_dir, "metrics.csv"))
    num_iters = len(train_dataloader)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy, best_epoch = 0.0, None
    run_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, _, _ = run_epoch(
            model, train_dataloader, criterion, device, optimizer=optimizer,
            desc=f"Train {epoch + 1}/{num_epochs}", writer=writer, global_step=epoch * num_iters)
        scheduler.step()
        val_loss, val_accuracy, all_predictions, all_labels = run_epoch(
            model, test_dataloader, criterion, device, desc=f"Val   {epoch + 1}/{num_epochs}")
        plot_confusion_matrix(writer, confusion_matrix(all_labels.numpy(), all_predictions.numpy(),
                                                       labels=list(range(num_classes))), class_names, epoch)

        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy, best_epoch = val_accuracy, epoch + 1
        checkpoint = {
            "epoch": epoch + 1, "model": model.state_dict(), "val_acc": val_accuracy,
            "best_acc": best_val_accuracy, "best_epoch": best_epoch,
            "num_class": num_classes, "version": args.version, "class_names": class_names,
        }
        torch.save(checkpoint, os.path.join(args.trained_model, "last_model.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(args.trained_model, "best_model.pth"))

        epoch_time = time.time() - epoch_start
        writer.add_scalar("Train/EpochLoss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)
        writer.add_scalar("LR", lr, epoch)
        metrics.append(epoch=epoch + 1, lr=lr, train_loss=round(train_loss, 6), train_acc=round(train_acc, 6),
                       val_loss=round(val_loss, 6), val_acc=round(val_accuracy, 6),
                       epoch_time_s=round(epoch_time, 1), is_best=int(is_best))
        logger.info(
            "Epoch %d/%d | lr %.3e | train loss %.4f acc %.4f | val loss %.4f acc %.4f | %.1fs%s",
            epoch + 1, num_epochs, lr, train_loss, train_acc, val_loss, val_accuracy, epoch_time,
            " | new best, saved best_model.pth" if is_best else "")
    writer.close()

    summary = {
        "best_val_acc": best_val_accuracy, "best_epoch": best_epoch,
        "final_val_acc": val_accuracy, "final_val_loss": val_loss, "final_epoch": num_epochs,
        "train_time_s_this_session": round(time.time() - run_start, 1),
        "note": "val = official test.txt split (also used to select best_model.pth)",
    }
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary))

if __name__ == '__main__':
    main()
